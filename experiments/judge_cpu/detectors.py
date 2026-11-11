"""Detector registry. Every detector maps a text to one float: higher = more likely AI.

Classifiers with short windows are run on consecutive token windows and the
per-window AI logit is averaged, weighted by window length (no truncation).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

WEIGHTS = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "vocab*", "merges*"]


def materialize(model: nn.Module) -> nn.Module:
    """Copy weights into anonymous memory.

    transformers 5 keeps safetensors weights memory-mapped. Under page-cache pressure on this box (slow
    network disk), the pages get evicted and re-read on every forward pass: one 1.5B Binoculars run read
    68 GB from disk and sat at ~80% iowait. Cloning once at load time fixes it.
    """
    with torch.no_grad():
        for t in list(model.parameters()) + list(model.buffers()):
            t.data = t.data.clone()
    return model


PINNED_REVISIONS = {
    "ShantanuT01/vanguard-ai-text-detector": "823061be63b90f2b42f64ac1e1f82772e872533b",
    "wasitaigeneratedcom/tropa-mini-ai-text-detector": "10fecd0856ed65ddfe92b88416b0a1e1350d8fb8",
}


def fetch(repo: str, bin_ok: bool = False) -> str:
    pats = WEIGHTS + (["pytorch_model.bin"] if bin_ok else [])
    return snapshot_download(repo, revision=PINNED_REVISIONS.get(repo), allow_patterns=pats)


@dataclass
class ClsSpec:
    name: str
    repo: str
    kind: str  # seqcls | desklib | tropa | superannotate
    window: int  # tokens per window incl. special tokens
    ai_index: int | None = 1  # softmax index for AI; None = single sigmoid logit
    released: str = ""
    notes: str = ""
    bin_ok: bool = False
    extra: dict = field(default_factory=dict)


CLASSIFIERS = {
    s.name: s
    for s in [
        ClsSpec("openai-roberta-base", "openai-community/roberta-base-openai-detector", "seqcls", 512, ai_index=0,
                released="2019", notes="GPT-2 output detector (old baseline)"),
        ClsSpec("hc3-chatgpt-roberta", "Hello-SimpleAI/chatgpt-detector-roberta", "seqcls", 512, released="2023-01",
                notes="trained on HC3 (ChatGPT answers)", bin_ok=True),
        ClsSpec("fakespot-roberta", "fakespot-ai/roberta-base-ai-text-detection-v1", "seqcls", 512, released="2025-02"),
        ClsSpec("tmr-roberta-raid", "Oxidane/tmr-ai-text-detector", "seqcls", 512, released="2026-01",
                notes="RoBERTa-base, RAID, hard-negative mining"),
        ClsSpec("superannotate-roberta-large", "SuperAnnotate/ai-detector", "superannotate", 512, ai_index=None,
                released="2024-09"),
        ClsSpec("desklib-deberta-large", "desklib/ai-text-detector-v1.01", "desklib", 768, ai_index=None,
                released="2025-02"),
        ClsSpec("tropa-mini-deberta-large", "wasitaigeneratedcom/tropa-mini-ai-text-detector", "tropa", 768,
                ai_index=None, released="2026-08", notes="4-class head; score = 1 - P(human)"),
        ClsSpec("tabularis-modernbert-base", "tabularisai/ai-text-detection", "seqcls", 512, released="2026-05",
                notes="RAID + 6 datasets, ~11M rows"),
        ClsSpec("drayson-modernbert-raid-mage", "GeorgeDrayson/modernbert-ai-detection-raid-mage", "seqcls", 8192,
                released="2025-05"),
        ClsSpec("rasbt-modernbert-base", "rasbt/ai-text-detector-modernbert", "seqcls", 8192, released="2026-08",
                notes="trained on rasbt/human-vs-ai-50k"),
        ClsSpec("vanguard-modernbert-large", "ShantanuT01/vanguard-ai-text-detector", "seqcls", 8192, ai_index=None,
                released="2026-05", notes="PAN-CLEF 2026 2nd-place system component"),
    ]
}


class MeanPoolHead(nn.Module):
    """desklib / tropa-mini: backbone + mean pooling + linear head."""

    def __init__(self, path: str, n_out: int):
        super().__init__()
        cfg = AutoConfig.from_pretrained(path)
        self.model = AutoModel.from_config(cfg)
        self.classifier = nn.Linear(cfg.hidden_size, n_out)

    def forward(self, input_ids, attention_mask):
        h = self.model(input_ids, attention_mask=attention_mask)[0]
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.classifier((h * m).sum(1) / m.sum(1).clamp(min=1e-9))


class SAHead(nn.Module):
    """SuperAnnotate RobertaClassifier: <s> token -> linear (reimplemented from their v1.1.0 source)."""

    def __init__(self, path: str):
        super().__init__()
        cfg = AutoConfig.from_pretrained("FacebookAI/roberta-large")
        from transformers import RobertaModel

        self.roberta = RobertaModel(cfg, add_pooling_layer=False)
        self.dense = nn.Linear(cfg.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        return self.dense(self.roberta(input_ids, attention_mask=attention_mask)[0][:, 0, :])


def _load_state(module: nn.Module, path: str) -> None:
    from safetensors.torch import load_file
    import glob

    sd = {}
    for f in sorted(glob.glob(f"{path}/*.safetensors")):
        sd.update(load_file(f))
    missing, unexpected = module.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "position_ids" not in k]
    unexpected = [k for k in unexpected if "position_ids" not in k]
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")


_MD = re.compile(r"(\*\*|__|^#+\s*)", re.M)


class Classifier:
    def __init__(self, spec: ClsSpec, dtype=torch.bfloat16):
        self.spec = spec
        self.dtype = dtype
        if spec.kind == "superannotate":
            snapshot_download("FacebookAI/roberta-large", allow_patterns=["config.json"])
        path = fetch(spec.repo, spec.bin_ok)
        self.tok = AutoTokenizer.from_pretrained(path)
        if spec.kind == "seqcls":
            self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32)
        elif spec.kind in ("desklib", "tropa"):
            n = AutoConfig.from_pretrained(path).to_dict().get("detector_num_labels", 1) if spec.kind == "tropa" else 1
            self.model = MeanPoolHead(path, n)
            _load_state(self.model, path)
        elif spec.kind == "superannotate":
            self.model = SAHead(path)
            _load_state(self.model, path)
        materialize(self.model).eval()

    def windows(self, text: str) -> list[list[int]]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        special = self.tok("", add_special_tokens=True)["input_ids"]
        assert len(special) == 2, special  # [CLS]/<s> ... [SEP]/</s> for every model used here
        step = self.spec.window - 2
        # equal-sized windows so no window is a tiny tail (window logits are length-weighted anyway)
        n_win = max(1, math.ceil(len(ids) / step))
        size = math.ceil(len(ids) / n_win) if ids else 0
        chunks = [ids[i : i + size] for i in range(0, len(ids), size)] if ids else [[]]
        return [[special[0], *c, special[1]] for c in chunks]

    @torch.inference_mode()
    def ai_logit(self, batch_ids: list[list[int]]) -> torch.Tensor:
        pad = self.tok.pad_token_id or 0
        L = max(len(x) for x in batch_ids)
        ids = torch.full((len(batch_ids), L), pad, dtype=torch.long)
        mask = torch.zeros((len(batch_ids), L), dtype=torch.long)
        for i, x in enumerate(batch_ids):
            ids[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        dev = next(self.model.parameters()).device  # cpu unless the caller moved the model (judge_run --device)
        with torch.autocast(dev.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            out = self.model(input_ids=ids.to(dev), attention_mask=mask.to(dev))
        logits = (out.logits if hasattr(out, "logits") else out).float().cpu()
        if self.spec.kind == "tropa":  # 4-class: AI-ness = log((1-p_h)/p_h)
            logp = F.log_softmax(logits, -1)
            lp_h = logp[:, 0]
            return torch.log1p(-lp_h.exp().clamp(max=1 - 1e-7)) - lp_h
        if self.spec.ai_index is None:
            return logits[:, 0]
        other = 1 - self.spec.ai_index
        return logits[:, self.spec.ai_index] - logits[:, other]

    def score(self, text: str) -> tuple[float, int]:
        if self.spec.kind == "superannotate":
            text = _MD.sub("", text)
        wins = self.windows(text)
        logits = self.ai_logit(wins)
        w = torch.tensor([len(x) for x in wins], dtype=torch.float32)
        return float((logits * w).sum() / w.sum()), int(w.sum())


# ---------------------------------------------------------------- zero-shot LM statistics

LM_PAIRS = {
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
}
MAX_LM_TOKENS = 1536


class LMStats:
    """One forward pass of a base (observer) and an instruct (performer) model sharing a tokenizer.

    Emits several zero-shot scores, all oriented so higher = more AI:
      binoculars  = -(log-PPL_performer / X-PPL(observer||performer))     [Hans et al. 2024]
      fdg_b2i     = Fast-DetectGPT analytic, sampling=base, scoring=instruct [Bao et al. 2024]
      fdg_b2b     = Fast-DetectGPT, base for both (white-box single model)
      loglik_base / loglik_inst = mean token log-likelihood
    """

    def __init__(self, pair: str, dtype=torch.bfloat16):
        base, inst = LM_PAIRS[pair]
        self.tok = AutoTokenizer.from_pretrained(fetch(base))
        self.obs = materialize(AutoModelForCausalLM.from_pretrained(fetch(base), dtype=dtype)).eval()
        self.perf = materialize(AutoModelForCausalLM.from_pretrained(fetch(inst), dtype=dtype)).eval()

    @torch.inference_mode()
    def score(self, text: str, chunk: int = 256) -> tuple[dict, int]:
        """Per-position statistics computed in position chunks, so only a [chunk x vocab] float32 block
        exists at a time. A full [tokens x 152k] float32 logit tensor is ~0.8 GB per copy, and holding
        several of them next to 6 GB of weights pushed the box into reclaim storms."""
        ids = self.tok(text, return_tensors="pt")["input_ids"][:, :MAX_LM_TOKENS]
        h_o = self.obs.model(ids).last_hidden_state[0, :-1]
        h_p = self.perf.model(ids).last_hidden_state[0, :-1]
        tgt = ids[0, 1:]
        acc = {k: [] for k in ("ll_o", "ll_p", "mu_b2i", "var_b2i", "mu_b2b", "var_b2b")}
        for s in range(0, tgt.shape[0], chunk):
            logp_o = F.log_softmax(self.obs.lm_head(h_o[s : s + chunk]).float(), -1)
            logp_p = F.log_softmax(self.perf.lm_head(h_p[s : s + chunk]).float(), -1)
            t = tgt[s : s + chunk, None]
            p_o = logp_o.exp()
            acc["ll_o"].append(logp_o.gather(1, t)[:, 0])
            acc["ll_p"].append(logp_p.gather(1, t)[:, 0])
            for key, lp in (("b2i", logp_p), ("b2b", logp_o)):
                mu = (p_o * lp).sum(-1)
                acc[f"mu_{key}"].append(mu)
                acc[f"var_{key}"].append((p_o * lp.pow(2)).sum(-1) - mu.pow(2))
            del logp_o, logp_p, p_o
        a = {k: torch.cat(v) for k, v in acc.items()}
        # Binoculars (official code): log-PPL from performer; X-PPL = CE(observer probs, performer log-probs)
        bino = (-a["ll_p"].mean() / -a["mu_b2i"].mean()).item()

        def fdg(ll, mu, var):  # Fast-DetectGPT analytic criterion
            return ((ll.sum() - mu.sum()) / var.sum().clamp(min=1e-6).sqrt()).item()

        out = dict(binoculars=-bino, fdg_b2i=fdg(a["ll_p"], a["mu_b2i"], a["var_b2i"]),
                   fdg_b2b=fdg(a["ll_o"], a["mu_b2b"], a["var_b2b"]),
                   loglik_base=a["ll_o"].mean().item(), loglik_inst=a["ll_p"].mean().item())
        if not all(math.isfinite(v) for v in out.values()):
            raise RuntimeError(f"non-finite LM stats: {out}")
        return out, int(ids.shape[1])
