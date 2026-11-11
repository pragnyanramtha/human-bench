"""Score texts with Pangram's open-weights EditLens Llama-3.2-3B model on a GPU, as released (bfloat16).

    python gpu_score.py --in texts.jsonl --out scores.jsonl

Input rows: {"sha", "text"}. Output rows: {"sha", "score", "tokens", "used_tokens"}; higher = more AI.
A manifest (versions, GPU, pins, timing) is written next to the output.

Standalone on purpose: it is copied to a rented GPU, so it imports nothing from this repository. The
preprocessing, the classification head and the score are the same as in editlens.py (the CPU path);
only the device differs: the base model runs in bfloat16 on CUDA with no quantisation. Texts are scored
one at a time, so there is no padding, and the model reads the first 1,024 tokens.
Resumable: rows already in --out are skipped.

Models, pinned: the adapter `ml-resources/editlens_Llama-3.2-3B` (byte-identical to the gated
`pangram/editlens_Llama-3.2-3B`) on the base `unsloth/Llama-3.2-3B` (the same weights as the gated
`meta-llama/Llama-3.2-3B`). Neither needs a Hugging Face login.

Preprocessing and NormedLinear adapted from Pangram/EditLens (Thai, Emi, Masrour, Iyyer),
https://github.com/pangramlabs/EditLens, CC BY-NC-SA 4.0. This derivative file is licensed
CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/ . Changes: GPU inference, hashed
input and output rows, resumable scoring.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch

ADAPTER = "ml-resources/editlens_Llama-3.2-3B"
BASE = "unsloth/Llama-3.2-3B"
PINS = {ADAPTER: "cf23fd0738fc97f2420a8132e0866fa0ca840c3d", BASE: "d4446454d87d51aa42e1fb174f25acc5f8762331"}
LIMIT = 1024


def clean(text: str) -> str:
    import emoji

    text = emoji.demojize(text)
    if "</think>" in text:
        text = text.split("</think>")[1].strip()
    paragraphs = [p for p in text.split("\n") if p.strip()]
    if paragraphs:
        first = emoji.replace_emoji(re.sub(r"^[^a-zA-Z0-9]*", "", paragraphs[0]), "")
        if any(first.startswith(p) for p in ["Sure", "Here", "Abstract", "Title", "I'm happy to help", "Certainly"]) \
                and len(paragraphs) > 1:
            text = "\n".join(paragraphs[1:])
    return re.sub(r"\s+", " ", text.lower()).strip()


class NormedLinear(torch.nn.Module):
    def __init__(self, h: int, n: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(h, dtype=torch.bfloat16)
        self.linear = torch.nn.Linear(h, n, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.linear(self.norm(x))


def load(device: str):
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    def fetch(repo: str) -> str:
        return snapshot_download(repo, revision=PINS[repo], allow_patterns=["*.json", "*.safetensors", "*.txt"])

    path = fetch(ADAPTER)
    tok = AutoTokenizer.from_pretrained(path)
    assert json.loads((Path(path) / "adapter_config.json").read_text())["base_model_name_or_path"] == "meta-llama/Llama-3.2-3B"
    with safe_open(str(Path(path) / "adapter_model.safetensors"), framework="pt") as f:
        n = next(f.get_tensor(k).shape[0] for k in f.keys() if "score" in k and "linear.weight" in k)
    base = AutoModelForSequenceClassification.from_pretrained(fetch(BASE), num_labels=n, dtype=torch.bfloat16)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False
    base.score = NormedLinear(base.config.hidden_size, n)
    model = PeftModel.from_pretrained(base, path).eval().to(device)
    return tok, model


@torch.inference_mode()
def score(tok, model, text: str, device: str) -> tuple[float, int, int]:
    cleaned = clean(text)
    total = tok(cleaned, return_tensors="pt", truncation=False)["input_ids"].shape[1]
    enc = tok(cleaned, return_tensors="pt", truncation=True, max_length=LIMIT).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**enc).logits.float()[0]
    p = logits.softmax(-1).cpu()
    s = float((p * torch.arange(len(p))).sum() / (len(p) - 1))
    if not math.isfinite(s):
        raise RuntimeError("non-finite EditLens score")
    return s, total, min(total, LIMIT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "this script needs a CUDA GPU"
    device = "cuda"
    rows = [json.loads(x) for x in a.inp.read_text(encoding="utf-8").split("\n") if x.strip()]
    done = {json.loads(x)["sha"] for x in a.out.read_text().split("\n") if x.strip()} if a.out.is_file() else set()
    todo = [r for r in rows if r["sha"] not in done and r["text"].strip()]
    print(f"{len(rows)} texts, {len(done)} already scored, {len(todo)} to do on {torch.cuda.get_device_name(0)}", flush=True)
    t0 = time.time()
    tok, model = load(device)
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    with a.out.open("a") as f:
        for i, r in enumerate(todo, 1):
            s, total, used = score(tok, model, r["text"], device)
            f.write(json.dumps({"sha": r["sha"], "score": s, "tokens": total, "used_tokens": used}) + "\n")
            if i % 200 == 0 or i == len(todo):
                f.flush()
                print(f"{i}/{len(todo)} ({(time.time() - t1) / i:.3f}s each)", flush=True)
    import importlib.metadata as md

    a.out.with_suffix(".manifest.json").write_text(json.dumps({
        "detector": "editlens-llama-3.2-3b", "adapter": ADAPTER, "base": BASE, "pins": PINS, "dtype": "bfloat16",
        "quantised": False, "max_tokens": LIMIT, "batch_size": 1, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "versions": {p: md.version(p) for p in ("transformers", "peft", "accelerate", "emoji", "safetensors", "huggingface_hub")},
        "texts": len(rows), "scored_this_run": len(todo), "seconds_scoring": round(time.time() - t1, 1),
        "seconds_loading": round(t1 - t0, 1),
    }, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
