"""Estimate CPU fine-tuning cost for a ModernBERT judge (no full training run).

Times real optimizer steps (forward + backward + AdamW) on this machine and
extrapolates. Uses already-downloaded checkpoints with the same architecture:
ModernBERT-base  <- GeorgeDrayson/modernbert-ai-detection-raid-mage
ModernBERT-large <- ShantanuT01/vanguard-ai-text-detector
  uv run python ft_timing.py [--steps 4] [--seq 512] [--batch 4]
"""

from __future__ import annotations

import argparse
import json
import resource
import time

import torch
from transformers import AutoModelForSequenceClassification

from common import RESULTS_DIR
from detectors import fetch

ARCH = {"modernbert-base": "GeorgeDrayson/modernbert-ai-detection-raid-mage",
        "modernbert-large": "ShantanuT01/vanguard-ai-text-detector"}


def time_steps(repo: str, steps: int, seq: int, batch: int, bf16: bool) -> float:
    m = AutoModelForSequenceClassification.from_pretrained(fetch(repo), num_labels=2, ignore_mismatched_sizes=True)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5)
    ids = torch.randint(5, 50000, (batch, seq))
    mask = torch.ones_like(ids)
    y = torch.randint(0, 2, (batch,))
    times = []
    for i in range(steps + 1):
        t = time.perf_counter()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            loss = torch.nn.functional.cross_entropy(m(input_ids=ids, attention_mask=mask).logits.float(), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if i:  # skip warm-up step
            times.append(time.perf_counter() - t)
    return sum(times) / len(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--arch", choices=list(ARCH), action="append")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out_path = RESULTS_DIR / "ft_timing.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name, repo in ARCH.items():
        if a.arch and name not in a.arch:
            continue
        for bf16 in (True,):
            s = time_steps(repo, a.steps, a.seq, a.batch, bf16)
            per_seq = s / a.batch
            # scenario: 3,000 training texts (~700 tokens each -> ~1.4 windows of 512), 3 epochs
            n_windows = 3000 * 1.4 * 3
            out[f"{name}{'-bf16' if bf16 else '-fp32'}-b{a.batch}"] = dict(
                sec_per_step=round(s, 2), batch=a.batch, seq=a.seq, sec_per_512tok_seq=round(per_seq, 2),
                est_hours_3000texts_3epochs=round(n_windows * per_seq / 3600, 1),
                peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
            print(name, out[f"{name}{'-bf16' if bf16 else '-fp32'}-b{a.batch}"], flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
