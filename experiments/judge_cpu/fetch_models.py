"""Download every model used (safetensors only where available)."""
from huggingface_hub import snapshot_download
from detectors import CLASSIFIERS, LM_PAIRS, fetch

for s in CLASSIFIERS.values():
    print(s.repo, fetch(s.repo, s.bin_ok), flush=True)
snapshot_download("FacebookAI/roberta-large", allow_patterns=["config.json"])
for base, inst in LM_PAIRS.values():
    print(base, fetch(base), flush=True)
    print(inst, fetch(inst), flush=True)
print("done")
