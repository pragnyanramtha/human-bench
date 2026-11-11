"""The text-free release evidence reproduces the published headline exactly."""
import importlib.util
import json
from pathlib import Path


def test_published_leaderboard():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('ledger', root / 'scripts/ledger.py')
    ledger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ledger)
    rows = [json.loads(x) for x in ledger.LEDGER.read_text().splitlines() if x.strip()]
    table = '\n'.join(ledger.leaderboard_lines(rows, json.loads(ledger.CALIB.read_text())))
    assert table in (root / 'reports/results-v0.1.md').read_text()
