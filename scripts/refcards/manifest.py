"""Write the public, text-free manifest of the human-reference tasks: tasks/ref/MANIFEST.json and .md.

    uv run python scripts/refcards/manifest.py

For every human text that was selected: where it came from (dataset, revision, record locator, date
evidence, licence note), what happened to it (active, held back, dropped, and why), and for active tasks
the memorisation-probe result. No human text is included. Source texts keep their original names and
details: the tasks are not anonymised.
"""
from __future__ import annotations

import os
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

from humanbench import refs

REPO = Path(__file__).resolve().parents[2]
PRIV = REPO / "private"


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()] if path.is_file() else []


def main() -> None:
    destination = Path(os.environ.get("HUMANBENCH_MANIFEST_OUT", REPO / "tasks" / "ref"))
    destination.mkdir(parents=True, exist_ok=True)
    eval_h = {f"judge_eval:{h['id']}": h for h in rows(PRIV / "judge_eval" / "human.jsonl")}
    corpus = {}
    for f in sorted((PRIV / "corpus").glob("*.jsonl")):
        for r in rows(f):
            corpus[f"corpus:{r['id']}"] = r
    selection = json.loads((PRIV / "refcards" / "selection.json").read_text())
    active = {d.name for d in (REPO / "tasks" / "ref").iterdir() if d.is_dir()}
    pend = PRIV / "tasks" / "ref_pending"  # built, not yet audited: kept out of the released set
    pending = {d.name for d in pend.iterdir() if d.is_dir()} if pend.is_dir() else set()
    held = {d.name for d in (PRIV / "refcards" / "held_back").iterdir()} if (PRIV / "refcards" / "held_back").is_dir() else set()
    dropped = {d.name for d in (PRIV / "refcards" / "dropped").iterdir()} if (PRIV / "refcards" / "dropped").is_dir() else set()
    reasons = {}
    for f in (PRIV / "refcards" / "rejected").glob("*.json"):
        last = json.loads(f.read_text()).get("last") or {}
        reasons[f.stem] = ("not usable: " + (last.get("reason") or "")[:200]) if last.get("usable") is False\
            else "failed the builder's automatic checks: " + "; ".join(last.get("problems") or [])[:200]
    for f in sorted((PRIV / "audit").glob("ref*/fixes/*.md")):
        first = f.read_text().split("\n", 1)[0]
        if first.strip().upper().startswith("DROP") and f.stem not in active:
            reasons[f.stem] = "dropped in audit: " + f.read_text().strip()[:300].replace("\n", " ")
    # Publication reasons are deliberately separate from the original private audit evidence.
    for f in (PRIV / "refcards" / "dropped").glob("*/reason.md"):
        first = f.read_text().splitlines()[0]
        if first.startswith("DROP: "):
            reasons[f.parent.name] = first.removeprefix("DROP: ")
    public_reasons = PRIV / "refcards" / "public_reasons.json"
    if public_reasons.is_file():
        reasons.update(json.loads(public_reasons.read_text()))
    sources = yaml.safe_load((REPO / "src/humanbench/sourcing/sources.yaml").read_text())["sources"]
    cutoff_evidence = {s["repo"]: s["construction_note"] for s in sources
                       if s.get("pre_cutoff_by_construction") and s.get("construction_note")}
    out = []
    for s in selection:
        sid, src = s["id"], s["src"]
        h, c = eval_h.get(src), corpus.get(src)
        row = {"id": sid, "channel": s["channel"], "words": s["words"], "source": s["source"]}
        if h:
            row |= {"dataset": h.get("dataset"), "record": h.get("src_id"), "date": h.get("date"),
                    "licence_note": h.get("licence")}
        elif c:
            row |= {"dataset": c.get("repo"), "revision": c.get("revision"), "record": c.get("original_id"),
                    "date": c.get("date"), "date_basis": c.get("date_basis"), "licence_note": c.get("licence"),
                    "licence_status": c.get("licence_status")}
            repo_licence = (c.get("meta") or {}).get("repo_licence")
            if repo_licence:
                row["licence_note"] = f"{repo_licence} (repository licence recorded by Common Pile; applicability to issue/PR text is uncertain)"
        if row.get("date") is None and row.get("dataset") in cutoff_evidence:
            row["date_basis"] = "corpus-level evidence: " + cutoff_evidence[row["dataset"]] +\
                                " No individual text date is recorded; the corpus predates the 2022-11-30 threshold."
        if h:  # downloaded as files, before revisions were recorded: see "raw_downloads" in the manifest
            row |= {"revision": None, "revision_note": "the source file was downloaded without a pinned revision; "
                                                       "its checksum is under raw_downloads, and text_sha256 "
                                                       "identifies the text whatever the file looks like later"}
        if sid in active:
            ok, why = refs.eligible(sid)
            ref = PRIV / "oracle" / sid / "draft.md"
            placeholders = sorted(set(re.findall(r"\[[A-Z][A-Z_]*\]", ref.read_text()))) if ref.is_file() else []
            row |= {"status": "active", "eligible": ok, "memorisation_probe": why.split(": ")[1],
                    "transformations": ("none added by the benchmark; the source's own placeholders are kept as found: "
                                        + ", ".join(placeholders)) if placeholders else
                                       "none (the text is used as written; names and details are the source's)",
                    # the text as the benchmark uses it, so a text fetched from the source can be checked
                    "text_sha256": hashlib.sha256(ref.read_bytes()).hexdigest() if ref.is_file() else None}
        elif sid in pending:
            row |= {"status": "awaiting audit", "reason": "a task was built from this text but has not been through "
                                                         "the audit loop; it is not released or used in any result"}
        elif sid in held:
            row |= {"status": "held back", "reason": "not clean after the audit rounds, or its human text failed "
                                                    "the LLM judge against the task"}
        elif sid in dropped:
            row |= {"status": "dropped", "reason": reasons.get(sid, "audit verdict: the text cannot support a fair task")}
        else:
            row |= {"status": "not built", "reason": reasons.get(sid, "rejected by the builder")}
        out.append(row)
    raw = PRIV / "judge_eval" / "raw"  # the files the first batch of human texts was read from
    downloads = [{"file": str(f.relative_to(raw)), "bytes": f.stat().st_size,
                  "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                  "retrieved": datetime.fromtimestamp(f.stat().st_mtime, UTC).date().isoformat()}
                 for f in sorted(raw.glob("*/*")) if f.suffix in (".parquet", ".csv")] if raw.is_dir() else []
    (destination / "MANIFEST.json").write_text(json.dumps(
        {"launch_models_probed": refs.launch_models(),
         "raw_downloads_note": "Files behind the tasks that have no dataset revision; what each one is and where it "
                               "came from is listed in experiments/judge_cpu/README.md (Raw downloads). The records "
                               "were cut from them by experiments/judge_cpu/build_human.py.",
         "raw_downloads": downloads, "cards": out}, indent=1, ensure_ascii=False) + "\n")
    st = Counter(r["status"] for r in out)
    act = [r for r in out if r["status"] == "active"]
    lines = ["# Human-reference tasks: where the human texts come from", "",
             "Generated by `scripts/refcards/manifest.py`. Per-task detail: `MANIFEST.json`. No human text is "
             "published here: each task records its dataset, record locator and date so the text can be retrieved "
             "from the source. Names and details in the tasks are the source's own; the tasks are not anonymised.", "",
             f"Selected {len(out)} texts: {st['active']} active, {st['held back']} held back, {st['dropped']} dropped "
             f"in audit, {st['not built']} not built, {st['awaiting audit']} built and awaiting audit (not "
             f"released). Of the active tasks, {sum(r['eligible'] for r in act)} pass the "
             f"memorisation probe for every launch model ({', '.join(refs.launch_models())}).", "",
             "| source | dataset | licence note | selected | active |", "|---|---|---|---|---|"]
    by = {}
    for r in out:
        k = (r["source"], r.get("dataset") or "", (r.get("licence_note") or "")[:110])
        by.setdefault(k, [0, 0])
        by[k][0] += 1
        by[k][1] += r["status"] == "active"
    for (src, ds, lic), (n, a) in sorted(by.items()):
        lines.append(f"| {src} | `{ds}` | {lic} | {n} | {a} |")
    lines += ["", "Each task's date evidence is in `MANIFEST.json`. All texts predate the 2022-11-30 sourcing threshold; most are "
                  "from 1999-2017, and two issue-tracker posts are from late 2022. Licence notes are the project's "
                  "reading of each source's terms. Several sources allow research or evaluation use but not "
                  "redistribution (Yelp's dataset terms, for one, permit academic use only), which is why the texts are "
                  "not included. The texts keep their sources' restrictions; the project's own code and tasks have the "
                  "licences listed in `NOTICE.md`.", ""]
    unpinned = sum(1 for r in act if not r.get("revision"))
    lines += ["## Checking a text you retrieved", "",
              f"Every active task has `text_sha256` in `MANIFEST.json`: the SHA-256 of the human text as the benchmark "
              f"stores it, which is the source text in UTF-8 with leading and trailing whitespace removed and one "
              f"newline added at the end. {len(act) - unpinned} active tasks also name the dataset revision and the "
              f"record's position in it. The other {unpinned} come from files downloaded without a pinned revision "
              "(a first batch, taken before revisions were recorded); for those, `raw_downloads` gives each file's "
              "size, SHA-256 and retrieval date, and `experiments/judge_cpu/README.md` says what each file is. If a "
              "source file has changed since, the record locator may point elsewhere, and the text hash is the check "
              "that you have the right piece.", ""]
    (destination / "MANIFEST.md").write_text("\n".join(lines))
    print(st, "eligible", sum(r["eligible"] for r in act))


if __name__ == "__main__":
    main()
