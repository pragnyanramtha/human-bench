#!/usr/bin/env python3
"""Read-only recomputation of the numbers in paper/humanbench.tex that the reports do not print.

    python3 paper/check_numbers.py

Reads reports/ledger-v0.1.jsonl (one row per model output) and reports/calibration-v0.1.json (each
detector's human-text scores and threshold). Writes nothing. Section A recomputes the headline table of
reports/results-v0.1.md from the ledger as a cross-check (point estimates only; the intervals in the
paper are quoted from the report). Sections B to F are the numbers the paper computes itself; each is
labelled with the tag used in paper/README.md. Section N checks the figures: the numbers that
paper/figures/make_figures.py plotted against the report, the example task against its task file, the
reasoning-effort control against its report, and the wording the paper must not use.

The pass rule is the one in scripts/ledger.py and src/humanbench/harness/report_v01.py:
an output passes when scripted == 1, floor == 1.0 and the headline detector's score is at or below the
threshold; an empty output never passes. The ledger fields `scripted` and `floor` are the programmatic checks and the
LLM judge.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HEADLINE = "editlens-llama-3.2-3b"

# The versions of the source files that paper/humanbench.tex was last checked against (first 16 hex of SHA-256).
SNAPSHOT = {'NOTICE.md': 'dae39ce1e2757681', 'README.md': '294907650f74f4b7', 'private/judge_cache/in/humans.jsonl': '9cf6fa977c640539', 'reports/calibration-v0.1.json': '48d2806683d4ad3f', 'reports/judge-cpu-baselines.md': 'ad3eaad82778f09f', 'reports/ledger-v0.1.jsonl': '022fca01054bd436', 'reports/results-v0.1.md': 'd170258a60184419', 'reports/surface-features-v0.1.md': '151916a221356a41', 'reports/workspace-format-v0.1.md': '13d84a4f4dcb9af1', 'scripts/audit/card_prompt.md': 'e495515818b7aaa9', 'scripts/refcards/build_prompt.md': '8e724098ade3c986', 'scripts/refcards/memprobe.py': '7ef0ba4f5570fbe3', 'src/humanbench/harness/report_v01.py': '3b1f79aeb7bd219e', 'src/humanbench/refs.py': '029c9987aa2141df', 'src/humanbench/scoring/floor.py': '757abc9d2e35c97d', 'src/humanbench/scoring/verifiers.py': '8e5fdb9f418a5bc4', 'tasks/ref/MANIFEST.json': '696bae4cbdc1846c', 'tasks/ref/MANIFEST.md': '35d0c9916128187b', 'tasks/ref/ref-review-02/card.yaml': '11048fe278003169', 'reports/effort-control-v0.1.md': '30ac9f6af4d6dd25', 'tasks/dev/work_docs-15/card.yaml': '851315ee142c15c3'}
import hashlib

# A source under private/ is not distributed: where it is absent it cannot be compared and is skipped.


def source_digest(path: str) -> str:
    """SHA-256 of a source file. A task card is hashed without its drafting stamp and its canary comment: the
    released copy of a card differs from the working copy in exactly those two lines."""
    data = (REPO / path).read_bytes()
    if path.startswith("tasks/") and path.endswith("card.yaml"):
        data = b"\n".join(line for line in data.split(b"\n")
                          if not line.startswith(b"drafted_by:") and not line.startswith(b"# HumanBench data should never"))
    return hashlib.sha256(data).hexdigest()


changed = [f for f, h in SNAPSHOT.items()
           if not (f.startswith("private/") and not (REPO / f).is_file())
           and (not (REPO / f).is_file() or source_digest(f)[:16] != h)]
print("0. sources: " + ("unchanged since the paper was last checked" if not changed else
                        "CHANGED since the paper was last checked, re-read the paper against them: " + ", ".join(changed)) + "\n")

rows = [json.loads(x) for x in (REPO / "reports" / "ledger-v0.1.jsonl").read_text().split("\n") if x.strip()]
cal = json.loads((REPO / "reports" / "calibration-v0.1.json").read_text())
cut = cal[HEADLINE]["cutoff"]


def human(r: dict, d: str = HEADLINE) -> bool:
    return (not r["empty"]) and r["detectors"][d] <= cal[d]["cutoff"]


def stop(r: dict) -> str:
    """The outcome of an output: the first gate it failed, or passed."""
    if not r["scripted"]:
        return "failed task checks"
    if r["floor"] == 0.0:
        return "failed quality review"
    if r["floor"] is None:
        return "not scored"
    return "passed" if human(r) else "flagged as AI"


def mean_of_task_means(rs: list[dict], f) -> float:
    by = defaultdict(list)
    for r in rs:
        by[r["task"]].append(float(f(r)))
    return sum(sum(v) / len(v) for v in by.values()) / len(by)


ref = [r for r in rows if r["split"] == "ref"]
models = sorted({r["model"] for r in rows})

print("A. cross-check of the by-kind breakdown (human-reference tasks), from the ledger")
print(f"   {'model':26s} score  quality  checks   review  flagged   outputs tasks")
for m in models:
    rs = [r for r in ref if r["model"] == m]
    c = Counter(stop(r) for r in rs)
    n = len(rs)
    print(f"   {m:26s} {100 * mean_of_task_means(rs, lambda r: stop(r) == 'passed'):5.1f}  "
          f"{100 * mean_of_task_means(rs, lambda r: r['scripted'] and r['floor'] == 1.0):6.1f}  "
          f"{100 * c['failed task checks'] / n:6.1f}  {100 * c['failed quality review'] / n:7.1f}  "
          f"{100 * c['flagged as AI'] / n:7.1f}   {n:5d} {len({r['task'] for r in rs}):5d}"
          + (f"  ({c['not scored']} not scored)" if c["not scored"] else ""))

print("\nB. [ledger-size] rows in the ledger, by task set")
by_split = Counter(r["split"] for r in rows)
print(f"   total {len(rows)}; " + ", ".join(f"{k} {v}" for k, v in sorted(by_split.items()))
      + f"; models {len(models)}; samples per (model, task): {sorted(Counter(Counter((r['model'], r['task']) for r in rows).values()).items())}")

print("\nC. [calibration-above] human calibration texts above each detector's threshold")
for d, c in cal.items():
    hs = c["human_scores"]
    print(f"   {c['label']:20s} {sum(s > c['cutoff'] for s in hs)} of {len(hs)} above the threshold {c['cutoff']:.4f}")

print("\nD. [pooled-ref] all eleven models pooled, reference tasks")
c = Counter(stop(r) for r in ref)
n = len(ref)
print(f"   outputs {n}; " + "; ".join(f"{k} {v} ({100 * v / n:.1f}%)" for k, v in c.most_common()))
print(f"   read as human to the headline detector, whatever else: {sum(human(r) for r in ref)} of {n} "
      f"({100 * sum(human(r) for r in ref) / n:.1f}%)")

print("\nE. [tasks-with-pass] reference tasks on which at least one output counts")
cards = sorted({r["task"] for r in ref})
won = {t for t in cards if any(stop(r) == "passed" for r in ref if r["task"] == t)}
print(f"   any model: {len(won)} of {len(cards)} tasks")
for m in models:
    w = {r["task"] for r in ref if r["model"] == m and stop(r) == "passed"}
    print(f"   {m:26s} {len(w):2d} of {len(cards)} tasks")
chan = lambda t: t.split("-")[1]
per = defaultdict(lambda: [0, 0, set()])
for r in ref:
    x = per[chan(r["task"])]
    x[0] += stop(r) == "passed"
    x[1] += 1
    x[2].add(r["task"])
print("   by channel (pooled over models): " + "; ".join(
    f"{k} {a}/{b} ({100 * a / b:.1f}%, {len(s)} tasks)" for k, (a, b, s) in sorted(per.items())))

print("\nF. [not-flagged-given-quality] of the outputs that passed the programmatic checks and the LLM judge, the share not flagged")
for m in models:
    good = [r for r in ref if r["model"] == m and r["scripted"] and r["floor"] == 1.0]
    ok = sum(human(r) for r in good)
    print(f"   {m:26s} {ok:3d} of {len(good):3d} ({100 * ok / len(good):.1f}%)")

print("\nG. [task-mix] composition of the released tasks, read from the task files (tasks/dev, tasks/ref)")
import re


def field(path: Path, name: str) -> str:
    m = re.search(rf"^{name}:\s*(\S+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    return m.group(1) if m else "?"


dev = sorted((REPO / "tasks" / "dev").glob("*/card.yaml"))
print(f"   synthetic public tasks: {len(dev)}")
for name in ("mode", "channel", "band"):
    print(f"   by {name}: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(field(p, name) for p in dev).items())))
used = [REPO / "tasks" / "ref" / t / "card.yaml" for t in cards]
gone = sorted(p.parent.name for p in used if not p.is_file())
print(f"   human-reference tasks in the ledger: {len(used)}; by channel (from the task ids): "
      + ", ".join(f"{k} {len(x[2])}" for k, x in sorted(per.items())))
print(f"   of these, task files no longer in tasks/ref (dropped after the ledger was written): {len(gone)} {gone}")
for name in ("band", "mode"):
    print(f"   by {name}, tasks still present: " + ", ".join(
        f"{k} {v}" for k, v in sorted(Counter(field(p, name) for p in used if p.is_file()).items())))

print("\nH. [pass-concentration] counted outputs on email and blog tasks against the other channels")
eb = [x for k, x in per.items() if k in ("email", "blog")]
ot = [x for k, x in per.items() if k not in ("email", "blog")]
print(f"   email + blog: {sum(x[0] for x in eb)} of {sum(x[1] for x in eb)} outputs on {sum(len(x[2]) for x in eb)} tasks; "
      f"other channels: {sum(x[0] for x in ot)} of {sum(x[1] for x in ot)} outputs on {sum(len(x[2]) for x in ot)} tasks")

print("\nI. [largest-loss] the most common of the three failures for each model, reference tasks")
for m in models:
    c = Counter(stop(r) for r in ref if r["model"] == m)
    c.pop("passed", None)
    print(f"   {m:26s} {c.most_common(1)[0][0]}")

print("\nJ. [parts] public tasks (human-reference + invented, split ref + dev) and held-out tasks (split test)")
PARTS = {"public": ("ref", "dev"), "private": ("test",)}
for part, splits in PARTS.items():
    rs = [r for r in rows if r["split"] in splits]
    c = Counter(stop(r) for r in rs)
    n = len(rs)
    print(f"   {part}: {len({r['task'] for r in rs})} tasks, {n} outputs pooled over the models; "
          + "; ".join(f"{k} {v} ({100 * v / n:.1f}%)" for k, v in c.most_common()))
print(f"   {'model':26s} public  private  quality-pub  quality-priv   (cross-check of the leaderboard, point estimates)")
for m in models:
    cells = []
    for f in (lambda r: stop(r) == "passed", lambda r: r["scripted"] and r["floor"] == 1.0):
        for splits in PARTS.values():
            cells.append(100 * mean_of_task_means([r for r in rows if r["model"] == m and r["split"] in splits], f))
    print(f"   {m:26s} {cells[0]:6.1f}  {cells[1]:7.1f}  {cells[2]:11.1f}  {cells[3]:12.1f}")
for part, splits in PARTS.items():
    big = Counter()
    for m in models:
        c = Counter(stop(r) for r in rows if r["model"] == m and r["split"] in splits)
        c.pop("passed", None)
        big[c.most_common(1)[0][0]] += 1
    print(f"   largest loss, {part}: " + ", ".join(f"{k}: {v} models" for k, v in big.most_common()))

print("\nK. [transcription] the tables in paper/humanbench.tex against the generated reports")
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def md_tables(path: Path) -> list[tuple[str, dict[str, list[str]]]]:
    """Every markdown table whose first column is a model: (header line, {model: cells})."""
    out, lines, i = [], path.read_text(encoding="utf-8").split("\n"), 0
    while i < len(lines):
        if lines[i].startswith("| model |"):
            head, body, i = lines[i], {}, i + 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                body[cells[0]] = cells[1:]
                i += 1
            out.append((head, body))
        i += 1
    return out


def nums(cells: list[str]) -> list[str]:
    return [n for c in cells for n in NUM.findall(re.sub(r"\(\d+ tasks\)", "", c.replace("**", "")))]


def first(tables, needle: str) -> dict[str, list[str]]:
    """The first table whose header contains the text (the human-reference section comes first in the report)."""
    return next(body for head, body in tables if needle in head)


res = md_tables(REPO / "reports" / "results-v0.1.md")
surf = md_tables(REPO / "reports" / "surface-features-v0.1.md")[0][1]
board, kinds_t = first(res, "HumanBench score"), first(res, "human-reference tasks")
# The outcome table (failed task checks, failed quality review, flagged as AI): three columns, the last "flagged as AI".
stops_t = next(body for head, body in res if head.rstrip(" |").endswith("flagged as AI") and head.count("|") == 5)
ref_all, ref_fail = first(res, "Open Pangram large (headline)"), first(res, "formatting the venue does not show")
expected = {
    "tab:main": {m: nums(c) for m, c in board.items()},
    "tab:outcomes": {m: nums(c) + [nums([board[m][0]])[0]] for m, c in stops_t.items()},  # the fourth outcome is the score
    "tab:bykind": {m: nums(c[:2]) for m, c in kinds_t.items()},  # public kinds; private is in tab:main
    "tab:bydetector": {m: nums(c[:4]) for m, c in ref_all.items()},
    "tab:failures": {m: [nums([c[k]])[0] for k in (0, 1, 2, 3, 8)] for m, c in ref_fail.items()},
    "tab:surface": {m: nums(c[1:]) for m, c in surf.items()},  # the output count is in the caption
}
tex = (REPO / "paper" / "humanbench.tex").read_text(encoding="utf-8")
bad = len(changed)
for label, want in expected.items():
    if f"\\label{{{label}}}" not in tex:
        print(f"   {label:16s} NOT IN THE PAPER")
        bad += 1
        continue
    block = tex[:tex.index(f"\\label{{{label}}}")]
    block = block[block.rindex("\\begin{tabular}"):block.rindex("\\end{tabular}")]
    got = {m: nums([rest.replace("\\%", "")]) for m, rest in re.findall(r"\\mdl\{([^}]+)\}([^\n]*)\\\\", block)}
    if label == "tab:main" and list(got) != list(want):
        print("   MISMATCH: paper leaderboard order differs from the report")
        bad += 1
    wrong = [m for m in want if got.get(m) != want[m]]
    bad += len(wrong)
    print(f"   {label:16s} {len(want) - len(wrong)} of {len(want)} rows match" + (f"; MISMATCH: {wrong}" if wrong else ""))
print("   " + ("transcription check passed" if not bad else f"{bad} PROBLEMS"))

# Overall leaderboard order and point estimates from the ledger, including secondary columns.
assert all(n == 3 for n in Counter((r["model"], r["task"]) for r in rows).values())
assert all(len({r["task"] for r in rows if r["model"] == m}) == 212 for m in models)
assert list(board) == sorted(board, key=lambda m: -mean_of_task_means(
    [r for r in rows if r['model'] == m], lambda r: stop(r) == 'passed'))
for m, cells in board.items():
    for cell, splits, quality_gate in zip(cells,
            [('ref', 'dev', 'test'), ('ref', 'dev'), ('test',), ('ref', 'dev', 'test')],
            [False, False, False, True]):
        rs = [r for r in rows if r['model'] == m and r['split'] in splits]
        score = mean_of_task_means(rs, lambda r: r['scripted'] and r['floor'] == 1.0
                                   if quality_gate else stop(r) == 'passed')
        assert nums([cell])[0] == f'{100 * score:.1f}', (m, cell)
print('   overall ordering and all headline point estimates match the ledger')

# L: Computed prose claims. Unlike transcription checks, these derive their expected
# strings from the ledger, manifest and calibration inputs on each invocation.
print('\nL. derived claims in prose')
def require(fragment):
    global bad
    if fragment not in tex:
        print('   MISMATCH: missing derived claim:', fragment)
        bad += 1

def pct(n, d):
    return f'{100*n/d:.1f}'

pub = [r for r in rows if r['split'] in ('ref','dev')]
priv = [r for r in rows if r['split'] == 'test']
for group in (rows,):
    rates = [100*mean_of_task_means([r for r in group if r['model']==m], lambda r: stop(r)=='passed') for m in models]
    require(f'{min(rates):.1f}\\%')
    require(f'{max(rates):.1f}\\%')
for rs in (ref,):
    counts = Counter(stop(r) for r in rs)
    for value in counts.values():
        require(f'{value:,} ({pct(value,len(rs))}\\%)')
    require(f'{len(rs):,}')
require(f'{len(rows):,}')
require(f'{len(won)} of {len(cards)} tasks')
require(f'{len(cards)} usable for every launch model')
for m in ('claude-fable-5-2', 'gpt-6.1-sol'):
    good = [r for r in ref if r['model']==m and r['scripted'] and r['floor']==1]
    n = sum(human(r) for r in good)
    require(f'{n} of {len(good)} ({pct(n,len(good))}\\%)')
# Channel concentration
for channel in ('email','blog'):
    a,b,_ = per[channel]
    require(f'{a} of {b:,} {channel} outputs')
other = [r for r in ref if chan(r['task']) not in ('email','blog')]
require(f"{sum(stop(r)=='passed' for r in other)} of {len(other):,}")

# Human pool: never print or write texts, only aggregate counts.
humans_path = REPO/'private/judge_cache/in/humans.jsonl'
if humans_path.is_file():
    humans = [json.loads(line) for line in humans_path.read_text().splitlines() if line.strip()]
    assert len(humans)==len(cal[HEADLINE]['human_scores'])
    for key in ('channel','source'):
        c = Counter(r[key] for r in humans)
        for name,n in c.items():
            require(f'{name.replace("_", " ")} {n}')
        print('   calibration',key,dict(sorted(c.items())))
else:  # the human texts are not distributed: the pool's make-up cannot be re-counted from a public copy
    print('   calibration pool make-up: not checked (the human texts are not in this copy)')
manifest = json.loads((REPO/'tasks/ref/MANIFEST.json').read_text())
active = [c for c in manifest['cards'] if c['status']=='active']
pinned = sum(bool(c.get('revision')) for c in active)
require(f'Of the {len(active)} active tasks in the manifest, {pinned}')
require(f'The other {len(active)-pinned}')
assert len(manifest['launch_models_probed']) == len(models)
require(f'{len(manifest["cards"])} texts were selected')

# Failure-table comparative claims, including the count that README misstated.
def value(x):
    return float(nums([x])[0])
fmt_count = sum(value(c[0]) == max(value(x) for x in c[:8]) for c in ref_fail.values())
words = {8:'eight',9:'nine',10:'ten',11:'eleven'}
require(f'formatting as the most frequent programmatic-check failure for {words[fmt_count]} of eleven models')
waivers = {m:value(c[8])-value(ref_all[m][0]) for m,c in ref_fail.items()}
winner=max(waivers,key=waivers.get)
require(f'raises Astra by {waivers[winner]:.1f} percentage points')
assert winner=='gpt-6-astra'
private_lower = sum(value(board[m][2]) < value(kinds_t[m][1]) for m in models)
require(f'yet {words[private_lower]} of the eleven models score lower')
# Validate detector table, formerly unchecked.
pos=tex.index('\\label{tab:detectors}')
detblock=tex[tex.rindex('\\begin{tabular}',0,pos):pos]
report=(REPO/'reports/results-v0.1.md').read_text()
refrows=[l for l in report.splitlines() if l.startswith('| public,') or l.startswith(('| held-out,', '| private,'))]
counts_by_kind=[[x for x in re.findall(r'(\d+/\d+) read as human',l)] for l in refrows]
for i,c in enumerate(cal.values()):
    line=[l for l in detblock.splitlines() if c['label'] in l][0]
    expected_counts=[counts_by_kind[k][i] for k in (2,0,1)]
    assert re.findall(r'\d+/\d+',line)==expected_counts
    assert f'{c["cutoff"]:.4f}' in line
    assert sum(x>c['cutoff'] for x in c['human_scores'])==8
# Appendix B: compare every question's complete wording with the source template.
import ast
floor_tree=ast.parse((REPO/'src/humanbench/scoring/floor.py').read_text())
constants={n.targets[0].id:ast.literal_eval(n.value) for n in floor_tree.body
           if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
           and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)}
question_text=constants['TEMPLATE'].split('Questions:\n')[1].split('\n\n{privacy_q}')[0].replace('{cover}','')
question_text+='\n'+constants['PRIVACY_Q'].split('\n\nPRIVATE ITEMS')[0]
for line in question_text.splitlines():
    if line.startswith('- '):
        name,body=line[2:].split(': ',1)
        require('\\item['+name.replace('_',r'\_')+'] '+body)
print('   derived prose, detector table and verbatim judge questions checked')


# Additional comparisons in the results/discussion: recompute extrema, order and
# group counts rather than preserving an old narrative when results regenerate.
public_scores={m:100*mean_of_task_means([r for r in pub if r['model']==m],lambda r:stop(r)=='passed') for m in models}
quality={m:100*mean_of_task_means([r for r in rows if r['model']==m],lambda r:r['scripted'] and r['floor']==1) for m in models}
require(f'{min(quality.values()):.1f}\\% to {max(quality.values()):.1f}\\%'.replace('\\\\','\\'))
assert max(public_scores,key=public_scores.get)=='claude-fable-5-2'
assert min(public_scores,key=public_scores.get)=='gpt-6-luna'
assert public_scores['gpt-6-astra']>public_scores['gpt-6.1-sol']
assert sorted(public_scores,key=public_scores.get,reverse=True).index('deepseek-v4p1-flash-1017')==6
private_scores={m:value(board[m][2]) for m in models}
assert sorted(private_scores,key=private_scores.get,reverse=True).index('deepseek-v4p1-flash-1017')==1
assert all(value(kinds_t[m][0])<value(kinds_t[m][1]) for m in models)
assert sum(value(kinds_t[m][0])==0 for m in models)==2
assert sorted(kinds_t,key=lambda m:value(kinds_t[m][0]),reverse=True)[:3]==['claude-fable-5-2','gpt-6-astra','gpt-6.1-sol']
largest=Counter()
for m in models:
    counts=Counter(stop(r) for r in rows if r['model']==m)
    counts.pop('passed',None)
    largest[counts.most_common(1)[0][0]]+=1
require(f'{words[largest["flagged as AI"]]} models')
for col,lo,hi in [(0,34,86),(1,.61,.87),(2,15,18.6),(3,.47,.69)]:
    actual=[value(c[col+1]) for c in surf.values()]
    assert min(actual)==lo and max(actual)==hi
# Judge-validation stats contain only aggregate counts; never expose pieces.
for name,passed in [('validate-*-claude-sonnet-5-5.json',217),('validate-*-gpt-6.1-terra.json',208)]:
    if not (REPO/'private/floor').is_dir():  # judge validation records are not distributed
        print('   judge validation counts: not checked (the validation records are not in this copy)')
        break
    (found,)=sorted((REPO/'private/floor').glob(name))
    stats=json.loads(found.read_text())['stats']
    assert stats['oracle:ok']==passed
    assert sum(n for k,n in stats.items() if k.startswith('oracle:'))==217
    assert sum(n for k,n in stats.items() if k.startswith('junk:') and k.endswith(':ok'))==260
    assert not any(n for k,n in stats.items() if k.startswith('junk:') and not k.endswith(':ok'))
require('217 of 217');require('208 of the 217');require('The 9 failures')
print('   ranks, extrema, comparisons and private validation aggregate stats checked')

# N: the figures. make_figures.py writes every number it plots to figures/figure_data.json; here those numbers
# are compared with the report's tables, so a figure cannot drift from the leaderboard without this failing.
print('\nN. figures, example task, effort control and wording')
FIG = REPO / 'paper' / 'figures'
fd = json.loads((FIG / 'figure_data.json').read_text())
for name in ('fig-leaderboard', 'fig-outcomes', 'fig-detector', 'fig-kinds'):
    if not (FIG / f'{name}.pdf').is_file() or f'figures/{name}.pdf' not in tex:
        print('   MISSING figure', name); bad += 1
if fd['sources'] != {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest()[:16]
                     for f in ('reports/ledger-v0.1.jsonl', 'reports/calibration-v0.1.json')}:
    print('   STALE figures: the ledger or the calibration changed after make_figures.py last ran'); bad += 1
cell = lambda t: f'{t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]'
flat = lambda c: ' '.join(nums([c]))
if fd['order'] != list(board):
    print('   MISMATCH: figure row order differs from the leaderboard'); bad += 1
for m in board:
    d = fd['models'][m]
    pairs = [(cell(d['overall']), board[m][0]), (cell(d['public']), board[m][1]), (cell(d['private']), board[m][2]),
             (cell(d['ref']), kinds_t[m][0]), (cell(d['dev']), kinds_t[m][1]), (cell(d['test']), kinds_t[m][2])]
    pairs += [(f"{d['outcomes'][k]:.1f}", stops_t[m][i]) for i, k in
              enumerate(('failed task checks', 'failed quality review', 'flagged as AI'))]
    pairs.append((f"{d['outcomes']['passed']:.1f}", f"{d['overall'][0]:.1f}"))
    for got, want in pairs:
        if flat(got) != flat(want):
            print(f'   MISMATCH figure data {m}: {got} against the report {want}'); bad += 1
det = fd['detector']
gate = [r['detectors'][HEADLINE] for r in rows if r['scripted'] and r['floor'] == 1.0 and not r['empty']]
assert det['humans'] == len(cal[HEADLINE]['human_scores']) and det['answers_reaching_gate'] == len(gate)
assert det['answers_at_or_below'] == sum(x <= cut for x in gate) == sum(stop(r) == 'passed' for r in rows)
assert abs(fd['cutoff'] - cut) < 1e-12
require(f"the {det['humans']} human calibration texts")
require(f"the {det['answers_reaching_gate']:,} outputs")
require(f"the median human text scores {det['human_median']:.3f} and the median output {det['answer_median']:.2f}")
require(f"{round(100 * det['humans_at_or_below'] / det['humans'])}\\% of them stay at or below")
nine = sum(max(('failed task checks', 'failed quality review', 'flagged as AI'), key=lambda k: fd['models'][m]['outcomes'][k])
           == 'flagged as AI' for m in board)
require(f'the most common failure for {words[nine]} of the eleven models')
assert all(fd['models'][m][k][0] == max(fd['models'][x][k][0] for x in board) for k in ('ref', 'dev', 'test')
           for m in ['claude-fable-5-2'])  # "the highest point estimate in all three panels"
assert all(fd['models'][m]['ref'][0] < min(fd['models'][m]['dev'][0], fd['models'][m]['test'][0]) for m in board)
top, second = (fd['models'][m]['overall'] for m in list(board)[:2])
assert top[1] > second[2]  # "Claude Fable 5.2 stands clear": its interval does not reach the next model's
print(f"   figure data: {len(board)} models x 10 values match the report; detector panel {det['humans']} human texts, "
      f"{det['answers_reaching_gate']} outputs, {det['answers_at_or_below']} at or below the threshold")

# The example card (Figure 2): every piece of the excerpt must be in the card file, word for word.
import yaml
card = yaml.safe_load((REPO / 'tasks' / 'dev' / 'work_docs-15' / 'card.yaml').read_text(encoding='utf-8'))
squash = lambda t: ' '.join(t.split())
fig2 = tex[tex.index('{\\textbf{What the model sees}}'):tex.index('\\label{fig:card}')]
seen = fig2[fig2.index('\\raggedright') + len('\\raggedright'):fig2.index('(About me:')]
notes = [n for k in ('facts', 'anecdotes', 'opinions', 'wants') for n in card['notes_pack'][k]]
haystack = squash(card['prompt']) + ' \n ' + ' \n '.join(squash(n) for n in notes)
shown = 0
for line in re.split(r'\\\\(?:\[\d+pt\])?', seen):
    for piece in line.split('\\ldots{}'):
        piece = squash(piece).lstrip('- ').strip()
        if not piece or piece.startswith('('):
            continue
        shown += 1
        if piece not in haystack:
            print('   MISMATCH: example task excerpt is not in the task file:', piece[:70]); bad += 1
left_out = len(notes) - sum(squash(line).startswith('-- ') and '(' not in line for line in re.split(r'\\\\(?:\[\d+pt\])?', seen))
if f"({'zero one two three four five six seven eight nine ten'.split()[left_out]} further notes)" not in seen:
    print(f'   MISMATCH: the example task leaves out {left_out} notes'); bad += 1
p_ = card['persona']
require(f"(About me: {p_['role']}, {p_['age']}, {p_['lives']}. Length: it should come to roughly 250-400 words.)")
assert card['band'] == 'short' and [card['must_include']] == [[['Saturday', 'Sat']]] and not card.get('must_not_include')
for point in card['must_cover']:
    require(point)
require('225 to 440 words')  # the 250-400 band of a short card with the 10% tolerance of the length check
print(f'   example task: {shown} excerpts, {len(card["must_cover"])} coverage points and the closing line match tasks/dev/work_docs-15')

# The reasoning-effort control, quoted in the setup: every figure from its report.
effort = REPO / 'reports' / 'effort-control-v0.1.md'
if not effort.is_file():
    print('   MISSING reports/effort-control-v0.1.md: the paper quotes it'); bad += 1
else:
    ctl = md_tables(effort)[0][1]
    minus = lambda x: x.replace('+', '').replace('-', '$-$')
    for m, c in ctl.items():
        diff, lo, hi = re.match(r'([+-][\d.]+) points \[([+-][\d.]+), ([+-][\d.]+)\]', c[5]).groups()
        require(f"{c[4].rstrip('%')}\\% at {c[3]}")
        require(f"against {c[2].rstrip('%')}\\% at {c[1]}")
        require(f"{minus(diff)} points [{minus(lo)}, {minus(hi)}]")
        require(f"{c[0]} tasks")
        if float(lo) > 0 or float(hi) < 0:
            print('   MISMATCH: an effort-control interval now excludes zero; the paper says none does'); bad += 1
    assert list(ctl) == ['claude-fable-5-2', 'gpt-6-astra', 'gpt-6.1-sol']
    print(f'   effort control: {len(ctl)} models match reports/effort-control-v0.1.md')
# The efforts of Table 2 against the runs listed in the results report.
stated = dict(re.findall(r'([\w.\-]+) (high|medium|low|provider default)[,.]', ' '.join(
    l for l in report.splitlines() if 'Reasoning effort:' in l)))
pos = tex.index('\\label{tab:models}')
for m, e in re.findall(r'\\mdl\{([^}]+)\} & \\code\{[^}]+\} & (\w+) &', tex[tex.rindex('\\begin{tabular}', 0, pos):pos]):
    if stated.get(m, '').replace('provider default', 'default') != e:
        print(f'   MISMATCH: effort of {m}: paper {e}, report {stated.get(m)}'); bad += 1
assert len(stated) == len(models)

# Wording. The outcome and gate names are fixed, and the paper names no product through which a model was
# reached. The words to keep out are stored as hashes so that this file does not print them either.
BANNED = {'5685de3ee5f9417f', 'c5c17eb76ea5d1d4', '706c697d64b01d28', 'a8fa7fd60893411a', '21e49e5e2065fb77',
          'a426560873fda49c', 'e6e9d554bdec5b35', 'a68e1872c573301b', '57de4cf40144bdf7', '75949eba4fa117d9',
          '0caee180da9d0f3f', '99bb88401742848e', 'e893d4b34355115b', '48820e6e74937bf8', '114766236d2ed982',
          'e97b1c01299966c5', 'a625159d18da5718'}
BANNED |= {'6ac9a32c0ca8e84e', '0db52f4076c08251', '8367cd66fdd136bb', '677fe21b65f4a6cf'}
BANNED |= {'21e49e5e2065fb77', 'a68e1872c573301b', '69e6e4f7c542e607', 'f6e054049dfeec23', 'c2e36ed36cf730e9', '706c697d64b01d28', '75b5e3c14bb6ccb4', '1cc5353c31ef4f18', '028e24aeb4cb91f0', 'd02f1a3c5c87fbd9', 'd794214e4e8af0f5', '114766236d2ed982', 'fe1991a9d8b6abd7', '4567d7082f00fc3c', '0f48c756a4d78ba5', 'ba5019abe5b20ae5', 'eb9b9848cdc37a72', '4153fb8b7177f223', '9ff7ec7c7496f7a3', '48820e6e74937bf8', 'f1ee41a83c23ca0d'}
reader_text = re.sub(r'(?<!\\)%[^\n]*', '', tex)
reader_text = re.sub(r'\\(?:label|ref|code|path|citep|mdl)\{[^}]*\}', '', reader_text)
text_only = re.sub(r'\\[a-zA-Z]+', ' ', reader_text).lower()
text_only = text_only.replace('answer quality', '').replace('card file', '').replace('stack exchange answers', '').replace('answers yes/no questions', '').replace('answers each question', '')
toks = re.findall(r"[a-z][a-z0-9\-']*", text_only)
grams = {' '.join(toks[i:i + n]) for n in range(1, 6) for i in range(len(toks) - n + 1)}
hit = sorted(g for g in grams if hashlib.sha256(g.encode()).hexdigest()[:16] in BANNED)
if hit:
    print(f'   WORDING: {len(hit)} phrase(s) the paper must not use, e.g. "{hit[0]}"'); bad += 1
for name in ('failed checks', 'rejected by judge', 'flagged as AI', 'programmatic checks', 'LLM judge', 'AI-text detector'):
    require(name)
for must in ('No human readers were involved', 'no human readers were involved',
             'task compliance, a model-judged quality bar and detectability under named',
             'differ from plain API calls', 'confounded with effort'):
    require(must)
print(f'   wording: {len(BANNED)} banned phrases checked, outcome and gate names checked')

# M: Exhaustive numerical-prose inventory. These are reviewed source contracts,
# not statistical re-estimation: fail if a reviewed claim or its source changes.
# The README distinguishes these guards from computations above.
print('\nM. reviewed numerical prose and protocol provenance')
contracts=json.loads((REPO/'paper/number_sources.json').read_text())
for contract in contracts:
    require(contract['text'])
    for path,digest in contract['sources'].items():
        if path.startswith('private/') and not (REPO/path).is_file():
            continue  # not distributed: cannot be compared in a public copy
        if source_digest(path)!=digest:
            print('   SOURCE CHANGED; re-review claim',contract['id'],path)
            bad+=1
# Cover every paragraph with a numerical literal outside model tables, equations,
# diagram/layout commands and bibliography (which has its own verified metadata).
def numeric_paragraphs(text):
    text=text.split('\\maketitle',1)[1]
    text=re.sub(r'\\begin\{(table\*?|figure\*?|equation)\}.*?\\end\{\1\}', '', text, flags=re.S)
    out=[]
    for para in text.split('\n\n'):
        para=para.strip()
        clean=re.sub(r'\\(?:mdl|code|citep|ref|label|path)\{[^}]*\}', '',para)
        if re.search(r'\d',clean) and not para.startswith(('\\bibliograph','\\clearpage','\\label')):
            out.append(para)
    return out
reviewed={c['text'] for c in contracts}
for para in numeric_paragraphs(tex):
    if para not in reviewed:
        print('   UNREVIEWED numerical paragraph:',para[:100]);bad+=1
print(f'   {len(contracts)} numerical paragraphs covered by reviewed source contracts')
print('\nPASS' if not bad else f'\nFAIL: {bad} problems')
raise SystemExit(1 if bad else 0)
