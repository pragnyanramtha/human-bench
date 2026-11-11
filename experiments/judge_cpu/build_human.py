"""Sample the human side of the judge eval set from public HF datasets.

All sources pre-date ChatGPT's release (2022-11-30). Raw files are downloaded
by hand into private/judge_eval/raw/ (see README.md). Output:
private/judge_eval/human.jsonl. Re-running with the same seed gives the same set.
"""

from __future__ import annotations

import csv
import hashlib
import html
import random
import re
import sys
from datetime import datetime

import pyarrow.parquet as pq

from common import HUMAN_PATH, RAW_DIR, word_count, write_jsonl

SEED = 20261109
MIN_W, MAX_W = 250, 1000
CUTOFF = datetime(2022, 11, 30)

SOURCES = {
    "aeslc": dict(dataset="Yale-LILY/aeslc", licence="Enron corpus (FERC public release); HF card: unknown",
                  date="1999-2002"),
    "stackexchange": dict(dataset="HuggingFaceH4/stack-exchange-preferences", licence="CC BY-SA 4.0"),
    "reddit": dict(dataset="webis/tldr-17", licence="CC BY 4.0 (dataset); posts by Reddit users",
                   date="2006-2016"),
    "yelp": dict(dataset="Yelp/yelp_review_full", licence="Yelp Dataset terms (non-commercial research)",
                 date="<=2015"),
    "imdb": dict(dataset="stanfordnlp/imdb", licence="research use (Maas et al. 2011)", date="<=2011"),
    "blog": dict(dataset="tasksource/blog_authorship_corpus",
                 licence="Blog Authorship Corpus, non-commercial research (Schler et al. 2006); HF card: apache-2.0"),
    "medium": dict(dataset="fabiochiu/medium-articles",
                   licence="HF card: MIT; articles remain the authors' copyright"),
    "writingprompts": dict(dataset="euclaise/writingprompts",
                           licence="HF card: MIT; stories by r/WritingPrompts users", date="2016-2018"),
}

# (channel, source, count, extra-filter key)
PLAN = [
    ("email", "aeslc", 28, None),
    ("forum", "stackexchange", 14, None),
    ("forum", "reddit", 14, "forum"),
    ("review", "yelp", 16, None),
    ("review", "imdb", 12, None),
    ("blog", "blog", 14, None),
    ("blog", "medium", 14, None),
    ("personal", "reddit", 14, "personal"),
    ("personal", "writingprompts", 14, None),
]

REDDIT_FORUM = {"personalfinance", "Fitness", "buildapc", "explainlikeimfive", "changemyview", "truegaming",
                "summonerschool", "askscience", "Android", "movies", "Games", "Cooking", "DIY", "cars",
                "legaladvice", "AskHistorians", "books", "photography", "running", "bicycling", "loseit",
                "Parenting", "college", "cscareerquestions", "jobs", "travel", "homebrewing", "gardening"}
REDDIT_PERSONAL = {"relationships", "relationship_advice", "TwoXChromosomes", "tifu", "offmychest",
                   "confession", "self", "raisedbynarcissists", "depression", "AskWomen", "AskMen",
                   "BreakUps", "dating_advice", "socialanxiety"}
SE_SITES = {"gardening.stackexchange.com", "homebrew.stackexchange.com", "poker.stackexchange.com",
            "opensource.stackexchange.com"}


def ok_text(t: str) -> bool:
    if not t:
        return False
    n = word_count(t)
    if not (MIN_W <= n <= MAX_W):
        return False
    ascii_share = sum(c.isascii() for c in t) / len(t)
    if ascii_share < 0.97:
        return False
    if t.count("http") > 2 or "[deleted]" in t or "[removed]" in t:
        return False
    letters = sum(c.isalpha() for c in t) / len(t)
    return letters > 0.7


def ws(t: str) -> str:
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def strip_html(t: str) -> str:
    t = re.sub(r"(?i)<br\s*/?>", "\n", t)
    t = re.sub(r"(?i)</p>|</li>|</blockquote>", "\n\n", t)
    t = re.sub(r"(?i)<li>", "- ", t)
    t = re.sub(r"<[^>]+>", "", t)
    return ws(html.unescape(t))


def load_aeslc():
    t = pq.read_table(RAW_DIR / "aeslc/train.parquet").to_pylist()
    bad = re.compile(r"(?i)original message|forwarded by|unsubscribe|click here|^>|-----|\bfrom:\s|\bsent:\s|"
                     r"confidential|this e-mail|disclaimer")
    for i, r in enumerate(t):
        body = ws(r["email_body"])
        if bad.search(body) or body.count("@") > 1:
            continue
        yield dict(src_id=f"aeslc-train-{i}", text=body, date=SOURCES["aeslc"]["date"],
                   meta=dict(subject=r["subject_line"]))


def load_stackexchange():
    import ast
    for f in sorted((RAW_DIR / "stackexchange").glob("s*.parquet")):
        for r in pq.read_table(f).to_pylist():
            meta = r["metadata"] if isinstance(r["metadata"], list) else ast.literal_eval(r["metadata"])
            site = meta[1].split("//")[-1]
            if site not in SE_SITES or r["date"] > "2018/12/31":
                continue
            answers = r["answers"] if isinstance(r["answers"], list) else ast.literal_eval(r["answers"])
            for a in answers:
                raw = a["text"]
                if "<code>" in raw or "<pre>" in raw or "<img" in raw:
                    continue
                yield dict(src_id=f"se-{site.split('.')[0]}-{r['qid']}-{a['answer_id']}", text=strip_html(raw),
                           date=f"question {r['date'].replace('/', '-')}; answer date not in dataset",
                           meta=dict(site=site, question=strip_html(r["question"])[:600]))


def load_reddit(kind: str):
    subs = REDDIT_FORUM if kind == "forum" else REDDIT_PERSONAL
    t = pq.read_table(RAW_DIR / "tldr17/part0.parquet", columns=["id", "subreddit", "content"])
    for sid, sub, content in zip(*(t.column(c).to_pylist() for c in ("id", "subreddit", "content"))):
        if sub in subs:
            yield dict(src_id=f"tldr17-{sid}", text=ws(content.replace(" \n ", "\n")),
                       date=SOURCES["reddit"]["date"], meta=dict(subreddit=sub))


def load_yelp():
    for i, r in enumerate(pq.read_table(RAW_DIR / "yelp/test.parquet").to_pylist()):
        yield dict(src_id=f"yelp-test-{i}", text=ws(r["text"].replace("\\n", "\n").replace('\\"', '"')),
                   date=SOURCES["yelp"]["date"], meta=dict(stars=r["label"] + 1))


def load_imdb():
    for i, r in enumerate(pq.read_table(RAW_DIR / "imdb/test.parquet").to_pylist()):
        yield dict(src_id=f"imdb-test-{i}", text=ws(re.sub(r"<br\s*/?>", "\n", r["text"])),
                   date=SOURCES["imdb"]["date"], meta=dict(sentiment="pos" if r["label"] else "neg"))


def _csv_rows(path):
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        while True:
            try:
                yield next(reader)
            except StopIteration:
                return
            except csv.Error:
                return  # truncated tail of the ranged download


def load_blog():
    for r in _csv_rows(RAW_DIR / "blog/head.csv"):
        if not r.get("text") or not r.get("date"):
            continue
        text = ws(r["text"].replace("urlLink", " "))
        yield dict(src_id=f"blog-{r['id']}-{hashlib.md5(text[:200].encode()).hexdigest()[:8]}", text=text,
                   date=r["date"].replace(",", " "),
                   meta=dict(gender=r["gender"], age=r["age"], topic=r["topic"]))


def load_medium():
    for i, r in enumerate(_csv_rows(RAW_DIR / "medium/head.csv")):
        ts = (r.get("timestamp") or "")[:19]
        try:
            when = datetime.fromisoformat(ts)
        except ValueError:
            continue
        if when >= CUTOFF or not r.get("text"):
            continue
        lines = [ln for ln in r["text"].split("\n") if not ln.startswith(("Photo by", "Image by", "Source:"))]
        yield dict(src_id=f"medium-{i}", text=ws("\n".join(lines)), date=when.date().isoformat(),
                   meta=dict(title=r["title"], tags=r.get("tags", "")[:200]))


def load_writingprompts():
    for i, r in enumerate(pq.read_table(RAW_DIR / "writingprompts/test.parquet").to_pylist()):
        story = r["story"].replace("<newline>", "\n")
        story = re.sub(r"``\s*", '"', story)
        story = re.sub(r"\s*''", '"', story)
        story = ws(re.sub(r"\s+”", "”", re.sub(r"“\s+", "“", story)))
        story = re.sub(r"(\w) (n't|'s|'re|'m|'ll|'ve|'d)\b", r"\1\2", story)
        yield dict(src_id=f"wp-test-{i}", text=story, date=SOURCES["writingprompts"]["date"],
                   meta=dict(prompt=r["prompt"].replace("[ WP ]", "").strip()[:400]))


LOADERS = {"aeslc": load_aeslc, "stackexchange": load_stackexchange, "yelp": load_yelp, "imdb": load_imdb,
           "blog": load_blog, "medium": load_medium, "writingprompts": load_writingprompts}


def main() -> None:
    rng = random.Random(SEED)
    rows, seen = [], set()
    for channel, source, count, kind in PLAN:
        pool = list(load_reddit(kind) if source == "reddit" else LOADERS[source]())
        pool = [p for p in pool if ok_text(p["text"]) and p["text"][:200] not in seen]
        # one item per author/question where we can tell, so the set isn't dominated by one person
        by_key = {}
        for p in pool:
            key = p["src_id"].rsplit("-", 1)[0] if source in ("blog", "stackexchange") else p["src_id"]
            by_key.setdefault(key, p)
        pool = list(by_key.values())
        rng.shuffle(pool)
        # half short (<=450 words), half longer, where the pool allows
        short = [p for p in pool if word_count(p["text"]) <= 450]
        longer = [p for p in pool if word_count(p["text"]) > 450]
        pick = longer[: count // 2]
        pick += short[: count - len(pick)]
        if source == "stackexchange":  # spread across the four sites
            pick, per_site = [], {}
            for p in longer[:200] + short[:200]:
                s = p["meta"]["site"]
                if per_site.get(s, 0) < count // len(SE_SITES) + 1 and len(pick) < count:
                    pick.append(p)
                    per_site[s] = per_site.get(s, 0) + 1
        print(f"{channel:9s} {source:15s} pool={len(pool):6d} picked={len(pick)}")
        for p in pick:
            seen.add(p["text"][:200])
            src = SOURCES[source]
            rows.append(dict(
                id=f"h{len(rows):03d}", channel=channel, source=source, dataset=src["dataset"],
                licence=src["licence"], src_id=p["src_id"], date=p["date"], words=word_count(p["text"]),
                text=p["text"], meta=p["meta"],
            ))
    write_jsonl(HUMAN_PATH, rows)
    print(f"wrote {len(rows)} human texts -> {HUMAN_PATH}")


if __name__ == "__main__":
    main()
