"""``humanbench-source``: data sourcing CLI (docs/sourcing.md).

Lane 1, public corpus (judge training and dev material only):
    humanbench-source sources [--channel review]
    humanbench-source verify [--source imdb]
    humanbench-source corpus --channel review --limit 200
    humanbench-source manifest

Lane 2, commissioned references:
    humanbench-source brief --card tasks/dev/forum-07.yaml
    humanbench-source screener --persona leeds-plumber | --all
    humanbench-source qa --card CARD --submission SUB.json [--model-outputs DIR] [--detector mod:Class]
    humanbench-source queue
    humanbench-source review --card CARD --submission-id ID --reviewer NAME --decision accept --note "..."
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import sys
from pathlib import Path

from humanbench.axes import Band, Channel
from humanbench.schema import load_personas
from humanbench.sourcing.paths import DEFAULT_PRIVATE, corpus_dir


def _registry(args: argparse.Namespace):
    from humanbench.sourcing.registry import load_registry

    return load_registry(Path(args.registry) if args.registry else None)


def _selected_sources(args: argparse.Namespace):
    reg = _registry(args)
    sources = reg.sources
    if args.channel:
        sources = [s for s in sources if s.channel == Channel(args.channel)]
        if not sources:
            why = reg.no_source.get(Channel(args.channel), "no sources registered")
            sys.exit(f"no public source for channel {args.channel}: {why}")
    if args.source:
        wanted = set(args.source)
        missing = wanted - {s.id for s in reg.sources}
        if missing:
            sys.exit(f"unknown source ids: {sorted(missing)}")
        sources = [s for s in sources if s.id in wanted]
    return reg, sources


def cmd_sources(args: argparse.Namespace) -> None:
    reg, sources = _selected_sources(args)
    print(f"{'channel':10} {'id':16} {'licence status':14} {'uses':22} repo")
    for s in sources:
        uses = ",".join(str(u) for u in s.allowed_uses)
        print(f"{s.channel!s:10} {s.id:16} {s.licence_status!s:14} {uses:22} {s.repo}@{s.revision[:8]}")
    for ch, why in reg.no_source.items():
        if not args.channel or ch == Channel(args.channel):
            print(f"{ch!s:10} {'(none)':16} {why}")


def cmd_verify(args: argparse.Namespace) -> None:
    import fnmatch

    from huggingface_hub import HfApi

    _, sources = _selected_sources(args)
    api = HfApi()
    bad = 0
    for s in sources:
        try:
            info = api.dataset_info(s.repo, revision=s.revision)
            files = [f.rfilename for f in info.siblings or []]
            n = sum(1 for f in files if any(fnmatch.fnmatch(f, g) for g in s.data_files)
                    and not any(fnmatch.fnmatch(f, g) for g in s.exclude_files))
            ok = info.sha == s.revision and n > 0
            print(f"{'ok  ' if ok else 'FAIL'} {s.id:16} {s.repo}@{s.revision[:8]} files={n} gated={info.gated}")
            bad += not ok
        except Exception as e:  # report every source, then exit non-zero
            print(f"FAIL {s.id:16} {s.repo}: {type(e).__name__}: {e}")
            bad += 1
    if bad:
        sys.exit(1)


def _parse_quotas(items: list[str] | None) -> dict[Band, int] | None:
    if not items:
        return None
    out: dict[Band, int] = {}
    for item in items:
        band, _, n = item.partition("=")
        out[Band(band)] = int(n)
    return out


def cmd_corpus(args: argparse.Namespace) -> None:
    from humanbench.sourcing.corpus import GateConfig, ResumeError, run_source, write_manifest
    from humanbench.sourcing.streams import HFParquetStream, needed_columns

    reg, sources = _selected_sources(args)
    out_dir = corpus_dir(Path(args.private_dir))
    override = _parse_quotas(args.quota)
    gates = GateConfig()
    failed = []
    for s in sources:
        quotas = {b: override.get(b, 0) for b in Band} if override else s.band_quotas(reg.default_quotas)
        cols = needed_columns([s.text_field, s.id_field, s.date_field, *[f.field for f in s.filters],
                               *s.meta_fields.values()])
        try:
            stream = HFParquetStream(s.repo, s.revision, s.data_files, s.exclude_files, columns=cols, seed_key=s.id)
            run_source(s, stream, out_dir, gates=gates, quotas=quotas, limit=args.limit, max_scan=args.max_scan,
                       restart=args.restart, checkpoint_every=args.checkpoint_every)
        except ResumeError as e:
            print(f"[{s.id}] {e}", file=sys.stderr)
            failed.append(s.id)
    path = write_manifest(out_dir, reg)
    print(f"manifest: {path}")
    if failed:
        sys.exit(f"not run: {failed}")


def cmd_manifest(args: argparse.Namespace) -> None:
    from humanbench.sourcing.corpus import write_manifest

    print(write_manifest(corpus_dir(Path(args.private_dir)), _registry(args)))


def cmd_brief(args: argparse.Namespace) -> None:
    from humanbench.sourcing.briefs import load_context, render_brief
    from humanbench.sourcing.refs import RefStore, load_card

    store = RefStore(Path(args.private_dir))
    for card_path in args.card:
        card = load_card(Path(card_path))
        include = {"auto": None, "yes": True, "no": False}[args.requirements]
        md = render_brief(card, context_text=load_context(card, Path(args.context_root)),
                          include_requirements=include)
        if args.stdout:
            print(md)
        else:
            print(store.write_brief(card.id, md))


def cmd_screener(args: argparse.Namespace) -> None:
    from humanbench.sourcing.refs import RefStore
    from humanbench.sourcing.screener import screener_for

    personas = load_personas()
    if not args.all:
        wanted = set(args.persona or [])
        personas = [p for p in personas if p.id in wanted]
        if missing := wanted - {p.id for p in personas}:
            sys.exit(f"unknown personas: {sorted(missing)}")
    store = RefStore(Path(args.private_dir))
    for p in personas:
        spec = screener_for(p)
        if args.stdout:
            print(spec.model_dump_json(indent=2))
        else:
            print(store.write_screener(spec))


def _load_detector(spec: str):
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)()


def cmd_qa(args: argparse.Namespace) -> None:
    from humanbench.sourcing.briefs import load_context
    from humanbench.sourcing.qa import QAContext, run_qa
    from humanbench.sourcing.refs import RefStore, load_card, read_model_outputs
    from humanbench.sourcing.submission import Submission

    store = RefStore(Path(args.private_dir))
    card = load_card(Path(args.card))
    sub = Submission.model_validate_json(Path(args.submission).read_text(encoding="utf-8"))
    store.save_submission(sub)
    context_text, material = _package_material(Path(args.card), card, Path(args.context_root))
    ctx = QAContext(
        card=card,
        model_outputs=read_model_outputs(Path(args.model_outputs)) if args.model_outputs else (),
        other_submissions=store.submissions(),
        detectors=[_load_detector(d) for d in args.detector or []],
        context_text=context_text,
        extra_material=material,
    )
    report = run_qa(sub, ctx)
    store.save_qa(report)
    if report.decision == "accept":
        store.accept(sub, card, decided_by="auto")
    print(f"{sub.submission_id}: {report.decision}")
    for g in report.gates:
        for r in g.reasons:
            print(f"  [{g.gate}:{g.outcome}] {r}")


def _package_material(card_path: Path, card, context_root: Path) -> tuple[str, tuple[str, ...]]:
    """Everything the writer was given, read the same way the model side reads it:
    the thread, the workspace items and the ghostwrite samples (TaskPackage.source_texts)."""
    from humanbench.taskpkg import PackageError, load_package

    try:
        pkg = load_package(card_path.parent, require_oracle=False)
    except PackageError:
        return load_context(card, context_root) or "", ()
    return pkg.thread_text(), tuple(pkg.source_texts())


def cmd_queue(args: argparse.Namespace) -> None:
    from humanbench.sourcing.refs import RefStore

    for report in RefStore(Path(args.private_dir)).review_queue():
        flagged = [f"{g.gate}: {r}" for g in report.gates for r in g.reasons if g.outcome in ("flag", "error")]
        print(json.dumps({"task_id": report.task_id, "submission_id": report.submission_id, "why": flagged}))


def cmd_review(args: argparse.Namespace) -> None:
    from humanbench.sourcing.refs import RefStore, Review, load_card

    card = load_card(Path(args.card))
    review = Review(submission_id=args.submission_id, task_id=card.id, reviewer=args.reviewer,
                    decision=args.decision, note=args.note, reviewed_at=dt.datetime.now(dt.UTC))
    print(RefStore(Path(args.private_dir)).record_review(review, card))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="humanbench-source", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--private-dir", default=str(DEFAULT_PRIVATE), help="gitignored output root (default: private)")
    ap.add_argument("--registry", help="alternative sources.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def with_selection(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--channel", choices=[c.value for c in Channel])
        p.add_argument("--source", action="append", help="source id (repeatable)")
        return p

    with_selection(sub.add_parser("sources", help="list the registry")).set_defaults(fn=cmd_sources)
    with_selection(sub.add_parser("verify", help="check every repo and pinned revision on Hugging Face")).set_defaults(fn=cmd_verify)

    p = with_selection(sub.add_parser("corpus", help="stream, gate and write private/corpus/<source>.jsonl"))
    p.add_argument("--limit", type=int, help="accepted records per source (total, across resumes)")
    p.add_argument("--max-scan", type=int, default=50_000, help="rows read per source (total, across resumes)")
    p.add_argument("--quota", action="append", metavar="BAND=N", help="override per-band quotas (forces --restart)")
    p.add_argument("--restart", action="store_true", help="discard this source's output and state first")
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.set_defaults(fn=cmd_corpus)

    sub.add_parser("manifest", help="rebuild private/corpus/manifest.json").set_defaults(fn=cmd_manifest)

    p = sub.add_parser("brief", help="writer brief for a task card")
    p.add_argument("--card", action="append", required=True)
    p.add_argument("--context-root", default=".", help="directory that card context paths are relative to")
    p.add_argument("--requirements", choices=["auto", "yes", "no"], default="auto",
                   help="show must_include / must_not_include (auto: yes for direct/thread, no for agent/ghostwrite)")
    p.add_argument("--stdout", action="store_true")
    p.set_defaults(fn=cmd_brief)

    p = sub.add_parser("screener", help="recruiting criteria for personas")
    p.add_argument("--persona", action="append")
    p.add_argument("--all", action="store_true")
    p.add_argument("--stdout", action="store_true")
    p.set_defaults(fn=cmd_screener)

    p = sub.add_parser("qa", help="run QA gates on a submission")
    p.add_argument("--card", required=True)
    p.add_argument("--submission", required=True)
    p.add_argument("--model-outputs", help="dir of model outputs for this task (copy check)")
    p.add_argument("--detector", action="append", metavar="MODULE:CLASS", help="AI-detector screen (review only)")
    p.add_argument("--context-root", default=".")
    p.set_defaults(fn=cmd_qa)

    sub.add_parser("queue", help="list submissions waiting for human review").set_defaults(fn=cmd_queue)

    p = sub.add_parser("review", help="record a person's decision on a needs_human_review submission")
    p.add_argument("--card", required=True)
    p.add_argument("--submission-id", required=True)
    p.add_argument("--reviewer", required=True)
    p.add_argument("--decision", choices=["accept", "reject"], required=True)
    p.add_argument("--note", required=True)
    p.set_defaults(fn=cmd_review)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
