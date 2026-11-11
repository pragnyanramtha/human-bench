"""Run a model using an OpenAI-compatible chat completions endpoint."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import urllib.request
from pathlib import Path

from humanbench.harness.cards import (TRIALS, TRACKS, _done, _write_json, approx_tokens,
                                      fingerprint, load_tasks, score_output, split_dir)
from humanbench.taskpkg import AGENTIC


def strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.split("</think>")[-1].strip()


def api_call(prompt: str, model: str, system: str, base_url: str, key: str,
             timeout: float = 900, max_tokens: int = 16000, effort: str | None = None) -> dict:
    body = {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        if not isinstance(text, str) or choice.get("finish_reason") is None:
            raise ValueError("incomplete response")
        usage = data.get("usage") or {}
        return {"ok": True, "text": strip_reasoning(text), "error": None,
                "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                          "output_tokens": usage.get("completion_tokens", 0),
                          "finish_reason": choice["finish_reason"]}}
    except Exception as exc:  # network and malformed responses are infrastructure failures
        return {"ok": False, "text": "", "usage": {}, "error": f"{type(exc).__name__}: {exc}"[:600]}


def configured_call(prompt: str, model: str, system: str, timeout: float = 600) -> dict:
    base = os.environ.get("HUMANBENCH_JUDGE_BASE_URL")
    key_env = os.environ.get("HUMANBENCH_JUDGE_KEY_ENV")
    if not base or not key_env or key_env not in os.environ:
        raise SystemExit("set HUMANBENCH_JUDGE_BASE_URL and HUMANBENCH_JUDGE_KEY_ENV (the name of the key variable)")
    return api_call(prompt, model, system, base, os.environ[key_env], timeout)


async def run(a: argparse.Namespace) -> None:
    if a.api_key_env not in os.environ:
        raise SystemExit(f"set {a.api_key_env} to the API key")
    if a.samples < 1 or a.concurrency < 1 or a.max_tokens < 1:
        raise SystemExit("samples, concurrency and max-tokens must be positive")
    name = a.model.split("/")[-1]
    if not name or name in (".", "..") or "/" in a.run_id or a.run_id in (".", ".."):
        raise SystemExit("invalid model or run id")
    run_dir = TRIALS / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    track = run_dir / "track.txt"
    if track.exists() and track.read_text().strip() != a.track:
        raise SystemExit("use another --run-id for a different track")
    track.write_text(a.track + "\n")
    tasks = load_tasks([split_dir(a.split)])
    if a.tasks:
        selected = set(a.tasks.split(","))
        if selected - tasks.keys():
            raise SystemExit(f"unknown tasks: {sorted(selected - tasks.keys())}")
        tasks = {k: v for k, v in tasks.items() if k in selected}
    fps = {k: fingerprint(v) for k, v in tasks.items()}
    system = TRACKS[a.track]
    _write_json(run_dir / "invocations" / f"{time.time_ns()}-{os.getpid()}.json", {
        "models": [name], "routes": {name: {"route": "api", "model_id": a.model}},
        "effort": {name: a.effort or "provider default"}, "system_line": system, "track": a.track,
        "samples": a.samples, "tasks": sorted(tasks), "fingerprints": fps, "max_tokens": a.max_tokens})
    sem = asyncio.Semaphore(a.concurrency)

    async def one(task: str, sample: int) -> None:
        pkg, fp = tasks[task], fps[task]
        out = run_dir / "raw" / name / task / f"s{sample}.json"
        claim = run_dir / "claims" / f"{name}__{task}__s{sample}"
        if _done(out, fp):
            return
        claim.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - claim.stat().st_mtime < 3600:
                return
            claim.unlink(missing_ok=True)
            return await one(task, sample)
        os.close(fd)
        try:
            prompt = pkg.model_prompt()
            async with sem:
                start = time.monotonic()
                res = await asyncio.to_thread(api_call, prompt, a.model, system, a.base_url,
                                              os.environ[a.api_key_env], a.timeout, a.max_tokens, a.effort)
                seconds = round(time.monotonic() - start, 1)
            with (run_dir / "attempts.jsonl").open("a") as f:
                f.write(json.dumps({"model": name, "task": task, "sample": sample, "seconds": seconds,
                                    "ok": res["ok"], "usage": res["usage"], "at": time.time(),
                                    "outcome": "accepted" if res["ok"] else "failed"}) + "\n")
            record = {"run_id": a.run_id, "model": name, "model_id": a.model, "route": "api",
                      "effort": a.effort or "provider default", "task": task, "sample": sample,
                      "channel": pkg.card.channel.value, "mode": pkg.card.mode.value, "tone": pkg.card.tone.value,
                      "band": pkg.card.band.value, "track": a.track, "card_fingerprint": fp, "seconds": seconds,
                      "rc": 0 if res["ok"] else 1, "usage": res["usage"], "used_tools": {},
                      "tools_simulated": pkg.card.mode in AGENTIC, "prompt_tokens_approx": approx_tokens(prompt),
                      "output": res["text"], "infra_failure": not res["ok"], "error_tail": res["error"],
                      "max_tokens": a.max_tokens}
            if res["ok"]:
                record.update(score_output(pkg, res["text"]))
            _write_json(out, record)
        finally:
            claim.unlink(missing_ok=True)
    for sample in range(1, a.samples + 1):
        await asyncio.gather(*(one(task, sample) for task in sorted(tasks)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for flag in ("run-id", "model", "base-url", "api-key-env"):
        r.add_argument("--" + flag, required=True)
    r.add_argument("--split", choices=["dev", "ref"], default="dev")
    r.add_argument("--samples", type=int, default=3)
    r.add_argument("--tasks")
    r.add_argument("--effort")
    r.add_argument("--max-tokens", type=int, default=16000)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--timeout", type=float, default=900)
    r.add_argument("--track", choices=sorted(TRACKS), default="instructed")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
