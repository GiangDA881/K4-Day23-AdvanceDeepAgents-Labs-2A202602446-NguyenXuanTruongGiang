"""research.py - The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from langchain_core.rate_limiters import InMemoryRateLimiter

from agents import (FINALIZER_PATH, MERGER_PATH, RECURSION_LIMIT, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR,
                    build_lead_agent)
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"                                # debugging traces (git-ignored)
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator
MERGER_SOURCE = ROOT / "merge_notes.py"             # deterministic notes -> sources.json merge


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(topic).lower()).strip("-")[:60].strip("-")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (f"Research topic: {topic}\n\n"
            "Produce the cited survey report for this topic by following your workflow from step 1 to step 9. "
            "Delegate the research to `researcher` subagents in parallel, merge their notes, write the report body, "
            "finalize the citations and run the validator until it prints OK.")


def _model_name(model):
    return getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.
    Lead messages only: subagent tokens are not included, so this undercounts the real cost."""
    calls, tokens = Counter(), Counter()
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls[call["name"]] += 1
        usage = getattr(message, "usage_metadata", None) or {}
        tokens["input"] += usage.get("input_tokens", 0)
        tokens["output"] += usage.get("output_tokens", 0)
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": calls["task"],
            "tool_calls": dict(calls), "tokens": {"input": tokens["input"], "output": tokens["output"]}}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.
    A failed run (missing/empty report, missing/invalid sources.json) raises RuntimeError and writes nothing."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report = (files.get(REPORT_PATH) or b"").decode("utf-8", errors="replace")
    if not report.strip():
        raise RuntimeError(f"the agent produced no report ({REPORT_PATH} missing or empty)")
    try:
        sources = json.loads((files.get(SOURCES_PATH) or b"").decode("utf-8"))
    except ValueError as exc:
        raise RuntimeError(f"{SOURCES_PATH} is missing or not valid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError(f"{SOURCES_PATH} is not a non-empty JSON list")

    meta = {"topic": topic, **summarize(messages, elapsed, model_name), "n_sources": len(sources),
            "source_families": sorted({str(s.get("source")) for s in sources if isinstance(s, dict) and s.get("source")})}
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = reports_dir / slugify(topic)
    stem.with_name(f"{stem.name}.sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2),
                                                           encoding="utf-8")
    stem.with_name(f"{stem.name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
    report_path = stem.with_name(f"{stem.name}.md")
    report_path.write_text(report, encoding="utf-8")
    return report_path


def save_trace(messages, topic, logs_dir=LOGS):
    """Write the lead's transcript (tool calls + truncated contents) to logs/<slug>.trace.md for debugging."""
    lines = []
    for message in messages:
        lines.append(f"### {type(message).__name__} {getattr(message, 'name', '') or ''}".rstrip())
        lines.append(str(message.content)[:3000])
        if getattr(message, "response_metadata", None):
            lines.append(f"(metadata: {json.dumps(message.response_metadata, default=str)[:500]})")
        lines += [f"-> {call['name']}({json.dumps(call['args'], ensure_ascii=False)[:1500]})"
                  for call in getattr(message, "tool_calls", None) or []]
    logs_dir.mkdir(exist_ok=True)
    (logs_dir / f"{slugify(topic)}.trace.md").write_text("\n".join(lines), encoding="utf-8")


def limit_rate(model):
    """Optional client-side throttle (env LAB_MAX_RPM): free LLM tiers allow ~15 requests/minute, and the parallel
    researchers share one model object, so every model call of the lead and the subagents goes through this limiter."""
    rpm = float(os.getenv("LAB_MAX_RPM") or 0)
    if rpm > 0:
        model.rate_limiter = InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.1,
                                                 max_bucket_size=1)
    return model


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = (topic or "").strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    model = limit_rate(make_model())
    start = time.monotonic()
    with open_sandbox() as backend:  # the sandbox is always stopped and removed, even on errors
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                         FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                         MERGER_PATH: MERGER_SOURCE.read_bytes()})
        agent = build_lead_agent(backend, model)
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                  config={"recursion_limit": RECURSION_LIMIT})
            save_trace(result["messages"], topic)
            check = backend.execute(f"python3 {VALIDATOR_PATH}")  # never save a report whose citations do not resolve
            if check.exit_code != 0:
                raise RuntimeError(f"the report does not pass the citation validator:\n{check.output.strip()}")
            report_path = save_outputs(backend, topic, result["messages"], time.monotonic() - start,
                                       _model_name(model))
        except Exception as exc:  # noqa: BLE001 - e.g. GraphRecursionError, missing report: a failed run writes nothing
            print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    print(f"report saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
