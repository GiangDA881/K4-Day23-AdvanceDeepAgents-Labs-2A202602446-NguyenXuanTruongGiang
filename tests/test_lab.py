"""Offline tests (no network, no LLM):   pip install pytest && python -m pytest -q"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_citations  # noqa: E402
import finalize_citations  # noqa: E402
import merge_notes  # noqa: E402
import research  # noqa: E402
import tools  # noqa: E402
from tools import RetryableError, with_retry  # noqa: E402


# ---- with_retry ----
@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(tools.time, "sleep", recorded.append)
    return recorded


def flaky(failures, retry_after=None):
    state = {"calls": 0}

    def fn():
        state["calls"] += 1
        if state["calls"] <= failures:
            raise RetryableError("busy", retry_after)
        return "ok"
    return fn, state


def test_retry_succeeds_after_transient_errors(sleeps):
    fn, state = flaky(2)
    assert with_retry(fn, attempts=5, base=1.0, cap=30.0) == "ok"
    assert state["calls"] == 3 and len(sleeps) == 2
    assert 0.5 <= sleeps[0] <= 1.0 and 1.0 <= sleeps[1] <= 2.0  # exponential with jitter


def test_retry_gives_up_without_sleeping_after_last_attempt(sleeps):
    fn, state = flaky(10)
    with pytest.raises(RetryableError):
        with_retry(fn, attempts=3)
    assert state["calls"] == 3 and len(sleeps) == 2


def test_retry_after_is_respected_and_capped(sleeps):
    fn, _ = flaky(2, retry_after=7)
    with_retry(fn, attempts=3, cap=5.0)
    assert sleeps == [5.0, 5.0]
    fn, _ = flaky(1, retry_after=2)
    with_retry(fn, attempts=3, cap=30.0)
    assert sleeps[-1] == 2


def test_backoff_capped(sleeps):
    fn, _ = flaky(6)
    with_retry(fn, attempts=7, base=10.0, cap=15.0)
    assert max(sleeps) <= 15.0


def test_other_errors_are_not_retried(sleeps):
    state = {"calls": 0}

    def boom():
        state["calls"] += 1
        raise KeyError("bug")
    with pytest.raises(KeyError):
        with_retry(boom)
    assert state["calls"] == 1 and sleeps == []


# ---- tools helpers ----
def test_arxiv_terms_sanitised():
    assert tools._arxiv_terms('"world model" AND all:video OR ti:"x-ray"') == ["world", "model", "video", "ti", "x-ray"]
    assert tools._arxiv_terms(' "" :: ') == []
    assert tools.arxiv_search.invoke({"query": '"" AND'}) == "NO RESULTS"


def test_exa_rate_limit_flag_and_redaction(monkeypatch):
    assert tools._rate_limited({"_meta": {"ai.exa/rateLimited": True}})
    assert not tools._rate_limited({"_meta": {}})
    monkeypatch.setenv("EXA_API_KEY", "secret123")
    assert "secret123" not in tools._redact("ConnectError for url https://mcp.exa.ai/mcp?exaApiKey=secret123")


def test_parse_arxiv_strips_version():
    xml = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2501.00001v2</id>'
           '<published>2025-01-02T00:00:00Z</published><title> A\n  title </title><summary> s \n x</summary></entry></feed>')
    assert tools._parse_arxiv(xml) == [{"id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001",
                                        "published": "2025-01-02", "title": "A title", "summary": "s x"}]


# ---- slugify ----
@pytest.mark.parametrize("topic,slug", [
    ("survey about world model", "survey-about-world-model"),
    ("../../x", "x"),
    ("", "topic"),
    ("!!!", "topic"),
])
def test_slugify(topic, slug):
    assert research.slugify(topic) == slug


def test_slugify_length():
    assert len(research.slugify("a " * 100)) <= 60


# ---- check_citations ----
SOURCES = [{"n": 1, "url": "https://arxiv.org/abs/1", "title": "A", "source": "arxiv"},
           {"n": 2, "url": "https://huggingface.co/papers/2", "title": "B", "source": "hf-search"}]
GOOD = ("# T\n\nClaim [1]. Other [2].\n\n## References\n"
        "[1] A. arxiv. https://arxiv.org/abs/1 (2025-01-01)\n[2] B. hf-search. https://huggingface.co/papers/2 (n.d.)\n")


def test_check_ok():
    assert check_citations.check(GOOD, SOURCES) == []


def test_check_groups_count_as_citations():
    report = GOOD.replace("Claim [1]. Other [2].", "Claim [1, 2]. Again [1-2].")
    assert check_citations.check(report, SOURCES) == []


@pytest.mark.parametrize("report,needle", [
    (GOOD.replace("## References", "## Refs"), "References"),
    (GOOD.replace("Other [2].", "Other [3]."), "[3] cited but missing"),
    (GOOD.replace("Other [2].", "Other."), "source [2] never cited"),
    (GOOD.replace("[2] B. hf-search. https://huggingface.co/papers/2 (n.d.)\n", ""), "source [2] has no line"),
    (GOOD + "[2] B again. https://huggingface.co/papers/2\n", "listed more than once"),
    (GOOD + "[9] Ghost. https://x.org\n", "not a source"),
    (GOOD.replace("https://arxiv.org/abs/1 (", "https://arxiv.org/abs/1; C https://c.org ("), "exactly one URL"),
    (GOOD.replace("https://arxiv.org/abs/1 (", "https://arxiv.org/abs/9 ("), "!="),
    (GOOD.replace("Other [2].", "Other `[2]` [x](u) [2](https://l)."), "source [2] never cited"),
])
def test_check_problems(report, needle):
    assert any(needle in p for p in check_citations.check(report, SOURCES))


def test_check_source_problems():
    assert check_citations.check(GOOD, []) == ["no sources in sources.json"]
    bad = [dict(SOURCES[0], n="1"), SOURCES[1], dict(SOURCES[1], n=3), dict(SOURCES[0], n=4, url="ftp://x")]
    problems = " | ".join(check_citations.check(GOOD, bad))
    assert "not an integer" in problems and "duplicates" in problems and "ftp://x" in problems


def test_finalizer_output_passes_validator():
    body = "# T\n\nA [2]. B [1, 2]. C [3].\n"
    sources = SOURCES + [{"n": 3, "url": "https://web.example/x", "title": "C", "source": "web"},
                         {"n": 4, "url": "https://unused.example", "title": "D", "source": "web"}]
    report, new_sources, problems = finalize_citations.finalize(body, sources)
    assert problems == [] and check_citations.check(report, new_sources) == []


# ---- merge_notes ----
NOTES = """# sub-question
### SOURCE
title: World Models
id: 1803.10122v4
url: http://arxiv.org/abs/1803.10122v4
date: 2018-03-27
source: arxiv
points:
- VAE + MDN-RNN + controller.
- key: value inside a point
### SOURCE
- **title:** A blog
- **url:** https://blog.example/post
- **date:** n.d.
- **source:** hf-search
- **points:**
  - mislabelled family
"""


def test_merge_notes_parse_and_normalise():
    blocks = [merge_notes.normalise(b) for b in merge_notes.parse_blocks(NOTES)]
    assert blocks[0]["url"] == "https://arxiv.org/abs/1803.10122" and blocks[0]["id"] == "1803.10122"
    assert blocks[0]["points"] == ["VAE + MDN-RNN + controller.", "key: value inside a point"]
    assert blocks[1]["source"] == "web" and blocks[1]["date"] == "n.d." and blocks[1]["points"] == ["mislabelled family"]


# ---- save_outputs ----
class FakeBackend:
    def __init__(self, files):
        self.files = files

    def download_files(self, paths):
        from deepagents.backends.protocol import FileDownloadResponse
        return [FileDownloadResponse(path=p, content=self.files.get(p), error=None) for p in paths]


@pytest.mark.parametrize("files", [
    {},
    {research.REPORT_PATH: b"  ", research.SOURCES_PATH: b"[]"},
    {research.REPORT_PATH: b"# r", research.SOURCES_PATH: b"not json"},
    {research.REPORT_PATH: b"# r", research.SOURCES_PATH: b"[]"},
])
def test_save_outputs_writes_nothing_on_failure(tmp_path, files):
    with pytest.raises(RuntimeError):
        research.save_outputs(FakeBackend(files), "t", [], 1.0, "m", reports_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_save_outputs_meta(tmp_path):
    from langchain_core.messages import AIMessage
    messages = [AIMessage(content="", tool_calls=[{"name": "task", "args": {}, "id": str(i)} for i in range(3)],
                          usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})]
    files = {research.REPORT_PATH: GOOD.encode(), research.SOURCES_PATH: json.dumps(SOURCES).encode()}
    path = research.save_outputs(FakeBackend(files), "My Topic", messages, 12.34, "m", reports_dir=tmp_path)
    meta = json.loads((tmp_path / "my-topic.meta.json").read_text())
    assert path.name == "my-topic.md" and meta["subagent_calls"] == 3 and meta["elapsed_s"] == 12.3
    assert meta["source_families"] == ["arxiv", "hf-search"] and meta["tokens"] == {"input": 10, "output": 2}
