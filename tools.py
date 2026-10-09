"""tools.py - Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
SUMMARY_CHARS = 600
FETCH_CHARS = 12_000
TIMEOUT = httpx.Timeout(60.0, connect=15.0)
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_MIN_INTERVAL = 3.0  # arXiv API etiquette: at least 3 s between two calls
ARXIV_STOPWORDS = {"and", "or", "not", "andnot", "all"}  # boolean operators / field prefixes the LLM may add


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again (at most `attempts` calls in total).

    Wait = the server's Retry-After when given, else exponential backoff base * 2**attempt with "equal jitter"
    (a random value between half and all of it); always capped at `cap`. No sleep after the last attempt: the
    error is re-raised. Any other exception is not retried.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(float(exc.retry_after), cap)
            else:
                delay = min(base * 2 ** attempt, cap) * random.uniform(0.5, 1.0)
            time.sleep(max(delay, 0.0))


def _retry_after(response):
    """Retry-After header in seconds (the delta-seconds form), else None."""
    try:
        return max(float(response.headers["Retry-After"]), 0.0)
    except (KeyError, ValueError):
        return None


def _request(method, url, **kwargs):
    """One HTTP call. Transient failures (network errors, 429/5xx) become RetryableError; other HTTP errors raise."""
    try:
        response = httpx.request(method, url, timeout=TIMEOUT, follow_redirects=True, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code in RETRYABLE_STATUS:
        raise RetryableError(f"HTTP {response.status_code} from {response.url.host}", _retry_after(response))
    response.raise_for_status()
    return response


def _clean(text):
    return " ".join(str(text or "").split())


def _short(text, limit=SUMMARY_CHARS):
    text = _clean(text)
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _clamp(value, low, high):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return low


def _error(exc):
    return _redact(f"ERROR: {type(exc).__name__}: {exc}")


def _records(records):
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


# ---- TODO 2: arXiv ----
_arxiv_lock = threading.Lock()  # subagents run in parallel: the 3 s spacing must hold across threads
_arxiv_last_call = 0.0


def _arxiv_get(params):
    """GET the arXiv API, never sooner than ARXIV_MIN_INTERVAL after the previous call."""
    global _arxiv_last_call
    with _arxiv_lock:
        wait = _arxiv_last_call + ARXIV_MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            return _request("GET", ARXIV_URL, params=params)
        finally:
            _arxiv_last_call = time.monotonic()


def _arxiv_terms(query):
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", str(query or ""))
    return [w for w in words if w.lower() not in ARXIV_STOPWORDS][:10]


def _parse_arxiv(xml_text):
    records = []
    for entry in ET.fromstring(xml_text).iter(f"{ATOM}entry"):
        raw_id = entry.findtext(f"{ATOM}id") or ""
        if "/abs/" not in raw_id:  # arXiv reports query errors as an entry without an abstract id
            continue
        paper_id = re.sub(r"v\d+$", "", raw_id.split("/abs/", 1)[1].strip())
        records.append({
            "id": paper_id,
            "url": f"https://arxiv.org/abs/{paper_id}",
            "published": (entry.findtext(f"{ATOM}published") or "")[:10],
            "title": _clean(entry.findtext(f"{ATOM}title")),
            "summary": _short(entry.findtext(f"{ATOM}summary")),
        })
    return records


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by a few plain keywords (e.g. "world model video prediction"), newest first.
    All keywords must match, so keep the query short (2-5 words); no quotes, field prefixes or AND/OR.
    Returns a JSON list of {id, url, published, title, summary}; url is https://arxiv.org/abs/<id>.
    Returns "NO RESULTS" when nothing matches (use fewer/broader keywords) and "ERROR: ..." when arXiv is unavailable."""
    terms = _arxiv_terms(query)
    if not terms:
        return "NO RESULTS"
    params = {"search_query": " AND ".join(f"all:{t}" for t in terms), "sortBy": "submittedDate",
              "sortOrder": "descending", "start": 0, "max_results": _clamp(max_results, 1, 30)}
    try:
        response = with_retry(lambda: _arxiv_get(params), attempts=6, base=3.0, cap=60.0)
        return _records(_parse_arxiv(response.text))
    except Exception as exc:  # noqa: BLE001 - a tool never raises
        return _error(exc)


# ---- TODO 3: Hugging Face ----
def _hf_record(item):
    """Daily/search item -> compact record, or None when it has no paper id."""
    paper = item.get("paper") or {}
    paper_id = paper.get("id")
    if not paper_id:
        return None
    return {
        "id": paper_id,
        "url": f"https://huggingface.co/papers/{paper_id}",
        "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
        "title": _clean(paper.get("title") or item.get("title")),
        "summary": _short(paper.get("ai_summary") or paper.get("summary") or item.get("summary")),
        "upvotes": paper.get("upvotes") or 0,
        "github": paper.get("githubRepo") or "",
        "stars": paper.get("githubStars") or 0,
    }


def _hf_get(url, params):
    response = with_retry(lambda: _request("GET", url, params=params), attempts=5, base=1.0, cap=30.0)
    items = response.json()
    if not isinstance(items, list):
        raise ValueError(f"unexpected Hugging Face answer: {str(items)[:200]}")
    return [record for record in map(_hf_record, items) if record]


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research right now (community upvotes, GitHub repo, stars).
    Returns a JSON list of {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes;
    url is https://huggingface.co/papers/<id>. `date` is YYYY-MM-DD (empty = latest day). `keyword` keeps only papers
    whose title/summary contains all its words (case-insensitive); keep it to 1-2 words. There is no topic search on
    this endpoint: use hf_search_papers to search a topic. Returns "NO RESULTS" when nothing matches."""
    params = {"limit": _clamp(limit, 1, 100)}
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date or "").strip()):
        params["date"] = date.strip()
    try:
        records = _hf_get(HF_DAILY_URL, params)
    except Exception as exc:  # noqa: BLE001
        return _error(exc)
    words = str(keyword or "").lower().split()
    if words:
        records = [r for r in records if all(w in f"{r['title']} {r['summary']}".lower() for w in words)]
    records.sort(key=lambda r: r["upvotes"], reverse=True)
    return _records(records)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic (search over the arXiv papers indexed on huggingface.co/papers).
    Returns a JSON list of {id, url, published, title, summary, upvotes, github, stars}; url is
    https://huggingface.co/papers/<id> (the id is the arXiv id). Returns "NO RESULTS" when nothing matches."""
    query = _clean(query)
    if not query:
        return "NO RESULTS"
    limit = _clamp(limit, 1, 50)
    try:
        return _records(_hf_get(HF_SEARCH_URL, {"q": query, "limit": limit})[:limit])
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _exa_key():
    return (os.getenv("EXA_API_KEY") or "").strip()


def _redact(text):
    """Never let the Exa key reach the agent (httpx error messages contain the full URL)."""
    key = _exa_key()
    if key:
        text = text.replace(key, "***")
    return re.sub(r"(exaApiKey=)[^&\s'\"]+", r"\1***", text)


def _rate_limited(result):
    """The free tier answers HTTP 200 and flags the limit in result._meta (e.g. {"ai.exa/rateLimited": true})."""
    meta = result.get("_meta") or {}
    return any(v and "ratelimit" in re.sub(r"[^a-z]", "", str(k).lower()) for k, v in meta.items())


def _exa_call_once(name, arguments):
    key = _exa_key()
    response = _request(
        "POST", EXA_URL, params={"exaApiKey": key} if key else None,
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
    message = None
    if response.text.lstrip().startswith("{"):  # plain JSON answer
        message = response.json()
    else:  # server-sent events: the JSON-RPC message is on a "data:" line
        for line in response.text.splitlines():
            if line.startswith("data:") and line[5:].strip():
                message = json.loads(line[5:])
    if message is None:
        raise RetryableError("empty answer from Exa")
    if "error" in message:
        raise RuntimeError(f"Exa JSON-RPC error: {message['error']}")
    result = message.get("result") or {}
    if _rate_limited(result):
        raise RetryableError("Exa rate limit")
    text = "\n".join(part.get("text", "") for part in result.get("content", []) if part.get("type") == "text")
    if result.get("isError"):
        raise RuntimeError(f"Exa tool error: {text[:300]}")
    return text.strip()


def _exa_call(name, arguments):
    return with_retry(lambda: _exa_call_once(name, arguments), attempts=6, base=5.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa): blogs, surveys, project pages, docs, benchmarks, news. `query` = short search query;
    `objective` = one sentence describing the ideal page (what you want to learn from it). Returns clean text of the
    top results, each with Title, URL, Published date and highlights. Returns "NO RESULTS" when nothing is found."""
    query = _clean(query)
    if not query:
        return "NO RESULTS"
    arguments = {"query": query, "objective": _clean(objective) or f"Find authoritative pages about: {query}",
                 "numResults": _clamp(num_results, 1, 10)}
    try:
        return _exa_call("web_search_exa", arguments) or "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of ONE web page (e.g. an arXiv abstract page, a blog post, a project page) as markdown,
    given its exact http(s) URL. Long pages are truncated to about 12000 characters."""
    url = str(url or "").strip()
    if not url.startswith(("http://", "https://")):
        return "ERROR: url must start with http:// or https://"
    try:
        text = _exa_call("web_fetch_exa", {"urls": [url]})
    except Exception as exc:  # noqa: BLE001
        return _error(exc)
    if not text:
        return "NO RESULTS"
    return text if len(text) <= FETCH_CHARS else text[:FETCH_CHARS] + "\n...[truncated]"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
