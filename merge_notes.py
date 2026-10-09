"""Runs INSIDE the sandbox (standard library only): python3 merge_notes.py

Deterministic merge of the researchers' notes (/tmp/work/research/notes/*.md) into
  * sources.json      - JSON array of {n, id, url, title, date, source}, numbered from 1, no duplicate URLs;
  * notes_merged.md   - every note block once, headed by its number [n], for the lead to write the report from.
It also enforces the URL <-> family contract (arxiv -> https://arxiv.org/abs/<id>, hf-* -> https://huggingface.co/papers/<id>)
and prints the catalogue plus the count per source family, so the lead can check that >= 3 families are covered.

Note format (written by the researcher subagents), one block per source:
    ### SOURCE
    title: <paper or page title>
    id: <arXiv id, HF paper id, or the URL for web pages>
    url: <exact URL returned by the tool>
    date: <YYYY-MM-DD or n.d.>
    source: <arxiv | hf-daily | hf-search | web>
    points:
    - <fact taken from the retrieved text>
"""
import glob
import json
import re
import sys

NOTES_GLOB = "/tmp/work/research/notes/*.md"
SOURCES = "/tmp/work/research/sources.json"
MERGED = "/tmp/work/research/notes_merged.md"
FAMILIES = ("arxiv", "hf-daily", "hf-search", "web")
FIELDS = ("title", "id", "url", "date", "source")
_ARXIV = re.compile(r"^https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf)/([^?#\s]+?)(?:v\d+)?(?:\.pdf)?/?$")
_HF = re.compile(r"^https?://(?:www\.)?huggingface\.co/papers/([^?#/\s]+)/?$")


def parse_blocks(text):
    """Return [{title, id, url, date, source, points: [...]}] for every '### SOURCE' block of one notes file."""
    blocks = []
    for chunk in re.split(r"(?mi)^#{2,4}\s*SOURCE\b.*$", text)[1:]:
        points, in_points = [], False
        entry = {"points": points}
        for raw in chunk.splitlines():
            line = raw.strip().lstrip("-*").strip().replace("**", "")
            key, sep, value = line.partition(":")
            key = key.strip().lower()
            if sep and key in FIELDS and not in_points:
                entry[key] = value.strip().strip("<>`\"'")
            elif sep and key in ("points", "key points"):
                in_points = True
                if value.strip():
                    points.append(value.strip())
            elif line and in_points:
                points.append(line)
        if entry.get("url"):
            blocks.append(entry)
    return blocks


def normalise(entry):
    """Enforce the URL <-> family contract. Returns the entry, or None when the URL is unusable."""
    url = entry["url"].strip().rstrip(".,;)")
    if not url.startswith(("http://", "https://")):
        return None
    source = entry.get("source", "").strip().lower()
    arxiv, hf = _ARXIV.match(url), _HF.match(url)
    if source == "arxiv":
        if not arxiv:
            source = "web"  # not an arXiv abstract URL: it cannot be labelled arxiv
        else:
            entry["id"], url = arxiv.group(1), f"https://arxiv.org/abs/{arxiv.group(1)}"
    elif source in ("hf-daily", "hf-search"):
        if not hf:
            source = "web"
        else:
            entry["id"], url = hf.group(1), f"https://huggingface.co/papers/{hf.group(1)}"
    elif source not in FAMILIES:
        source = "web"
    if source == "web" and arxiv:
        url = f"https://arxiv.org/abs/{arxiv.group(1)}"  # same paper, canonical URL (still found by web search)
    date = entry.get("date", "")
    entry.update(url=url, source=source, date=date[:10] if re.match(r"\d{4}-\d{2}-\d{2}", date) else "n.d.",
                 id=entry.get("id") or url, title=" ".join(entry.get("title", "").split()) or url)
    return entry


def main():
    merged = {}
    files = sorted(glob.glob(NOTES_GLOB))
    for path in files:
        with open(path, encoding="utf-8") as f:
            for block in parse_blocks(f.read()):
                entry = normalise(block)
                if entry is None:
                    continue
                if entry["url"] in merged:
                    merged[entry["url"]]["points"] += [p for p in entry["points"] if p not in merged[entry["url"]]["points"]]
                else:
                    merged[entry["url"]] = entry
    if not merged:
        print(f"NO SOURCES: found {len(files)} notes files but no valid '### SOURCE' block with a url")
        return 1
    sources, lines = [], []
    for n, entry in enumerate(merged.values(), start=1):
        sources.append({"n": n, **{k: entry[k] for k in ("id", "url", "title", "date", "source")}})
        lines.append(f"## [{n}] {entry['title']}\nsource: {entry['source']} | date: {entry['date']} | url: {entry['url']}")
        lines += [f"- {p}" for p in entry["points"]] + [""]
    with open(SOURCES, "w", encoding="utf-8") as f:
        json.dump(sources, f, ensure_ascii=False, indent=2)
    with open(MERGED, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    counts = {family: sum(s["source"] == family for s in sources) for family in FAMILIES}
    print(f"MERGED {len(files)} notes files -> {len(sources)} sources in {SOURCES}; notes in {MERGED}")
    print("families: " + ", ".join(f"{k}={v}" for k, v in counts.items())
          + f"  ({sum(v > 0 for v in counts.values())} of 4 families present)")
    for s in sources:
        print(f"[{s['n']}] {s['source']} {s['date']} {s['title'][:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
