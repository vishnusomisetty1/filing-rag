"""Parse an earnings call transcript into speaker turns and Q&A exchanges.

Handles the common "Name -- Title" / "Name -- Firm -- Analyst" layout (Motley Fool style)
and "Name: text" layouts. Prepared remarks become one chunk per turn; the Q&A becomes one
chunk per analyst exchange (question plus every executive answer until the next analyst).

Usage: python transcript_parser.py call.txt --ticker AAPL --period "Q3 FY2025"
"""
import re
import sys
import argparse
import pathlib
from dataclasses import dataclass

CHUNK_CHARS = 2200
NAME = r"[A-Z][A-Za-z.'\-]+(?: [A-Z][A-Za-z.'\-]+){0,4}"
DASHED_RE = re.compile(rf"^({NAME})\s+(?:--|—|–|-)\s+(.{{2,120}})$")
COLON_RE = re.compile(rf"^({NAME})(?:\s*\(([^)]{{2,80}})\))?:\s*(.*)$")
QA_START_RE = re.compile(r"^(questions?\s*(and|&)\s*answers?|question-and-answer session)\b", re.I)
PREPARED_RE = re.compile(r"^prepared remarks\b", re.I)
PARTICIPANTS_RE = re.compile(r"^(call participants|participants|duration)\b", re.I)


@dataclass
class Turn:
    speaker: str
    title: str
    firm: str
    is_analyst: bool
    section: str  # "prepared_remarks" | "qa"
    text: str = ""


def _role(parts: list[str]) -> tuple[str, str, bool]:
    parts = [p.strip() for p in parts if p.strip()]
    is_analyst = any(p.lower() == "analyst" or "analyst" in p.lower() for p in parts)
    if is_analyst:
        firm = next((p for p in parts if "analyst" not in p.lower()), "")
        return "Analyst", firm, True
    return " -- ".join(parts), "", False


def parse_turns(text: str) -> list[Turn]:
    turns: list[Turn] = []
    section = "prepared_remarks"
    in_participants = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if PARTICIPANTS_RE.match(line):
            in_participants = True
            continue
        if PREPARED_RE.match(line):
            section, in_participants = "prepared_remarks", False
            continue
        if QA_START_RE.match(line):
            section, in_participants = "qa", False
            continue
        if in_participants:
            continue

        speaker = None
        if line == "Operator" or line.startswith("Operator:"):
            speaker = ("Operator", "", "", False)
            rest = line.partition(":")[2].strip()
        elif m := DASHED_RE.match(line):
            speaker = (m.group(1), *_role(re.split(r"\s+(?:--|—|–)\s+", m.group(2))))
            rest = ""
        elif (m := COLON_RE.match(line)) and len(m.group(1).split()) >= 2:
            speaker = (m.group(1), *_role(re.split(r"\s*[,–—-]\s*", m.group(2) or "")))
            rest = m.group(3)
        if speaker:
            turns.append(Turn(*speaker, section=section, text=rest))
        elif turns:
            turns[-1].text += ("\n\n" if turns[-1].text else "") + line

    # No explicit Q&A header: everything from the first analyst turn on is Q&A.
    if turns and all(t.section == "prepared_remarks" for t in turns):
        first = next((i for i, t in enumerate(turns) if t.is_analyst), len(turns))
        for t in turns[max(0, first - 1):] if first < len(turns) else []:
            t.section = "qa"
    return [t for t in turns if t.text.strip()]


def _label(t: Turn) -> str:
    who = t.speaker + (f" ({t.firm})" if t.firm else "")
    return f"{who} -- {t.title}" if t.title else who


def _split(text: str, limit: int) -> list[str]:
    out, buf = [], ""
    for p in text.split("\n\n"):
        if buf and len(buf) + len(p) > limit:
            out.append(buf)
            buf = ""
        buf = f"{buf}\n\n{p}" if buf else p
    return out + ([buf] if buf else [])


def chunk_transcript(text: str, ticker: str, period: str, source: str = "") -> list[dict]:
    turns = parse_turns(text)
    header = f"[{ticker} Earnings Call {period}"
    base = {"ticker": ticker.upper(), "doc_type": "Transcript", "fiscal_period": period,
            "source_url": source, "table_name": "", "units": "", "chunk_type": "text"}
    chunks: list[dict] = []

    for t in turns:
        if t.section != "prepared_remarks" or t.speaker == "Operator":
            continue
        for body in _split(t.text, CHUNK_CHARS):
            chunks.append({
                "text": f"{header} | Prepared Remarks]\n{_label(t)}:\n{body}",
                "metadata": {**base, "section_id": "Prepared Remarks", "section_title": "Prepared Remarks",
                             "speaker": t.speaker, "speaker_title": t.title, "firm": ""},
            })

    # Group Q&A into exchanges keyed on the asking analyst.
    exchange: list[Turn] = []

    def flush():
        if not exchange:
            return
        analyst = exchange[0] if exchange[0].is_analyst else None
        speakers = list(dict.fromkeys(t.speaker for t in exchange))
        body = "\n\n".join(f"{_label(t)}:\n{t.text}" for t in exchange)
        pieces = _split(body, CHUNK_CHARS)
        q = f"{_label(analyst)}:\n{analyst.text}" if analyst else ""
        for k, piece in enumerate(pieces):
            if k and q:  # keep the question attached to every continuation piece
                piece = f"(continuing answer to) {q[:600]}\n\n{piece}"
            chunks.append({
                "text": f"{header} | Q&A]\n{piece}",
                "metadata": {**base, "section_id": "Q&A", "section_title": "Questions and Answers",
                             "speaker": ", ".join(speakers), "speaker_title": "",
                             "firm": analyst.firm if analyst else ""},
            })
        exchange.clear()

    for t in turns:
        if t.section != "qa":
            continue
        if t.speaker == "Operator":
            flush()
            continue
        if t.is_analyst and exchange and any(not x.is_analyst for x in exchange):
            flush()
        exchange.append(t)
    flush()

    slug = re.sub(r"\W+", "", period)
    for n, c in enumerate(chunks):
        c["id"] = f"{ticker.upper()}|Transcript|{slug}|{n}"
    return chunks


def ingest_transcript(path: str | pathlib.Path, ticker: str, period: str, retriever=None) -> list[dict]:
    path = pathlib.Path(path)
    chunks = chunk_transcript(path.read_text(errors="ignore"), ticker, period, source=path.name)
    if retriever is not None:
        retriever.add(chunks)
    return chunks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--period", required=True, help='e.g. "Q3 FY2025"')
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    r = None
    if not a.dry_run:
        from hybrid_retriever import HybridRetriever
        r = HybridRetriever()
    cs = ingest_transcript(a.path, a.ticker, a.period, r)
    by = {s: sum(c["metadata"]["section_id"] == s for c in cs) for s in ("Prepared Remarks", "Q&A")}
    print(f"{a.ticker} {a.period}: {len(cs)} chunks {by}", file=sys.stderr)
