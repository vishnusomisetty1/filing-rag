"""Download a company's latest 10-K from EDGAR and split it into Item sections and tables.

Usage: SEC_USER_AGENT="Your Name you@example.com" python sec_ingest.py AAPL [MSFT ...]
"""
import os
import re
import sys
import json
import pathlib

import requests
from bs4 import BeautifulSoup

RAW_DIR = pathlib.Path("data/raw")
TEXT_CHUNK_CHARS = 1800
TABLE_CHUNK_CHARS = 3500

ITEM_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity", "6": "Reserved",
    "7": "Management's Discussion and Analysis", "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data", "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information", "9C": "Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services", "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
# "ITEM 1.BUSINESS" (no space) is common, so only require that the number ends there.
ITEM_RE = re.compile(r"^\s*item\s*(1a|1b|1c|7a|9a|9b|9c|1[0-6]|[1-9])(?![0-9a-z])", re.I)
# Page running headers like "Item 1" or "PART II Item 7" repeated on every page.
RUNNING_HEADER_RE = re.compile(r"^(part\s+[ivx]+\s*)?item\s*\w{1,2}\.?$", re.I)
UNITS_RE = re.compile(r"in (thousands|millions|billions)", re.I)
FOOTER_RE = re.compile(r"(form 10-k|^page \d+|\|\s*\d+$|^\d+$|^\(?in (thousands|millions|billions))", re.I)
TABLE_MARK = "\x00TABLE{}\x00"


# ---------------------------------------------------------------- EDGAR download

def _get(url: str) -> requests.Response:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError('SEC requires a contact User-Agent: export SEC_USER_AGENT="Your Name you@example.com"')
    r = requests.get(url, headers={"User-Agent": ua}, timeout=60)
    r.raise_for_status()
    return r


def cik_for(ticker: str) -> int:
    for row in _get("https://www.sec.gov/files/company_tickers.json").json().values():
        if row["ticker"].upper() == ticker.upper():
            return int(row["cik_str"])
    raise ValueError(f"Unknown ticker {ticker}")


def download_latest_10k(ticker: str) -> tuple[str, dict]:
    """Return (html, filing_meta) for the most recent 10-K, caching the HTML under data/raw."""
    cik = cik_for(ticker)
    recent = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]
    i = recent["form"].index("10-K")
    acc = recent["accessionNumber"][i].replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{recent['primaryDocument'][i]}"
    meta = {
        "ticker": ticker.upper(),
        "fiscal_period": f"FY{recent['reportDate'][i][:4]}",
        "source_url": url,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{meta['ticker']}_10-K_{meta['fiscal_period']}.html"
    if not path.exists():
        path.write_text(_get(url).text)
        path.with_suffix(".json").write_text(json.dumps(meta))
    return path.read_text(), meta


# ---------------------------------------------------------------- HTML -> text + markdown tables

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def table_to_rows(table) -> list[list[str]]:
    """Flatten an EDGAR table, merging the split cells EDGAR uses for '$', ')' and '%'."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", _clean(td.get_text(" "))))
                 for td in tr.find_all(["td", "th"])]
        merged: list[str] = []
        for c in cells:
            if not c:
                continue
            if merged and c in (")", "%", ")%"):
                merged[-1] += c
            elif merged and merged[-1] in ("$", "(", "$("):
                merged[-1] += c
            else:
                merged.append(c)
        if merged:
            rows.append(merged)
    return rows


def rows_to_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    # Short rows are label-only rows (e.g. "Operating expenses:") or column-header rows
    # (dates/years) missing the label column; pad labels on the right, headers on the left.
    padded = []
    for r in rows:
        pad = [""] * (width - len(r))
        is_header = len(r) > 1 and all(re.search(r"\d", c) for c in r) and not re.search(r"\d,\d{3}", r[0])
        padded.append(pad + r if is_header else r + pad)
    esc = [[c.replace("|", "\\|") for c in r] for r in padded]
    lines = ["| " + " | ".join(esc[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in esc[1:]]
    return "\n".join(lines)


def html_to_lines(html: str) -> tuple[list[str], dict[int, list[list[str]]]]:
    soup = BeautifulSoup(html, "lxml")
    for el in soup.find_all(["script", "style", "ix:header"]):
        el.decompose()
    for el in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        el.decompose()
    tables: dict[int, list[list[str]]] = {}
    for n, t in enumerate(soup.find_all("table")):
        rows = table_to_rows(t)
        numeric = sum(bool(re.search(r"\d", c)) for r in rows for c in r[1:])
        if len(rows) >= 2 and numeric >= 2:
            tables[n] = rows
            t.replace_with(soup.new_string("\n" + TABLE_MARK.format(n) + "\n"))
        else:  # layout table: keep its text inline
            t.replace_with(soup.new_string("\n" + "\n".join(" ".join(r) for r in rows) + "\n"))
    for br in soup.find_all(["br"]):
        br.replace_with("\n")
    for blk in soup.find_all(["p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
        blk.insert_after("\n")
    text = soup.get_text()
    lines = [_clean(l) for l in text.split("\n")]
    return [l for l in lines if l], tables


def split_items(lines: list[str]) -> dict[str, list[str]]:
    """Split into Items. The table of contents repeats every header, so for each Item keep
    the longest span, which is the real body rather than the TOC entry."""
    heads = [(i, ITEM_RE.match(l).group(1).upper()) for i, l in enumerate(lines)
             if len(l) < 250 and ITEM_RE.match(l)]
    # Back-to-back headers for the same Item are page running headers, not new sections.
    heads = [h for k, h in enumerate(heads) if k == 0 or h[1] != heads[k - 1][1]]
    best: dict[str, list[str]] = {}
    for k, (start, item) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        span = lines[start:end]
        if len("".join(span)) > len("".join(best.get(item, []))):
            best[item] = span
    return best


# ---------------------------------------------------------------- chunking

def _pack(paragraphs: list[str], limit: int) -> list[str]:
    out, buf = [], ""
    for p in paragraphs:
        if buf and len(buf) + len(p) > limit:
            out.append(buf)
            buf = ""
        buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        out.append(buf)
    return out


def _table_name_and_units(prev_lines: list[str]) -> tuple[str, str]:
    """Nearest caption above the table, and its units note verbatim (they often carry
    exceptions like 'except shares, which are in thousands')."""
    units = ""
    name = ""
    for l in reversed(prev_lines[-5:]):
        if UNITS_RE.search(l) and not units and len(l) < 200:
            units = l.strip("()")
        elif not name and len(l) < 160 and "\x00" not in l and not FOOTER_RE.search(l):
            name = l
    return name, units


def chunk_10k(html: str, meta: dict) -> list[dict]:
    lines, tables = html_to_lines(html)
    chunks: list[dict] = []
    for item, span in split_items(lines).items():
        title = f"Item {item}. {ITEM_TITLES.get(item, '')}".strip()
        base = {**meta, "doc_type": "10-K", "section_id": f"Item {item}", "section_title": title,
                "speaker": "", "table_name": "", "units": "", "chunk_type": "text"}
        header = f"[{meta['ticker']} 10-K {meta['fiscal_period']} | {title}]"
        paras: list[str] = []
        units_ctx = ""

        def flush():
            for body in _pack(paras, TEXT_CHUNK_CHARS):
                chunks.append({"text": f"{header}\n{body}", "metadata": dict(base)})
            paras.clear()

        for j, line in enumerate(span):
            m = re.fullmatch(r"\x00TABLE(\d+)\x00", line)
            if not m:
                if j and RUNNING_HEADER_RE.match(line):
                    continue
                paras.append(line)
                if UNITS_RE.search(line):
                    units_ctx = "in " + UNITS_RE.search(line).group(1).lower()
                continue
            flush()
            rows = tables[int(m.group(1))]
            name, units = _table_name_and_units(span[max(0, j - 5):j])
            units = units or units_ctx
            head_md = rows_to_markdown(rows[:1])
            # Sub-chunk long tables by rows, re-prepending the column header to each piece.
            body_rows = rows[1:]
            groups, cur = [], []
            for r in body_rows:
                cur.append(r)
                if len(str(cur)) > TABLE_CHUNK_CHARS:
                    groups.append(cur)
                    cur = []
            if cur or not groups:
                groups.append(cur)
            for g in groups:
                md = rows_to_markdown(rows[:1] + g) if g else head_md
                label = f"Table: {name}" + (f" ({units})" if units else "")
                # Row labels up front: tables are mostly numbers, and the embedding model only
                # reads the first ~256 tokens, so without this "Net income" is invisible to search.
                items = list(dict.fromkeys(r[0].rstrip(":") for r in g if re.search(r"[A-Za-z]", r[0])))
                line_items = "Line items: " + "; ".join(items[:60]) if items else ""
                chunks.append({
                    "text": f"{header}\n{label}\n{line_items}\n\n{md}",
                    "metadata": {**base, "table_name": name, "units": units, "chunk_type": "table"},
                })
        flush()
    for n, c in enumerate(chunks):
        c["id"] = f"{meta['ticker']}|10-K|{meta['fiscal_period']}|{n}"
    return chunks


def ingest_10k(ticker: str, retriever=None) -> list[dict]:
    html, meta = download_latest_10k(ticker)
    chunks = chunk_10k(html, meta)
    if retriever is not None:
        retriever.add(chunks)
    return chunks


if __name__ == "__main__":
    from hybrid_retriever import HybridRetriever

    r = HybridRetriever()
    for t in sys.argv[1:]:
        cs = ingest_10k(t, r)
        items = sorted({c["metadata"]["section_id"] for c in cs})
        print(f"{t}: {len(cs)} chunks, {sum(c['metadata']['chunk_type'] == 'table' for c in cs)} tables, items={items}",
              flush=True)
    os._exit(0)  # skip interpreter teardown: onnxruntime threads abort on macOS at exit
