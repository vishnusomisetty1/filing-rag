"""Streamlit chat frontend for Filing-RAG.  Run: streamlit run app.py

PUBLIC_MODE=1 turns this into a shareable demo: passcode gate, no ingestion, and
server-side caps so visitors can't monopolise the Spark (see Usage below).
"""
import os
import hmac
import time
import pathlib
import datetime
import tempfile
import threading

import streamlit as st

import llm
from hybrid_retriever import HybridRetriever
from query_engine import QueryEngine, citation_label, md_safe, render_citations

PUBLIC = os.environ.get("PUBLIC_MODE") == "1"
PASSCODE = os.environ.get("DEMO_PASSCODE", "")
MAX_PER_DAY = int(os.environ.get("DEMO_MAX_QUESTIONS_PER_DAY", "300"))
MAX_PER_SESSION = int(os.environ.get("DEMO_MAX_QUESTIONS_PER_SESSION", "20"))
MAX_CONCURRENT = int(os.environ.get("DEMO_MAX_CONCURRENT", "2"))
MAX_QUESTION_CHARS = 500

st.set_page_config(page_title="Filing-RAG", page_icon="📑", layout="wide")


class Usage:
    """Process-wide counters shared by every visitor, so reloading the page resets nothing."""

    def __init__(self):
        self.lock = threading.Lock()
        self.day = datetime.date.today()
        self.count = 0
        self.slots = threading.BoundedSemaphore(MAX_CONCURRENT)

    def try_start(self) -> str | None:
        """Reserve a question; returns a refusal message or None if allowed."""
        with self.lock:
            if datetime.date.today() != self.day:
                self.day, self.count = datetime.date.today(), 0
            if self.count >= MAX_PER_DAY:
                return "The demo has hit its daily question limit. Please try again tomorrow."
            if not self.slots.acquire(blocking=False):
                return "The demo is busy answering other visitors. Please try again in a few seconds."
            self.count += 1
            return None

    def finish(self) -> None:
        self.slots.release()


@st.cache_resource
def get_engine() -> QueryEngine:
    return QueryEngine(HybridRetriever(), max_answer_tokens=1500 if PUBLIC else 4096)


@st.cache_resource
def get_usage() -> Usage:
    return Usage()


# ---------------------------------------------------------------- public-mode gate
if PUBLIC:
    if len(PASSCODE) < 8:
        st.error("PUBLIC_MODE requires DEMO_PASSCODE (8+ characters). Refusing to start.")
        st.stop()
    if not st.session_state.get("authed"):
        st.title("📑 Filing-RAG")
        st.write("Chat with SEC 10-K filings and earnings calls, with every figure cited to its source. "
                 "Runs on a self-hosted DeepSeek V4 Flash across two NVIDIA DGX Sparks.")
        attempts = st.session_state.get("attempts", 0)
        if attempts >= 5:
            st.error("Too many attempts. Reload the page to try again.")
            st.stop()
        code = st.text_input("Demo passcode", type="password")
        if code:
            if hmac.compare_digest(code.encode(), PASSCODE.encode()):
                st.session_state.authed = True
                st.rerun()
            st.session_state.attempts = attempts + 1
            time.sleep(1)
            st.error("Wrong passcode.")
        st.stop()

engine = get_engine()
retriever = engine.retriever
usage = get_usage()

# ---------------------------------------------------------------- sidebar: ingest + filters
with st.sidebar:
    st.header("Filing-RAG")
    st.caption(f"Model: `{llm.LLM_MODEL}`")

    if not PUBLIC:
        from sec_ingest import ingest_10k
        from transcript_parser import ingest_transcript

        with st.expander("Add a 10-K from EDGAR"):
            t = st.text_input("Ticker", key="tenk_ticker").strip().upper()
            if st.button("Fetch latest 10-K", disabled=not t):
                with st.spinner(f"Downloading and indexing {t} 10-K…"):
                    try:
                        n = len(ingest_10k(t, retriever))
                        st.success(f"Indexed {n} chunks")
                    except Exception as e:
                        st.error(str(e))

        with st.expander("Add an earnings call transcript"):
            up = st.file_uploader("Transcript (.txt)", type=["txt"])
            tt = st.text_input("Ticker", key="tx_ticker").strip().upper()
            period = st.text_input("Period", placeholder="Q3 FY2025")
            if st.button("Index transcript", disabled=not (up and tt and period)):
                with tempfile.TemporaryDirectory() as d:
                    p = pathlib.Path(d) / up.name
                    p.write_bytes(up.getvalue())
                    n = len(ingest_transcript(p, tt, period, retriever))
                st.success(f"Indexed {n} chunks")

    st.subheader("Scope")
    tickers = st.multiselect("Tickers", retriever.tickers())
    doc_types = st.multiselect("Document types", ["10-K", "Transcript"])
    with st.expander("Indexed documents"):
        for tk, dt, fp in retriever.inventory():
            st.write(f"{tk} · {dt} · {fp}")
    if st.button("Clear conversation"):
        st.session_state.messages = []


# ---------------------------------------------------------------- chat
def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            with st.expander(f"S{i} · {citation_label(s['metadata'])}"):
                st.markdown(md_safe(s["text"]))


if "messages" not in st.session_state:
    st.session_state.messages = []

if not retriever.docs:
    st.info("No documents indexed yet. Add a 10-K or a transcript from the sidebar.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(md_safe(m["content"]))
        show_sources(m.get("sources", []))

asked = st.session_state.get("asked", 0)
limit_hit = PUBLIC and asked >= MAX_PER_SESSION
if limit_hit:
    st.info("You've reached this demo's per-visitor question limit. Thanks for trying it!")

question = st.chat_input("Ask about a 10-K or an earnings call…", disabled=limit_hit,
                         max_chars=MAX_QUESTION_CHARS if PUBLIC else None)
if question:
    refusal = usage.try_start() if PUBLIC else None
    if refusal:
        st.warning(refusal)
        st.stop()
    st.session_state.asked = asked + 1
    try:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(md_safe(question))

        with st.chat_message("assistant"):
            with st.spinner("Retrieving…"):
                ans = engine.ask(question, history=history, tickers=tickers, doc_types=doc_types)
            if ans.query != question:
                st.caption(f"Searched for: {ans.query}")
            box, text = st.empty(), ""
            for tok in ans.stream:
                text += tok
                box.markdown(md_safe(text) + "▌")
            text = render_citations(text.removeprefix("CLARIFY:").strip(), ans.sources)
            sources = ans.sources if "`[" in text else []
            box.markdown(md_safe(text))
            show_sources(sources)
        st.session_state.messages.append({"role": "assistant", "content": text, "sources": sources})
    finally:
        if PUBLIC:
            usage.finish()
