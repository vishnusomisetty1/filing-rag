# Filing-RAG

Chat over SEC 10-K filings and earnings call transcripts, answered by the local
DeepSeek V4 Flash on the Spark. No API key.

```
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
export SEC_USER_AGENT="Your Name you@example.com"      # SEC requires a contact
export LLM_BASE_URL=http://localhost:8000/v1       # any OpenAI-compatible endpoint
.venv/bin/python sec_ingest.py AAPL MSFT                # latest 10-K per ticker
.venv/bin/python transcript_parser.py call.txt --ticker AAPL --period "Q4 FY2025"
.venv/bin/streamlit run app.py
```

Both kinds of documents can also be added from the app's sidebar.

| Module | Role |
|---|---|
| `llm.py` | OpenAI-compatible client for `LLM_BASE_URL` / `LLM_MODEL` |
| `sec_ingest.py` | EDGAR download, Item splitter, tables → Markdown with caption + units |
| `transcript_parser.py` | Speaker turns, Prepared Remarks vs Q&A, analyst exchange grouping |
| `hybrid_retriever.py` | BM25 + Chroma vectors → RRF → rerank |
| `query_engine.py` | Query rewrite, grounded answering, clarification, citations |
| `app.py` | Streamlit chat UI |

Per query: the model rewrites the question into filing vocabulary ("operating margin"
→ operating income, total net sales, statements of operations), hybrid retrieval pulls
candidates, the model reranks them, then answers with `[S#]` tags that the app expands
to `[Ticker | Doc | Period | Section | Speaker/Table]`.

**Embeddings.** The Spark's vLLM serves only the chat model, so embeddings default to
Chroma's in-process ONNX MiniLM. Set `EMBED_BASE_URL`/`EMBED_MODEL` to use an
OpenAI-compatible embeddings endpoint instead. The Chroma collection is named after
the embedding model, so switching models builds a new index rather than silently
mixing dimensions. Re-run ingestion after switching.

Transcripts are read from local `.txt` files; there is no free transcript API.

## Public demo mode

`./run_demo.sh` serves a shareable, passcode-gated demo (settings in a git-ignored
`.env.demo`: `DEMO_PASSCODE`, plus optional caps). In public mode:

- ingestion (EDGAR fetch, transcript upload) is removed from the UI
- a shared daily question cap and a limit of 2 in-flight model calls apply across all
  visitors, plus a per-visitor cap. They're enforced server-side, so reloading doesn't reset them.
- questions are capped at 500 characters and answers at 1,500 tokens

The app binds to `127.0.0.1` and is published with Tailscale Funnel, which exposes only
this one port. The model server itself stays private on the tailnet.

```
tailscale funnel --bg --https=8443 8502     # publish
tailscale funnel --https=8443 off           # unpublish
```
