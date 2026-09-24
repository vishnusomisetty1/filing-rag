"""Hybrid retrieval: BM25 + ChromaDB vectors, fused with Reciprocal Rank Fusion, then reranked.

Embeddings: the Spark's vLLM serves only the chat model (no /v1/embeddings), so by default
this uses Chroma's built-in ONNX MiniLM, which runs in-process on CPU. Set EMBED_BASE_URL +
EMBED_MODEL to point at any OpenAI-compatible embeddings endpoint (e.g. the planned LiteLLM
`embed` alias). The collection name includes the embedding model, so switching models
builds a fresh index instead of silently mixing vector dimensions.

Reranking (RERANKER): "llm" (default) asks DeepSeek to order the candidates; "cross-encoder"
uses sentence-transformers locally (pip install sentence-transformers); "none" keeps RRF order.
"""
import os
import re
import json

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

import llm

CHROMA_PATH = os.environ.get("CHROMA_PATH", "data/chroma")
RERANKER = os.environ.get("RERANKER", "llm")
RRF_K = 60
# Primary statements answer most numeric questions but are long, number-heavy tables that
# rank poorly against narrative MD&A text, so scoped searches always offer them to the reranker.
CORE_STATEMENT_RE = re.compile(
    r"^(consolidated )?(statements? of (income|operations|earnings|cash flows|comprehensive income)|balance sheets?"
    r"|(comprehensive )?income statements?|cash flows? statements?)$", re.I)
MAX_CORE = 8
_STOP = set("the a an of and or to in on for is are was were be by with as at from that this it its".split())


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9.%$\-]*", text.lower()) if t not in _STOP]


def _embedding_function():
    base = os.environ.get("EMBED_BASE_URL")
    if base:
        name = os.environ.get("EMBED_MODEL", "embed")
        return name, embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("EMBED_API_KEY", "not-needed"), api_base=base, model_name=name)
    return "all-MiniLM-L6-v2", embedding_functions.DefaultEmbeddingFunction()


def build_where(tickers: list[str] | None = None, doc_types: list[str] | None = None) -> dict | None:
    clauses = []
    if tickers:
        clauses.append({"ticker": {"$in": list(tickers)}})
    if doc_types:
        clauses.append({"doc_type": {"$in": list(doc_types)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _matches(meta: dict, where: dict | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_matches(meta, w) for w in where["$and"])
    (field, cond), = where.items()
    return meta.get(field) in cond["$in"] if isinstance(cond, dict) else meta.get(field) == cond


class HybridRetriever:
    def __init__(self, path: str = CHROMA_PATH):
        embed_name, ef = _embedding_function()
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(
            f"filings__{re.sub(r'[^A-Za-z0-9_-]', '-', embed_name)}"[:60],
            embedding_function=ef, metadata={"hnsw:space": "cosine"})
        self._build_bm25()

    # ---------------------------------------------------------------- indexing

    def add(self, chunks: list[dict], batch: int = 128) -> None:
        for i in range(0, len(chunks), batch):
            b = chunks[i:i + batch]
            self.col.upsert(ids=[c["id"] for c in b], documents=[c["text"] for c in b],
                            metadatas=[{k: (v if v is not None else "") for k, v in c["metadata"].items()} for c in b])
        self._build_bm25()

    def _build_bm25(self) -> None:
        data = self.col.get(include=["documents", "metadatas"])
        self.ids, self.docs, self.metas = data["ids"], data["documents"], data["metadatas"]
        self.bm25 = BM25Okapi([_tokenize(d) for d in self.docs]) if self.docs else None

    def tickers(self) -> list[str]:
        return sorted({m["ticker"] for m in self.metas})

    def inventory(self) -> list[tuple[str, str, str]]:
        return sorted({(m["ticker"], m["doc_type"], m["fiscal_period"]) for m in self.metas})

    # ---------------------------------------------------------------- retrieval

    def search(self, query: str, k: int = 8, where: dict | None = None, candidates: int = 30,
               rerank_query: str | None = None) -> list[dict]:
        if not self.docs:
            return []
        n = min(candidates, len(self.docs))
        vec = self.col.query(query_texts=[query], n_results=n, where=where)["ids"][0]

        lex: list[str] = []
        if self.bm25:
            scores = self.bm25.get_scores(_tokenize(query))
            order = sorted(range(len(self.ids)), key=lambda i: -scores[i])
            lex = [self.ids[i] for i in order if scores[i] > 0 and _matches(self.metas[i], where)][:n]

        fused: dict[str, float] = {}
        for ranking in (vec, lex):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(fused, key=lambda c: -fused[c])[:max(k * 3, 24)]
        if where:
            core = [cid for cid, m in zip(self.ids, self.metas)
                    if _matches(m, where) and CORE_STATEMENT_RE.match(m.get("table_name", ""))]
            if len(core) <= MAX_CORE:
                top += [c for c in core if c not in top]

        pos = {cid: i for i, cid in enumerate(self.ids)}
        hits = [{"id": c, "text": self.docs[pos[c]], "metadata": self.metas[pos[c]], "rrf": fused.get(c, 0.0)}
                for c in top if c in pos]
        return self.rerank(rerank_query or query, hits)[:k]

    def rerank(self, query: str, hits: list[dict]) -> list[dict]:
        if len(hits) <= 1 or RERANKER == "none":
            return hits
        if RERANKER == "cross-encoder":
            from sentence_transformers import CrossEncoder
            if not hasattr(self, "_ce"):
                self._ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            scores = self._ce.predict([(query, h["text"]) for h in hits])
            return [h for _, h in sorted(zip(scores, hits), key=lambda x: -x[0])]
        return self._llm_rerank(query, hits)

    def _llm_rerank(self, query: str, hits: list[dict]) -> list[dict]:
        passages = "\n\n".join(f"[{i}] {h['text'][:1200]}" for i, h in enumerate(hits))
        prompt = (
            "You rank passages from SEC filings and earnings call transcripts by how useful they are "
            "for answering a financial question. Prefer passages containing the exact figures, tables, "
            "or executive/analyst statements needed.\n\n"
            f"Question: {query}\n\nPassages:\n{passages}\n\n"
            "Return ONLY a JSON array of passage numbers, most useful first. Omit irrelevant passages."
        )
        try:
            out = llm.chat([{"role": "user", "content": prompt}], max_tokens=200)
            order = [i for i in json.loads(re.search(r"\[[\d,\s]*\]", out).group()) if 0 <= i < len(hits)]
        except Exception:
            return hits  # reranker failure should never break retrieval
        seen = list(dict.fromkeys(order))
        return [hits[i] for i in seen] + [h for i, h in enumerate(hits) if i not in seen]
