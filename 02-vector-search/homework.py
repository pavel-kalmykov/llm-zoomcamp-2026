import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    import numpy as np
    from dotenv import load_dotenv
    from embedder import Embedder
    from gitsource import GithubRepositoryDataReader, chunk_documents
    from minsearch import Index, VectorSearch

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    emb = Embedder()
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda p: "/lessons/" in p,
    )
    documents = [f.parse() for f in reader.read()]
    return Index, VectorSearch, chunk_documents, documents, emb, mo, np


@app.cell
def _(mo):
    mo.md("""
    # Module 2 homework: Vector Search

    Embeddings with a lightweight ONNX model (`all-MiniLM-L6-v2`, 384 dims),
    cosine similarity, vector search, text vs vector, and hybrid search (RRF).
    """)
    return


@app.cell
def _(emb, mo):
    q1 = "How does approximate nearest neighbor search work?"
    v = emb.encode(q1)
    mo.md(f"**Q1.** First value of the 384-dim vector: **{float(v[0]):.4f}**")
    return (v,)


@app.cell
def _(documents, emb, mo, np, v):
    page = next(
        d for d in documents
        if d["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md"
    )
    cos = float(np.dot(v, emb.encode(page["content"])))
    mo.md(f"**Q2.** Cosine similarity with that page: **{cos:.4f}**")
    return


@app.cell
def _(VectorSearch, chunk_documents, documents, emb, mo, np, v):
    chunks = chunk_documents(documents, size=2000, step=1000)
    X = np.array(emb.encode_batch([c["content"] for c in chunks]))
    scores = X.dot(v)
    best = int(np.argmax(scores))
    vs = VectorSearch(keyword_fields=["filename"])
    vs.fit(X, chunks)
    mo.md(
        f"**Q3.** Chunks: {len(chunks)}. Highest-scoring chunk belongs to "
        f"`{chunks[best]['filename']}` (score {float(scores[best]):.3f})."
    )
    return chunks, vs


@app.cell
def _(emb, mo, vs):
    q4 = "What metric do we use to evaluate a search engine?"
    res4 = vs.search(emb.encode(q4), num_results=5)
    mo.md(f"**Q4.** First result: `{res4[0]['filename']}`")
    return


@app.cell
def _(Index, chunks, mo):
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(chunks)
    mo.md("Indexed chunks with minsearch `Index` (content as text field).")
    return (index,)


@app.cell
def _(emb, index, mo, vs):
    q5 = "How do I store vectors in PostgreSQL?"
    vec5 = [d["filename"] for d in vs.search(emb.encode(q5), num_results=5)]
    txt5 = [d["filename"] for d in index.search(q5, num_results=5)]
    only_vector = sorted(set(vec5) - set(txt5))
    mo.md(f"**Q5.** In vector top-5 but not in text top-5: `{only_vector}`")
    return


@app.cell
def _(emb, index, mo, vs):
    def rrf(result_lists, k=60, num_results=5):
        scores = {}
        docs = {}
        for results in result_lists:
            for rank, doc in enumerate(results):
                key = (doc["filename"], doc["start"])
                scores[key] = scores.get(key, 0) + 1 / (k + rank)
                docs[key] = doc
        ranked = sorted(scores, key=scores.get, reverse=True)
        return [docs[key] for key in ranked[:num_results]]

    q6 = "How do I give the model access to tools?"
    vec6 = vs.search(emb.encode(q6), num_results=5)
    txt6 = index.search(q6, num_results=5)
    fused = rrf([vec6, txt6])
    mo.md(f"**Q6.** First after RRF: `{fused[0]['filename']}`")
    return


if __name__ == "__main__":
    app.run()
