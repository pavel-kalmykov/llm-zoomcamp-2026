import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    from pathlib import Path

    import numpy as np
    import openai
    import pandas as pd
    from dotenv import load_dotenv
    from gitsource import GithubRepositoryDataReader, chunk_documents
    from minsearch import Index, VectorSearch

    from evaluation_utils import Questions, evaluate, llm_structured

    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")

    import sys

    sys.path.insert(0, str(root / "02-vector-search"))
    from embedder import Embedder

    DEEPSEEK_MODEL = "deepseek-chat"
    client = openai.OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1"
    )
    emb = Embedder(path=str(root / "02-vector-search/models/Xenova/all-MiniLM-L6-v2"))

    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda p: "/lessons/" in p,
    )
    documents = [f.parse() for f in reader.read()]
    return (
        DEEPSEEK_MODEL,
        Index,
        VectorSearch,
        chunk_documents,
        client,
        documents,
        emb,
        evaluate,
        llm_structured,
        mo,
        np,
        pd,
    )


@app.cell
def _(mo):
    mo.md("""
    # Module 4 homework: Evaluation

    Ground-truth generation with structured output (via DeepSeek, since the
    Z.ai balance ran out mid-course), then Hit Rate / MRR over text, vector,
    and hybrid search from Module 2.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Q1. Generating questions
    """)
    return


@app.cell
def _(DEEPSEEK_MODEL, client, documents, llm_structured, mo):
    data_gen_instructions = """
    You emulate a student who is taking our LLM course.
    You are given one lesson page from the course.
    Formulate 5 questions this student might ask that are answered by this page.

    Rules:
    - The page should contain the answer to each question.
    - Make the questions complete and not too short.
    - Use as few words as possible from the page; don't copy its phrasing.
    - The questions should resemble how people actually ask things online:
      not too formal, not too short, not too long.
    - Ask about the content of the lesson, not about its formatting or filename.
    """.strip()

    target_pages = [
        "01-agentic-rag/lessons/01-intro.md",
        "01-agentic-rag/lessons/02-environment.md",
        "01-agentic-rag/lessons/03-rag.md",
    ]
    input_tokens = []
    for _fn in target_pages:
        _doc = next(d for d in documents if d["filename"] == _fn)
        _prompt = f'{{"filename": {_doc["filename"]!r}, "content": {_doc["content"]!r}}}'
        _qs, _usage = llm_structured(
            client, data_gen_instructions, _prompt, model=DEEPSEEK_MODEL
        )
        input_tokens.append(_usage.prompt_tokens)

    avg_input_tokens = sum(input_tokens) / len(input_tokens)
    mo.md(
        f"Average input tokens across 3 calls: **{avg_input_tokens:.0f}** "
        f"(per page: {input_tokens})"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Ground truth + search setup
    """)
    return


@app.cell
def _(Index, VectorSearch, chunk_documents, documents, emb, mo, np, pd):
    import io

    import requests

    gt_url = (
        "https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/"
        "cohorts/2026/04-evaluation/ground-truth.csv"
    )
    gt_csv = requests.get(gt_url, timeout=30).text
    gt_df = pd.read_csv(io.StringIO(gt_csv))
    ground_truth = gt_df.to_dict(orient="records")

    chunks = chunk_documents(documents, size=2000, step=1000)

    text_index = Index(text_fields=["content"], keyword_fields=["filename"])
    text_index.fit(chunks)

    X = np.array(emb.encode_batch([c["content"] for c in chunks]))
    vector_index = VectorSearch(keyword_fields=["filename"])
    vector_index.fit(X, chunks)

    def text_search(query, num_results=5):
        return text_index.search(query, num_results=num_results)

    def vector_search(query, num_results=5):
        return vector_index.search(emb.encode(query), num_results=num_results)

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

    def hybrid_search(query, k=60, num_results=5):
        text_results = text_search(query, num_results=10)
        vector_results = vector_search(query, num_results=10)
        return rrf([text_results, vector_results], k=k, num_results=num_results)

    mo.md(f"Ground truth: **{len(ground_truth)}** questions. Chunks: **{len(chunks)}**.")
    return ground_truth, hybrid_search, text_search, vector_search


@app.cell
def _(mo):
    mo.md("""
    ## Q2. First result with text search
    """)
    return


@app.cell
def _(ground_truth, mo, text_search):
    q = ground_truth[0]["question"]
    text_top = text_search(q)[0]["filename"]
    mo.md(f"Query: {q!r}\n\nFirst text_search result: `{text_top}`")
    return (q,)


@app.cell
def _(mo):
    mo.md("""
    ## Q3. First result with vector search
    """)
    return


@app.cell
def _(mo, q, vector_search):
    vector_top = vector_search(q)[0]["filename"]
    mo.md(f"First vector_search result: `{vector_top}`")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Q4. Evaluating text search
    """)
    return


@app.cell
def _(evaluate, ground_truth, mo, text_search):
    res_text = evaluate(ground_truth, text_search)
    mo.md(f"text_search: {res_text}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Q5. Evaluating vector search
    """)
    return


@app.cell
def _(evaluate, ground_truth, mo, vector_search):
    res_vector = evaluate(ground_truth, vector_search)
    mo.md(f"vector_search: {res_vector}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Q6. Tuning hybrid search
    """)
    return


@app.cell
def _(evaluate, ground_truth, hybrid_search, mo):
    hybrid_results = {}
    for _k in [1, 50, 100, 200]:
        hybrid_results[_k] = evaluate(
            ground_truth, lambda qq, k=_k: hybrid_search(qq, k=k)
        )
    best_k = max(hybrid_results, key=lambda k: hybrid_results[k]["mrr"])
    mo.md(f"Results per k: {hybrid_results}\n\nBest k: **{best_k}**")
    return


if __name__ == "__main__":
    app.run()
