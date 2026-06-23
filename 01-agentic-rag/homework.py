import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    from pathlib import Path

    import anthropic
    from dotenv import load_dotenv
    from gitsource import GithubRepositoryDataReader, chunk_documents
    from minsearch import Index

    from rag_helper import RAG

    # Load .env from the repo root (one level up from this module's folder).
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    QUERY = "How does the agentic loop keep calling the model until it stops?"
    MODEL = os.environ.get("GLM_MODEL", "glm-4.6")
    client = anthropic.Anthropic(
        api_key=os.environ["ZAI_API_KEY"],
        base_url=os.environ["ZAI_BASE_URL"],
    )
    return (
        GithubRepositoryDataReader,
        Index,
        MODEL,
        QUERY,
        RAG,
        chunk_documents,
        client,
        mo,
        os,
    )


@app.cell
def _(mo):
    mo.md("""
    # Module 1 homework: Agentic RAG

    Knowledge base: the course's own lesson pages, fetched from
    `DataTalksClub/llm-zoomcamp` at commit `8c1834d` via `gitsource`.

    LLM: `glm-4.6` through Z.ai's Anthropic-compatible endpoint.
    """)
    return


@app.cell
def _(GithubRepositoryDataReader, mo):
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    documents = [f.parse() for f in reader.read()]
    mo.md(f"**Q1.** How many lesson pages are in the dataset? **{len(documents)}**")
    return (documents,)


@app.cell
def _(mo):
    mo.md("""
    ## Q2. Indexing and searching
    """)
    return


@app.cell
def _(Index, QUERY, documents, mo):
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(documents)
    results = index.search(QUERY, num_results=5)
    top = results[0]["filename"]
    _body = "\n".join(
        [f"First result: `{top}`", "", "Top 5:", ""]
        + [f"- `{r['filename']}`" for r in results]
    )
    mo.md(_body)
    return (index,)


@app.cell
def _(mo):
    mo.md("""
    ## Q3. RAG (input tokens)
    """)
    return


@app.cell
def _(MODEL, QUERY, RAG, client, index, mo):
    rag = RAG(index=index, llm_client=client, model=MODEL)
    answer, usage = rag.rag(QUERY)
    _body = "\n".join(
        [
            f"Input (prompt) tokens: **{usage.input_tokens}**",
            "",
            "Answer (start):",
            "",
            "```text",
            answer.strip(),
            "```",
        ]
    )
    mo.md(_body)
    return (usage,)


@app.cell
def _(mo):
    mo.md("""
    ## Q4. Chunking
    """)
    return


@app.cell
def _(chunk_documents, documents, mo):
    chunks = chunk_documents(documents, size=2000, step=1000)
    mo.md(
        f"""
        Chunks with `size=2000, step=1000`: **{len(chunks)}**

        Chunk keys: `{list(chunks[0].keys())}`
        """
    )
    return (chunks,)


@app.cell
def _(mo):
    mo.md("""
    ## Q5. RAG with chunking (fewer input tokens)
    """)
    return


@app.cell
def _(Index, MODEL, QUERY, RAG, chunks, client, mo, usage):
    chunk_index = Index(text_fields=["content"], keyword_fields=["filename"])
    chunk_index.fit(chunks)
    chunk_rag = RAG(index=chunk_index, llm_client=client, model=MODEL)
    _, chunk_usage = chunk_rag.rag(QUERY)
    ratio = round(usage.input_tokens / chunk_usage.input_tokens, 1)
    mo.md(
        f"""
        Chunked input tokens: **{chunk_usage.input_tokens}**
        (Q3 was {usage.input_tokens}, so **{ratio}x fewer**)
        """
    )
    return (chunk_index,)


@app.cell
def _(mo):
    mo.md("""
    ## Q6. Turning it into an agent

    The LLM gets a `search` tool over the chunk index and decides when
    (and what) to search. Counting how many times it calls `search`.
    """)
    return


@app.cell
def _(MODEL, chunk_index, mo, os):
    from toyaikit.chat.interface import IPythonChatInterface
    from toyaikit.chat.runners import AnthropicMessagesRunner
    from toyaikit.llm import AnthropicClient
    from toyaikit.tools import Tools

    calls = {"n": 0}

    def search(query: str) -> str:
        """Search the LLM Zoomcamp lessons and return the most relevant passages."""
        calls["n"] += 1
        hits = chunk_index.search(query, num_results=5)
        return "\n\n".join(f"File: {h['filename']}\n{h['content'][:1500]}" for h in hits)

    tools = Tools()
    tools.add_tool(search)

    llm = AnthropicClient(
        model=MODEL,
        api_key=os.environ["ZAI_API_KEY"],
        base_url=os.environ["ZAI_BASE_URL"],
        extra_kwargs={"max_tokens": 4096},
    )
    runner = AnthropicMessagesRunner(
        tools=tools,
        developer_prompt=(
            "You're a course teaching assistant. Answer the student's question "
            "using the search tool. Make multiple searches with different "
            "keywords before answering."
        ),
        chat_interface=IPythonChatInterface(),
        llm_client=llm,
    )
    result = runner.loop(
        prompt="How does the agentic loop work, and how is it different from plain RAG?"
    )
    _body = "\n".join(
        [
            f"The agent called `search` **{calls['n']}** times.",
            "",
            "Final answer (start):",
            "",
            "```text",
            str(result.last_message).strip(),
            "```",
        ]
    )
    mo.md(_body)
    return


if __name__ == "__main__":
    app.run()
