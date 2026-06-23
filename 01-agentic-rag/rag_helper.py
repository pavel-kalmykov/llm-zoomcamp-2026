"""Minimal RAG helper for the LLM Zoomcamp Module 1 homework.

Adapted from the course's RAGBase: it works with the lesson-page schema
(`filename` / `content`) instead of the FAQ schema (`section` / `question` /
`answer`), talks to an Anthropic-compatible endpoint, and exposes token usage
so the homework can count input tokens.
"""

from minsearch import Index

INSTRUCTIONS = (
    "You're a course teaching assistant for the LLM Zoomcamp. "
    "Answer the student's question using only the provided lesson context."
)

PROMPT_TEMPLATE = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT
from the course lessons. Use only the facts from the CONTEXT when answering.

QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAG:
    def __init__(self, index: Index, llm_client, model: str, num_results: int = 5):
        self.index = index
        self.llm_client = llm_client
        self.model = model
        self.num_results = num_results

    def search(self, query: str) -> list[dict]:
        return self.index.search(query, num_results=self.num_results)

    def build_context(self, search_results: list[dict]) -> str:
        return "\n\n".join(
            f"File: {doc['filename']}\n{doc['content']}" for doc in search_results
        )

    def build_prompt(self, query: str, search_results: list[dict]) -> str:
        context = self.build_context(search_results)
        return PROMPT_TEMPLATE.format(question=query, context=context)

    def llm(self, prompt: str) -> tuple[str, object]:
        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=INSTRUCTIONS,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text
        return answer, response.usage

    def rag(self, query: str) -> tuple[str, object]:
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer, usage = self.llm(prompt)
        return answer, usage
