"""Evaluation helpers for the LLM Zoomcamp Module 4 homework.

Adapted from the course's evaluation_utils.py: generates ground-truth questions
with structured output via an OpenAI-compatible tool call (used against
DeepSeek), and provides the relevance / hit rate / MRR metrics from the
Search Evaluation lesson, adjusted to key on `filename` instead of `id`.
"""

import json

from pydantic import BaseModel
from tqdm.auto import tqdm


class Questions(BaseModel):
    questions: list[str]


def llm_structured(client, instructions, user_prompt, model="deepseek-chat"):
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_prompt},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "record_questions",
                    "description": "Record the generated questions",
                    "parameters": Questions.model_json_schema(),
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "record_questions"}},
    )
    call = response.choices[0].message.tool_calls[0]
    args = json.loads(call.function.arguments)
    return Questions(**args), response.usage


def compute_relevance(record, search_function):
    filename = record["filename"]
    results = search_function(record["question"])
    return [int(doc["filename"] == filename) for doc in results]


def compute_relevance_total(ground_truth, search_function):
    return [compute_relevance(record, search_function) for record in tqdm(ground_truth)]


def hit_rate(relevance):
    return sum(1 for line in relevance if 1 in line) / len(relevance)


def mrr(relevance):
    total_score = 0.0
    for line in relevance:
        for rank, value in enumerate(line):
            if value == 1:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance)


def evaluate(ground_truth, search_function):
    relevance_total = compute_relevance_total(ground_truth, search_function)
    return {"hit_rate": hit_rate(relevance_total), "mrr": mrr(relevance_total)}
