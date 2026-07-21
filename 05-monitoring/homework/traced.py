"""RAGTraced: wraps RAGBase's rag(), search(), and llm() each in their own
OpenTelemetry span. Used for Q1-Q3 (console exporter) and Q4-Q6 (SQLite
exporter) of the module 5 homework.
"""

from opentelemetry import trace
from rag_helper import RAGBase


class RAGTraced(RAGBase):
    def __init__(self, *args, tracer_name="llm-zoomcamp", **kwargs):
        super().__init__(*args, **kwargs)
        self.tracer = trace.get_tracer(tracer_name)

    def search(self, query, num_results=5):
        with self.tracer.start_as_current_span("search"):
            return super().search(query, num_results=num_results)

    def llm(self, prompt):
        with self.tracer.start_as_current_span("llm") as span:
            response = super().llm(prompt)
            usage = response.usage
            span.set_attribute("input_tokens", usage.prompt_tokens)
            span.set_attribute("output_tokens", usage.completion_tokens)
            # DeepSeek pricing (per 1M tokens): $0.28 input, $0.42 output
            cost = (usage.prompt_tokens / 1_000_000) * 0.28 + (
                usage.completion_tokens / 1_000_000
            ) * 0.42
            span.set_attribute("cost", cost)
            return response

    def rag(self, query):
        with self.tracer.start_as_current_span("rag"):
            return super().rag(query)
