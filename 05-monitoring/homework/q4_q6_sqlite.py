import sys

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from sqlite_exporter import SQLiteSpanExporter

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter("traces.db")))
trace.set_tracer_provider(provider)

from starter import client, index  # noqa: E402
from traced import RAGTraced  # noqa: E402

rag = RAGTraced(index=index, llm_client=client, model="deepseek-chat")

query = "How does the agentic loop keep calling the model until it stops?"
n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1

for i in range(n_runs):
    answer = rag.rag(query)
    print(f"run {i + 1}/{n_runs} done")
