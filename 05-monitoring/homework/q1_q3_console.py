from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

from starter import client, index  # noqa: E402
from traced import RAGTraced  # noqa: E402

rag = RAGTraced(index=index, llm_client=client, model="deepseek-chat")

query = "How does the agentic loop keep calling the model until it stops?"
answer = rag.rag(query)
print("\n=== ANSWER ===")
print(answer)
