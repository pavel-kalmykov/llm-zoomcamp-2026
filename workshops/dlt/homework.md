# Homework: dlt

In this homework we will take the FAQ agent from Module 1,
instrument it with [Pydantic Logfire](https://logfire.dev) for
observability,
then pull the trace data back out with dlt and analyze it.

In Module 1 we wrote the agent loop by hand and then we saw toyaikit -
an agentic framework.

For this homework we rewrote into [Pydantic AI](https://ai.pydantic.dev/),
so it's easier to integrate it with Logfire. Pydantic AI and Logfire
work really well together, that's why we use them here.

In Module 5 we learn about monitoring and observability, and implement
our own monitoring solution. Logfire is an alternative for that.

> Solved with `deepseek-chat` (DeepSeek's OpenAI-compatible API) instead of
> OpenAI: the Z.ai balance used for modules 1-4 ran out mid-course, and
> `DEEPSEEK_API_KEY` was already available. Pydantic AI's `OpenAIChatModel` +
> `OpenAIProvider(base_url=...)` makes any OpenAI-compatible endpoint a drop-in
> replacement for `openai:gpt-5.4-mini`. Logfire project is EU-region
> (`logfire-eu.pydantic.dev`).

## Getting the code

The rewritten agent is in the [homework/](homework/) directory.

The agent code is in [homework/agent.py](homework/agent.py). Here we use
Pydantic AI which we didn't cover previously.
Conceptually there's nothing new: we covered everything already in module 1.

## Setup

```bash
uv init
uv add openai minsearch requests python-dotenv pydantic-ai logfire
uv add "dlt[duckdb]>=1.0"
```

> Careful: an unrelated, unmaintained PyPI package is also called `dlt`
> (version `0.1.0`, depends on `keras`/`tensorflow`, nothing to do with
> dlthub). Pin `>=1.0` or the resolver may grab the wrong one; if it does,
> `import dlt` crashes trying to import `keras.datasets`.

`.env`:

```
DEEPSEEK_API_KEY=...
LOGFIRE_TOKEN=...        # write token
LOGFIRE_READ_TOKEN=...   # read token
LOGFIRE_REGION=eu
```

Verify that the agent runs:

```bash
uv run python homework/main.py
```

## Question 1. Instrument the agent with Logfire

Sign up for a free [Logfire](https://logfire.dev) account, create a
project, and generate a write token.

Instrument the agent:

```python
logfire.configure()
logfire.instrument_pydantic_ai()
```

Run the agent a few times with different questions and open your
project on Logfire to see the traces.

For the following query

> How do I run Ollama locally?

how many spans does a single agent run produce?

* 1
* 5
* 15
* 30

```python
import os
import requests

READ_TOKEN = os.environ["LOGFIRE_READ_TOKEN"]
r = requests.get(
    "https://logfire-eu.pydantic.dev/v1/query",
    headers={"Authorization": f"Bearer {READ_TOKEN}"},
    params={"sql": "SELECT trace_id, COUNT(*) FROM records "
                    "WHERE span_name IN ('invoke_agent faq_agent', "
                    "'chat deepseek-chat', 'execute_tool search') "
                    "GROUP BY trace_id"},
)
r.json()
```

```
4 separate runs of the same question: 4, 4, 5, 4 spans
(invoke_agent faq_agent + chat deepseek-chat + execute_tool search +
 chat deepseek-chat, x2 chat/execute_tool pairs on the 5-span run)
```

**Answer:** 5 (closest option; DeepSeek made 1 search in 3/4 runs, 2 in one run)

## Question 2. Load traces into DuckDB with dlt

Generate a read token for your Logfire project.

Pull the trace records via the Logfire Query API and load them with dlt:

```python
import dlt

@dlt.resource(name="records")
def agent_traces():
    yield from fetch_agent_traces()  # paginated Query API call

pipeline = dlt.pipeline(
    pipeline_name="agent_traces_pipeline",
    destination=dlt.destinations.duckdb("traces_db.duckdb"),
    dataset_name="agent_traces",
)
pipeline.run(agent_traces())
```

The logfire traces contain deeply nested JSON (span attributes with
LLM messages, tool calls, token usage, etc.). dlt automatically
normalizes this into a set of tables - one for the main records, plus
child tables for each nested level.

How many tables did dlt create? Check with:

```sql
SELECT COUNT(*) FROM information_schema.tables
WHERE table_schema = 'agent_traces';
```

* 1
* 3
* 24
* 100

```
24

records, records__attributes__gen_ai_input_messages,
records__attributes__gen_ai_input_messages__parts,
records__attributes__gen_ai_input_messages__parts__result,
records__attributes__gen_ai_output_messages,
records__attributes__gen_ai_output_messages__parts,
records__attributes__gen_ai_response_finish_reasons,
records__attributes__gen_ai_system_instructions,
records__attributes__gen_ai_tool_call_result,
records__attributes__gen_ai_tool_definitions,
records__attributes__gen_ai_tool_definitions__parameters__required,
records__attributes__logfire_metrics__gen_ai_client_token_usage__details,
records__attributes__logfire_metrics__operation_cost__details,
records__attributes__logfire_scrubbed,
records__attributes__logfire_scrubbed__path,
records__attributes__model_request_parameters__function_tools,
records__attributes__model_request_parameters__function_tools__parameters_json_schema__required,
records__attributes__model_request_parameters__instruction_parts,
records__attributes__pydantic_ai_all_messages,
records__attributes__pydantic_ai_all_messages__parts,
records__attributes__pydantic_ai_all_messages__parts__result,
_dlt_loads, _dlt_pipeline_state, _dlt_version
```

**Answer:** 24

## Question 3. Query traces with an agent

Find the input token usage for the agent run from Q1.

The token counts are stored in the span attributes as
`gen_ai.usage.input_tokens` (normalized by dlt to the column
`attributes__gen_ai_usage_input_tokens`). Sum them across all LLM calls
within the trace.

```python
import duckdb

con = duckdb.connect("traces_db.duckdb")
con.execute("""
    SELECT trace_id, SUM(attributes__gen_ai_usage_input_tokens)
    FROM agent_traces.records
    GROUP BY trace_id
""").fetchall()
```

```
[('...f94844', 2151), ('...b8a8a', 2150), ('...e2f4e', 3587), ('...c06a95', 2151)]
```

The number depends on how many searches the agent made (2151 for the
3 single-search runs, 3587 for the one run that searched twice).

* 100 - 500
* 1500 - 5000
* 10000 - 20000
* 50000 - 100000

**Answer:** 1500 - 5000

## Submit the results

* Submit your results here: https://courses.datatalks.club/llm-zoomcamp-2026/homework/dlt
