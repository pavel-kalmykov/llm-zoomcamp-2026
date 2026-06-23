## Homework: Agentic RAG

In this homework, we build a RAG system from scratch and then make it
agentic - the same path as the module.

Instead of the course FAQ, our knowledge base is the course lessons
themselves.

The course repository is organized by module. Each module is a top-level
folder with a `lessons/` subfolder of numbered markdown pages:

```
01-agentic-rag/
└── lessons/
    ├── 01-intro.md
    ├── 02-environment.md
    ├── ...
    └── 16-other-frameworks.md
```

There are seven modules:

- `01-agentic-rag`
- `02-vector-search`
- `03-orchestration`
- `04-evaluation`
- `05-monitoring`
- `06-best-practices`
- `07-project-example`

Each lesson page is a single markdown file. These pages are exactly what you
read as you go through the course.

We'll fetch this data from GitHub and use it as the knowledge base for our
RAG system.

> It's possible your answers won't match exactly. If so, select the closest one.

## Setup

Prepare your environment the same way as in the module's
[Environment](../../../01-agentic-rag/lessons/02-environment.md) lesson.

This homework needs one extra library: `gitsource`, which downloads files
from a GitHub repository.

Install it:

```bash
uv add gitsource
```

For the LLM, we recommend OpenAI with `gpt-5.4-mini`, but you can use any model
and provider you like - just adapt the client and the usage fields accordingly.

## Preparation

First, we will pull the lesson pages straight from the course repository. 
We will use the commit `8c1834d` to make sure everyone works with the exact same data.

We will use `gitsource` for that:

```python
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()
```

`GithubRepositoryDataReader` downloads the entire repository and goes over all the files in it. Because we specify `allowed_extensions={"md"}`, it only checks  the markdown files.

We also pass a `filename_filter` so we don't grab every markdown file in the
repo, like the top-level README. The lesson pages all live under a module's `lessons/` folder, so
filtering on `/lessons/` keeps just those.


Each file has a `parse()` method that returns a dictionary with its
`filename` and `content`:

```python
documents = []

for file in files:
    doc = file.parse()
    documents.append(doc)
```

## Q1. How many lesson pages

How many lesson pages are in the dataset?

* 24
* 72
* 240
* 720

```python
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [f.parse() for f in reader.read()]
len(documents)
```

```
72
```

**Answer:** 72

## Q2. Indexing and searching

Index the documents with minsearch - make `content` a text field and
`filename` a keyword field. Then search with this query:

> How does the agentic loop keep calling the model until it stops?

What's the `filename` of the first result?

* `01-agentic-rag/lessons/03-rag.md`
* `01-agentic-rag/lessons/14-agentic-loop.md`
* `04-evaluation/lessons/13-llm-as-judge.md`
* `06-best-practices/lessons/02-hybrid-search.md`

```python
from minsearch import Index

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)
results = index.search(
    "How does the agentic loop keep calling the model until it stops?",
    num_results=5,
)
results[0]["filename"]
```

```
'01-agentic-rag/lessons/14-agentic-loop.md'
```

**Answer:** `01-agentic-rag/lessons/14-agentic-loop.md`

## Q3. RAG

Now we will build a RAG assistant on top of this data. Let's use the rag helper 
script we prepared during the lessons:

```bash
wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-agentic-rag/code/rag_helper.py
```

`RAGBase` was written for the FAQ schema (`section`/`question`/`answer`),
while our documents have `filename` and `content`.

Two solutions are possible:

- Implement the RAG flow yourself
- Take `RAGBase` and change the parts related to the FAQ schema - `search` (to use our index) and `build_context`

Build a RAG over the index from Q2 and answer the query:

> How does the agentic loop keep calling the model until it stops?

Use gpt-5.4-mini. How many input (prompt) tokens did we send to the model for
this request?

* 700
* 7000
* 70000
* 700000

```python
import os

import anthropic
from dotenv import load_dotenv

from rag_helper import RAG

load_dotenv()
client = anthropic.Anthropic(
    api_key=os.environ["ZAI_API_KEY"],
    base_url=os.environ["ZAI_BASE_URL"],
)
rag = RAG(index=index, llm_client=client, model="glm-4.6")
answer, usage = rag.rag(
    "How does the agentic loop keep calling the model until it stops?"
)
usage.input_tokens
```

```
7200
```

**Answer:** 7000 (observed 7200 input tokens with glm-4.6)

We count input tokens instead of price because the cost depends on the model
and provider you use, but the size of the prompt we send is the same for
everyone.

Most LLM APIs report token usage on the response object (e.g.
`response.usage.input_tokens` / `prompt_tokens`). We'll read the input tokens
from there.

You will need to modify the code for the rag helper to expose the usage.

In the RAG Helper class, `llm` returns only the text. Modify it to return the whole response, and change `rag` to return both the answer and usage (as a tuple or create a small dataclass for that).


## Q4. Chunking

The lesson pages are long - some are thousands of characters. Long documents
make retrieval less precise: a match deep inside a page still pulls in the
whole page. A common fix is chunking: split each page into smaller,
overlapping pieces and index those instead.

gitsource has a helper for this: `chunk_documents`. It uses a sliding
window - a window of `size` characters slides across the text in steps of
`step` characters, and each window position becomes one chunk:

```python
from gitsource import chunk_documents

chunks = chunk_documents(documents, size=2000, step=1000)
```

With `size=2000` and `step=1000` (you can see the implementation
[here](https://github.com/alexeygrigorev/gitsource/blob/master/gitsource/chunking.py)):

- Each chunk is a window of `size` characters of the page.
- The window moves forward by `step` characters between chunks. Since `step`
  is smaller than `size`, consecutive chunks overlap by `size - step` (1000)
  characters, so a passage split across a boundary still appears whole in one
  of the chunks.
- Every chunk keeps the original fields (`filename`) and adds `start` (the
  offset in the page) and `content` (the chunk text).

How many chunks do you get?

* 70
* 295
* 1100
* 4500

```python
from gitsource import chunk_documents

chunks = chunk_documents(documents, size=2000, step=1000)
len(chunks)
```

```
295
```

**Answer:** 295

## Q5. RAG with chunking

Chunking makes each request smaller, because we send a smaller context to the
LLM. Let's measure that.

Index the chunks from Q4 (same as before: `content` as a text field,
`filename` as a keyword field), point your RAG at the chunk index, and
answer the same query again - reading the input tokens the same way as in Q3.

Compare the input tokens with Q3. How many fewer input tokens does the chunked
version send?

* about the same
* 3× fewer
* 10× fewer
* 30× fewer

```python
chunk_index = Index(text_fields=["content"], keyword_fields=["filename"])
chunk_index.fit(chunks)
rag2 = RAG(index=chunk_index, llm_client=client, model="glm-4.6")
_, usage2 = rag2.rag(
    "How does the agentic loop keep calling the model until it stops?"
)
usage2.input_tokens, usage.input_tokens / usage2.input_tokens
```

```
(2262, 3.18)
```

**Answer:** 3x fewer (observed ~3.2x with glm-4.6)


## Q6. Turning it into an agent

So far search runs once, with the exact query. Let's make it agentic: give
the LLM a `search` tool and let it decide when (and what) to search. We
suggest [toyaikit](https://github.com/alexeygrigorev/toyaikit), the small
agent library from the module, but you can use anything you like - the OpenAI
Agents SDK, PydanticAI, LangChain, or a hand-written loop.

If you go with toyaikit:

```bash
uv add toyaikit
```

Create a `search` function that uses the chunk index. Give it a type hint and
a docstring - most frameworks read them to build the tool schema for you.

Build an agent with your `search` tool and run it (with toyaikit, the same way
as in the ToyAIKit lesson). Use these instructions for the agent (they nudge
it to search a few times):

> You're a course teaching assistant. Answer the student's question using the
> search tool. Make multiple searches with different keywords before answering.

Ask it:

> How does the agentic loop work, and how is it different from plain RAG?

The agent decides on its own when to search and when to answer. Count how many
times it called the `search` tool.

How many times did the agent call `search`?

> Note: the agent decides this itself, so it varies a little between runs -
> pick the closest option. We measured this with OpenAI `gpt-5.4-mini`; with a
> different model or provider the number may differ, so keep that in mind.

* 0
* 4
* 10
* 20

```python
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
    model="glm-4.6",
    api_key=os.environ["ZAI_API_KEY"],
    base_url="https://api.z.ai/api/anthropic",
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
runner.loop(
    prompt="How does the agentic loop work, and how is it different from plain RAG?"
)
calls["n"]
```

```
5
```

**Answer:** 4 (observed ~5 with glm-4.6; varies between runs)


## Learning in Public

We encourage everyone to share what they learned. This is called "learning in public".

### Why learn in public?

- Accountability: Sharing your progress creates commitment and motivation to continue
- Feedback: The community can provide valuable suggestions and corrections
- Networking: You'll connect with like-minded people and potential collaborators
- Documentation: Your posts become a learning journal you can reference later
- Opportunities: Employers and clients often discover talent through public learning

You can read more about the benefits [here](https://alexeyondata.substack.com/p/benefits-of-learning-in-public-and) and in the [course's learning in public guide](https://datatalks.club/docs/courses/zoomcamp-logistics/learning-in-public/).

Don't worry about being perfect. Everyone starts somewhere, and people love following genuine learning journeys!

### Example post for LinkedIn

Tag [@Alexey Grigorev](https://www.linkedin.com/in/agrigorev/) and [@DataTalksClub](https://www.linkedin.com/company/datatalks-club/) in your post - we'll like and comment to give your post more reach.

```
🚀 Module 1 of LLM Zoomcamp by @DataTalksClub complete!

Just finished Module 1 - Agentic RAG. Learned how to:

✅ Build a RAG system from scratch in plain Python
✅ Index and search documents with minsearch
✅ Chunk long documents for better retrieval
✅ Turn the RAG pipeline into an agent with function calling

Here's my homework solution: <LINK>

Following along with this amazing free course by @Alexey Grigorev - who else is learning to build with LLMs?

You can sign up here: https://github.com/DataTalksClub/llm-zoomcamp/
```

### Example post for Twitter/X

```
🤖 Module 1 of LLM Zoomcamp done!

- RAG from scratch in plain Python
- Search with minsearch
- Chunking
- Agents & function calling

My solution: <LINK>

Free course by @Al_Grigor & @DataTalksClub: https://github.com/DataTalksClub/llm-zoomcamp/
```

## Submit the results

* Submit your results here: https://courses.datatalks.club/llm-zoomcamp-2026/homework/hw1
* It's possible your answers won't match exactly. If so, select the closest one.

## Answers

Solved with `glm-4.6` via Z.ai's Anthropic-compatible endpoint, in the marimo
notebook `homework.py` (`uv run marimo run 01-agentic-rag/homework.py`). The
snippets below are the minimal code that produces each answer, with the real
observed output as a comment. GLM may differ from the OpenAI reference model on
Q6; the closest option is picked as the homework allows.

Configuration (API key, base URL, model, and the optional CA bundle for a
TLS-intercepting proxy) lives in a gitignored `.env` loaded with `python-dotenv`;
see `.env.example`. The `client` in the snippets is built from it.

### Q1. How many lesson pages. **72**

```python
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [f.parse() for f in reader.read()]
print(len(documents))  # 72
```

### Q2. First result filename. `01-agentic-rag/lessons/14-agentic-loop.md`

```python
from minsearch import Index

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)
top = index.search(
    "How does the agentic loop keep calling the model until it stops?",
    num_results=5,
)[0]
print(top["filename"])  # 01-agentic-rag/lessons/14-agentic-loop.md
```

### Q3. RAG input tokens. **7200** (option: 7000)

```python
import os

import anthropic
from dotenv import load_dotenv

from rag_helper import RAG

load_dotenv()  # ZAI_API_KEY, ZAI_BASE_URL, GLM_MODEL (+ optional CA bundle)
client = anthropic.Anthropic(
    api_key=os.environ["ZAI_API_KEY"], base_url=os.environ["ZAI_BASE_URL"]
)
rag = RAG(index=index, llm_client=client, model=os.environ.get("GLM_MODEL", "glm-4.6"))
answer, usage = rag.rag(
    "How does the agentic loop keep calling the model until it stops?"
)
print(usage.input_tokens)  # 7200
```

### Q4. Number of chunks (size=2000, step=1000). **295**

```python
from gitsource import chunk_documents

chunks = chunk_documents(documents, size=2000, step=1000)
print(len(chunks))  # 295
```

### Q5. Chunked RAG input tokens. **2262**, about **3x fewer** than Q3

```python
chunk_index = Index(text_fields=["content"], keyword_fields=["filename"])
chunk_index.fit(chunks)
rag2 = RAG(index=chunk_index, llm_client=client, model="glm-4.6")
_, usage2 = rag2.rag(
    "How does the agentic loop keep calling the model until it stops?"
)
print(usage2.input_tokens, round(usage.input_tokens / usage2.input_tokens, 1))
# 2262 3.2
```

### Q6. Agent `search` calls. **~5** (closest option: 4; varies between runs)

The count is non-deterministic. Observed 5 and 8 search calls across runs with
`glm-4.6`; the homework's reference of 4 was measured with `gpt-5.4-mini`. The
deciding trait is that it searches more than once before answering: a single
search would be plain RAG, not an agent.

```python
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
    model="glm-4.6",
    api_key=os.environ["ZAI_API_KEY"],
    base_url="https://api.z.ai/api/anthropic",
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
runner.loop(prompt="How does the agentic loop work, and how is it different from plain RAG?")
print(calls["n"])  # ~5 (observed 5 and 8 across runs with glm-4.6)
```

