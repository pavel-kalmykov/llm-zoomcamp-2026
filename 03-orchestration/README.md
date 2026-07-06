# Module 3: AI Orchestration with Kestra

Homework solved against a local Kestra instance (Docker), with the course's
`zoomcamp` flows imported and executed. Answers are backed by real execution
outputs where relevant.

## Setup

```bash
# API key (gitignored, lives in 03-orchestration/.env, read by docker compose)
# GEMINI_API_KEY=...
# SECRET_GEMINI_API_KEY=$(base64 of the above)

docker compose up -d          # kestra (v1.3.21) + postgres, UI on :8080
```

Import the homework flows (auth `admin@kestra.io:Admin1234!`):

```bash
for f in flows/*.yaml; do
  curl -X POST -u 'admin@kestra.io:Admin1234!' \
    http://localhost:8080/api/v1/flows/import -F fileUpload=@$f
done
```

Flows used: `1_chat_without_rag`, `2_chat_with_rag`, `4_simple_agent` (the last
one modified for Q5). Flows 5 and 6 need a Tavily key, which is not needed for
the questions here.

## Answers

### Q1. Context Engineering
AI Copilot generates better Kestra flows because it **has access to current
Kestra plugin documentation** (RAG over the docs), not a stronger model.

### Q2. RAG vs No RAG (executed)
`1_chat_without_rag.textOutput` **hallucinates** plausible-but-wrong Kestra 1.1
features (Declarative Python Tasks, an LDAP user store, GCS as internal
storage); `2_chat_with_rag.textOutput` lists real ones from the release notes.
**Answer: the non-RAG response is vague, generic, or fabricated — the model
guesses from training data.**

### Q3. Token usage — short summary (executed)
`4_simple_agent` with `summary_length=short` (SUCCESS). `multilingual_agent`
output tokens for this run: 133. **Answer: 60-100 tokens** (expected range for a
1-2 sentence summary; closest option).

### Q4. Token usage — long summary (executed)
`multilingual_agent` output tokens: short=133, long=127 (ratio ~0.95).
**Answer: about the same (within 20%).** The "long" guideline did not multiply
output on this input.

### Q5. Modifying a flow (executed)
`english_brevity` changed from 1 to 3 sentences, run with `summary_length=long`
(SUCCESS). `english_brevity` output tokens: 3 sentences=88 vs 1 sentence=41.
**Answer: 2-4x more** (~2.15x).

### Q6. Best practices
**Answer: use traditional task-based workflows for predictability and
auditability.** Agents trade determinism for flexibility, the wrong trade for
compliance-heavy production work.
