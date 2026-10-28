# Agentic AI course: LangChain · LangGraph · LangSmith · Langfuse

One project, two notebooks: a customer-support agent for **Chinook Music Store** (Chinook SQLite database + 8 store-policy documents for RAG).

| File | Contents |
|---|---|
| `01_build_agents_langchain_langgraph.ipynb` | Agent theory, LangChain (models, prompts, LCEL, structured output), tools, RAG, `create_agent` + middleware, LangGraph (state, reducers, edges, ReAct from scratch), workflow patterns (chaining, routing, parallel orchestrator-workers, evaluator-optimizer), memory, human-in-the-loop, the full support graph, multi-agent systems (supervisor with agents-as-tools, handoffs/swarm, hierarchical team of specialist subgraphs with shared state and approvals inside a sub-agent), production practices |
| `02_observe_evaluate_improve_langsmith_langfuse.ipynb` | Tracing (LangSmith + Langfuse), golden + synthetic datasets, component evals (router, guardrail, retrieval, single-step tools), end-to-end and trajectory evals, LLM-as-judge and judge validation, experiments, error analysis, accuracy improvements (prompting, prompt registry, context engineering, few-shot, self-consistency, hybrid/rerank/multi-query retrieval, chunking, reflection, automatic prompt optimization, model choice), bootstrap CIs, pairwise evals, architecture comparison, multi-agent evals (coordination metrics, failure attribution, per-agent and multi-turn evals), online evaluation + feedback, data flywheel, CI regression tests, red teaming |
| `music_store_agent.py` | The project code both notebooks import: data setup, tools, retrievers, the LangGraph support graph, the supervisor variant, `run_turn` |

## Run (free by default)
1. Python 3.10+ in a fresh environment (e.g. `conda create -n agents python=3.11`). Open notebook 1; the first cell installs everything.
2. **LLM — free:** create a Groq API key at https://console.groq.com/keys (no credit card). Agent = `groq:llama-3.3-70b-versatile`, judge = `groq:openai/gpt-oss-120b`.
3. **Embeddings — free, local:** `sentence-transformers/all-MiniLM-L6-v2` runs on your CPU (no key).
4. Optional, also free plans: `LANGSMITH_API_KEY`, `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`. Without them, platform cells are skipped.
5. Switch providers with env vars before running the setup cell: `MODEL`, `JUDGE_MODEL`, `EMBED_MODEL` (any `provider:model`, e.g. `ollama:qwen2.5:7b` for fully local, `openai:gpt-4.1-mini` for paid). `RPM` sets the requests-per-minute limit.
6. Free tiers have daily request limits. Notebook 2 runs in `QUICK = True` mode (12-example stratified subset) by default; if you hit the daily limit, continue the next day.
7. The Chinook database downloads automatically into `./data` on first run.

Run notebook 1 before notebook 2 (both use `music_store_agent.py` in the same folder).
