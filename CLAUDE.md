# CLAUDE.md

Durable context for this repository. Read this fully at the start of every session.
Keep it accurate: if an architectural decision changes or the goals of the project change, update this file in the same PR.

---

## 1. What this project is

An automated pipeline that generates short-form videos (YouTube Shorts / Reels / TikTok
format) and publishes them, built as a **LangGraph multi-agent system**.

**Primary goal: a portfolio-grade demonstration of LLM-systems engineering.**
The interesting, reviewable parts are the *system*, not the video output:
- a stateful graph with real branching and a checkpointer that provably resumes,
- an evaluation harness with a dataset and metrics that improve over time,
- a human-in-the-loop approval gate,
- a retrieval/memory loop that learns from real performance data.

Secondary goal: it should be *runnable* to grow real accounts if it gets traction.
Every decision favors the portfolio goal first, but must not foreclose the "go real" path.

Because of that, two design invariants (Section 4) are **non-negotiable**. Everything else
is changeable.

---

## 2. Tech stack

- **Language:** Python 3.12+, fully type-hinted.
- **Orchestration:** LangGraph (+ langchain-core for message types only).
- **State:** Pydantic v2 models. No untyped dicts flowing through the graph.
- **Checkpointer:** SQLite for local dev, Postgres for the "real" deployment story.
  Code against the checkpointer interface so swapping is a config change.
- **LLM:** OpenRouter free models, two capability tiers behind one interface (Section 4a).
  OpenAI-compatible endpoint at `https://openrouter.ai/api/v1`; auth via `OPENROUTER_API_KEY`.
  - **Draft tier:** a small/fast free model for the scriptwriter and ideation nodes
    (the high-volume calls).
  - **Judge tier:** a distinctly stronger free model for the eval critic (must outclass
    the drafter — see Section 5).
  - Both model IDs come from env (`LLM_DRAFT_MODEL`, `LLM_JUDGE_MODEL`), never hardcoded:
    `:free` models are retired/moved to paid without notice, so a hardcoded ID is a
    time-bomb.
  - **Free-tier limits are load-bearing:** 20 requests/minute always; 50 requests/day
    until a one-time $10 credit purchase raises it to 1,000/day permanently. Assume the
    $10 top-up is done. A 429 means back off; failed attempts still consume daily quota,
    so the provider MUST use exponential backoff, never blind retry (see Section 7).
  - Free endpoints may log prompts for training. Fine here (no sensitive data), but the
    OpenRouter data-policy toggle must be enabled to reach some free models.
- **TTS:** behind an interface. ElevenLabs for quality; a local fallback (e.g. Piper)
  for zero-cost dev.
- **Video assets:** behind an interface. Default = stock-assembly (Pexels/Pixabay + ffmpeg).
  Generative video (fal.ai / Veo / etc.) is an alternate implementation, off by default.
- **Rendering:** ffmpeg (via a thin wrapper). Keep it Python-only; do not pull in Node
  for rendering unless a task explicitly calls for Remotion.
- **Publishing:** behind an interface. Default = file-writer (writes the video + metadata
  locally). Real implementations: YouTube Data API (direct) and a pre-audited posting
  aggregator for TikTok/Reels. See Section 4b — never auto-post by default.
- **API layer:** FastAPI, streaming graph events via SSE/WebSocket to the UI.
- **UI (later phase):** Next.js + Tailwind. It is an *operator console*, not a video editor
  (see Section 6).
- **Deps / env:** `uv` for dependency + venv management.
- **Tests:** pytest. **Evals:** pytest-driven harness (see Section 5).
- **Tracing:** structured logging always; LangSmith optional via env.

---

## 3. Repository layout

```
src/videoagent/
  graph/
    state.py          # Pydantic state schema — the single source of truth for graph state
    graph.py          # build_graph(): nodes + edges + conditional edges + checkpointer
    nodes/
      ideation.py     # picks topic; retrieves past winners from memory (RAG)
      scriptwriter.py # writes hook/body/CTA  (LLM: draft tier)
      eval_critic.py  # LLM-as-judge rubric score; drives the retry conditional edge (judge tier)
      assets.py       # TTS + visuals via providers
      render.py       # ffmpeg render + QA gate (duration, aspect ratio, safety)
      approval.py     # human-in-the-loop interrupt()
      publish.py      # PublishProvider call; idempotent
      metrics.py      # ingest real performance data; write back to memory
  providers/
    llm.py            # LLMProvider protocol + OpenRouterProvider (draft + judge tiers)
    tts.py            # TTSProvider protocol + ElevenLabsProvider, PiperProvider
    video.py          # VideoProvider protocol + StockAssemblyProvider, GenerativeProvider
    publish.py        # PublishProvider protocol + FileProvider, YouTubeProvider, AggregatorProvider
  memory/
    store.py          # embedding + retrieval wrapper over the video/performance history
  evals/
    dataset/          # labeled examples (JSONL)
    rubric.py         # the scoring rubric, shared by eval_critic node and offline harness
    run_evals.py      # offline regression suite
  config.py           # Settings; selects providers/tiers from env
  api/
    main.py           # FastAPI app; SSE stream of graph events
tests/
ui/                   # Next.js operator console (built in a later phase)
CLAUDE.md  PLAN.md  README.md  pyproject.toml  .env.example
```

---

## 4. Invariants — DO NOT violate without an explicit instruction to change them

### 4a. Everything external goes behind a provider interface
LLM, TTS, video, and publishing are each a Protocol with swappable implementations,
selected in `config.py` from environment variables. Nodes depend on the *interface*,
never on a concrete SDK, and never on a hardcoded model ID. Reasons: models and prices
move fast (the Sora 2 API is slated to sunset; OpenRouter `:free` models rotate out
without notice), and the two-tier LLM strategy depends on this seam. A node that imports
an SDK client (openai/openrouter/elevenlabs) or bakes in a model string directly is a bug.

### 4b. Publishing is idempotent and never silently auto-posts
- Compute a content hash for each video; a retry must never double-publish. Persist
  published hashes and check before posting.
- The default `PublishProvider` is `FileProvider` (writes locally). Real posting
  (YouTube/aggregator) is opt-in via config.
- The `approval.py` node uses LangGraph's `interrupt()`; the graph pauses and only a
  human resume advances to publish. Do not add a "skip approval" default path.

### 4c. State is typed and checkpointed
All graph state is the Pydantic model in `state.py`. Every node reads/writes that model.
The checkpointer persists after each node so the graph resumes from the last completed
node. Preserve this — it is a core portfolio feature. A demonstrable
"kill mid-run, restart, resume" must keep working.

### 4d. The eval rubric is shared
`evals/rubric.py` defines the scoring rubric once. The `eval_critic` node and the offline
`run_evals.py` harness both import it. Do not duplicate the rubric.

---

## 5. Evals are the centerpiece — treat them as first-class

Two kinds, both required:
- **Inline eval:** `eval_critic` scores each script against the rubric; a conditional edge
  loops back to `scriptwriter` (bounded retries, e.g. max 2) when below threshold.
- **Offline eval:** `run_evals.py` runs the pipeline's LLM steps against a small labeled
  dataset and reports metrics, so prompt changes that cause regressions are caught. This
  suite is the most differentiating artifact in the repo. When you change a prompt, run it.

Use the **judge-tier** model as the judge — a distinctly stronger free model than the
drafter (the draft model writes, the judge model grades). If eval-harness results look
noisy or arbitrary, suspect the judge model is too weak before you blame the rubric.

---

## 6. UI = operator console (later phase; do not start until PLAN.md says so)

Build a control/observability surface, NOT a consumer editor:
- live graph execution (nodes lighting up), current checkpointed state,
- eval scores + visible retry loop,
- the human-approval panel (approve/reject → resumes the graph),
- content library with per-video real performance metrics feeding memory.
When building UI, read `/mnt/skills/public/frontend-design/SKILL.md` first.

---

## 7. Conventions & workflow

- Small, task-scoped changes. One PLAN.md task per change where possible.
- Run `pytest` before considering a task done. Add/adjust tests with the code.
- Never hardcode secrets. All keys via env; keep `.env.example` current.
- Prefer async where it touches I/O (providers, API). Keep node functions pure w.r.t.
  state in / state out.
- Keep this file and PLAN.md truthful. Check off / update PLAN.md as tasks complete.
- Cost awareness: record per-run spend in state and surface it. A "cost per video"
  number is a portfolio detail worth keeping accurate.

## 8. Don'ts
- Don't import a provider SDK inside a node (Section 4a).
- Don't add a default that auto-publishes or bypasses approval (Section 4b).
- Don't replace typed state with dicts (Section 4c).
- Don't build the consumer-editor UI (Section 6).
- Don't start the UI or generative-video/real-posting work before Phase 1's graph +
  evals are green (see PLAN.md).
- Don't add heavyweight deps (Node rendering, vector DB servers) without a task that
  requires it; start with the simplest local option.