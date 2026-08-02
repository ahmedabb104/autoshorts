# PLAN.md

Sequential build plan. Work top-to-bottom. Do not start a phase until the previous
phase's checklist is fully green (`pytest` passing, invariants in CLAUDE.md upheld).
Check items off as you complete them and keep this file current.

The ordering is deliberate: the impressive, hard-to-fake parts (typed state, checkpointing,
evals, the retry loop, the human gate) are built **first**, against mocked assets. Real
assets, real posting, and the UI come after the system is proven.

---

## Phase 0 — Project skeleton

- [x] `uv` project init; `pyproject.toml` with pinned deps (langgraph, langchain-core,
      pydantic v2, fastapi, uvicorn, httpx, pytest, python-dotenv).
      *Note: `uv` itself lives in a separate `.venv-tools/` venv so that `uv sync`
      (which prunes non-project packages) cannot uninstall its own binary. The project
      env is `.venv/`, driven by `UV_PROJECT_ENVIRONMENT=.venv`.*
- [x] Create the directory layout from CLAUDE.md Section 3 (empty modules + docstrings).
      *`ui/` deliberately not created — it belongs to Phase 3d.*
- [x] `config.py`: `Settings` (pydantic-settings) reading env, including provider/tier
      selectors (`LLM_TIER`, `TTS_PROVIDER`, `VIDEO_PROVIDER`, `PUBLISH_PROVIDER`).
      *33 settings. `LLM_TIER` implemented as a global tier override (`auto|draft|judge`,
      default `auto`), not a model picker — the writer/grader split must survive it.
      `LLM_DRAFT_MODEL`/`LLM_JUDGE_MODEL` have no defaults; `require_model()` raises at
      use time. `Settings()` succeeds on an empty env, so tests need no secrets.*
- [x] `.env.example` documenting every env var. README with run/test instructions.
      *Verified 33/33 against `Settings.model_fields` — no missing, no extras.*
- [x] CI-ish: a `make`/`uv run` task that runs `pytest` and a linter (ruff).
      *`scripts/check.py` (ruff check → ruff format --check → pytest; `--fix` variant).
      pytest exit 5 = warning, not failure. `Makefile` forwards to it for unix/CI;
      `make` is not installed on the primary dev machine. `[project.scripts]` was tried
      and does not work — `scripts/` isn't in the installed distribution.*

**Done when:** repo installs clean, `pytest` runs (even with 0 tests), config loads from env.
✅ **Phase 0 complete** — `uv sync --all-groups` clean, `scripts/check.py` green
(ruff + 29 tests), config loads from env.

Open questions surfaced during Phase 0, to settle in the phase that hits them:
- **1b:** `LLM_TIER` turned out to be a tier override, so there is still no
  `LLM_PROVIDER` (openrouter|fake) selector for offline tests. Decide whether Phase 1b
  wants one, or whether tests inject a fake provider directly.
- **1c:** `EVAL_SCORE_THRESHOLD` is bounded `0-10` with default `7.0`. `rubric.py` must
  either use that scale or the bound needs revisiting.

---

## Phase 1 — The graph and the evals (assets fully MOCKED)

This is the core portfolio deliverable. No real API spend required.

### 1a. State + graph shell ✅
- [x] `graph/state.py`: the Pydantic state schema — topic, script fields (hook/body/cta),
      eval score, asset paths, content hash, status enum, cost accumulator, retry count.
      *`completed_nodes` and `costs` are reducer-backed (`operator.add`) so they accumulate
      instead of last-writer-wins. `content_fingerprint()` hashes topic + script only, so
      a retry of identical content dedupes.*
- [x] `graph/graph.py`: `build_graph()` wiring nodes with a **SQLite checkpointer**.
      *`build_graph(checkpointer)` is pure; `open_checkpointer()`/`open_graph()` own the
      connection lifecycle. `AsyncSqliteSaver`, `durability="sync"` (LangGraph defaults to
      `"async"`, which can lose the last node's checkpoint on a hard kill).*
- [x] Stub every node to read/write state and return; graph runs end-to-end on stubs.
      *All 8 nodes are `async` — they all do I/O from 1b onward.*
- [x] Test: run the graph on a fixture; assert state transitions and that a checkpoint row
      is written after each node. *10 checkpoints for 8 nodes (input + one per node
      boundary + terminal), asserted both via `aget_state_history` and by counting rows in
      the SQLite file after the connection closes.*
- [x] Test: **resumability** — interrupt after node N, rebuild graph from the checkpointer,
      resume, assert it continues from N+1 (not from the start). *Crash injected inside
      `render`; graph, checkpointer, and connection all torn down and rebuilt; the four
      completed nodes each ran exactly once across both halves.*

Decisions and findings from 1a:
- **LangGraph pinned to 1.x** (was `<0.8`, two majors stale; 1.2.10 now). Required raising
  langchain-core to `>=1.4.7` and brought `langgraph-checkpoint-sqlite` 3.1.1.
- **Checkpoint serializer allowlist.** LangGraph only reconstructs allowlisted types when
  reading a checkpoint. Our Pydantic types were not on it — tolerated with a warning today,
  blocked in a future release, which would have broken resume at upgrade time.
  `state.CHECKPOINTED_TYPES` + `graph.build_serde()` register them. **Anything new in
  `VideoState` that is not a plain builtin must be added to `CHECKPOINTED_TYPES`** — a
  missing type is *blocked on read* and returns a non-object rather than raising. There is
  a test for exactly this.
- **The approval gate is already enforced**, ahead of 1e: `publish` refuses to act unless
  `approved is True`, so the not-yet-implemented `interrupt()` fails closed, not open.
- **Resolved from Phase 0:** `open_checkpointer()` creates the parent dir of
  `SQLITE_CHECKPOINT_PATH`. `CHECKPOINTER=postgres` raises `NotImplementedError` rather
  than silently falling back to SQLite.

### 1b. LLM provider (two-tier) + scriptwriter + ideation
- [ ] `providers/llm.py`: `LLMProvider` Protocol + `OpenRouterProvider` (OpenAI-compatible
      endpoint `https://openrouter.ai/api/v1`). Model IDs come from env
      (`LLM_DRAFT_MODEL`, `LLM_JUDGE_MODEL`) — never hardcoded (`:free` IDs get retired).
      Include exponential backoff on 429 (failed calls still burn daily quota).
- [ ] `nodes/scriptwriter.py`: real prompt, uses the **draft** model (a `:free` model).
- [ ] `nodes/ideation.py`: for now, pick a topic from a seed list (memory retrieval is
      wired in 3c). Uses the draft model.
- [ ] Tests with a fake LLM provider (no network) asserting prompt shape + parsing,
      plus a test that a simulated 429 triggers backoff rather than a blind retry.

### 1c. Eval critic + retry loop  ← highest-value work
- [ ] `evals/rubric.py`: the shared rubric (e.g. hook strength, clarity, factual-risk flag),
      returning a structured score.
- [ ] `nodes/eval_critic.py`: LLM-as-judge using the **judge** model (`LLM_JUDGE_MODEL`,
      distinctly stronger than the drafter); returns rubric score. Runs once per video, so
      pointing this one model at a cheap *paid* ID is the low-cost way to keep evals
      reliable if free judge models get throttled — it's a config change, no code change.
- [ ] `graph/graph.py`: conditional edge — below threshold → back to `scriptwriter`
      (bounded, max 2 retries via the retry counter in state); at/above → continue.
- [ ] Test: a deliberately weak script triggers the retry edge; a strong one does not;
      retries are bounded.

### 1d. Offline eval harness
- [ ] `evals/dataset/`: ~15–25 labeled examples (script + expected judgment) as JSONL.
- [ ] `evals/run_evals.py`: runs the judged step over the dataset, reports metrics
      (agreement / precision on the factual-risk flag / mean score). Prints a summary table.
- [ ] Wire it into the test task so regressions surface. Document how to run it in README.

### 1e. Human-approval interrupt + mocked publish
- [ ] `nodes/approval.py`: `interrupt()` — graph pauses awaiting human resume.
- [ ] `providers/publish.py`: `PublishProvider` Protocol + `FileProvider` (writes video +
      metadata JSON locally). Compute + persist the **content hash**; refuse duplicates.
- [ ] `nodes/publish.py`: idempotent call through the provider.
- [ ] Test: approve → publishes once; resume-with-same-state twice → no double publish.

**Done when:** the full graph runs on mocked assets, retries on weak scripts, pauses for
approval, publishes idempotently to disk, resumes after a kill, and `run_evals.py` reports
metrics. This alone is a strong portfolio project.

---

## Phase 2 — Real assets, cheaply

- [ ] `providers/tts.py`: `TTSProvider` Protocol + `ElevenLabsProvider` and a local
      `PiperProvider` (zero-cost dev default).
- [ ] `providers/video.py`: `VideoProvider` Protocol + `StockAssemblyProvider`
      (Pexels/Pixabay search → download clips). `GenerativeProvider` stubbed/off by default.
- [ ] `nodes/assets.py`: real TTS + visual selection through the providers.
- [ ] `nodes/render.py`: ffmpeg wrapper — assemble clips + voiceover + captions to a
      vertical (9:16) short; QA gate checks duration, aspect ratio, presence of audio.
- [ ] Cost tracking: each provider reports spend into the state cost accumulator.
- [ ] Tests: providers behind fakes; a small real smoke test gated behind an env flag.

**Done when:** a real (if simple) vertical short renders from a topic, voiced and captioned,
for a few cents, with cost recorded.

---

## Phase 3 — Close the loop + operator console

### 3a. Real publishing (opt-in)
- [ ] `YouTubeProvider` (YouTube Data API, direct — the platform where we can also read
      real metrics for our own channel).
- [ ] `AggregatorProvider` for TikTok/Reels via a pre-audited posting API.
- [ ] Keep `FileProvider` as default; real providers opt-in via `PUBLISH_PROVIDER`.

### 3b. Metrics ingestion
- [ ] `nodes/metrics.py`: pull real performance (start with YouTube Analytics for our own
      channel: views, retention). Write results back to memory. For platforms without easy
      metrics access, allow simulated metrics behind a flag.

### 3c. RAG memory loop
- [ ] `memory/store.py`: embed + store each published video with script, topic, and its
      real performance; retrieval of top performers.
- [ ] `nodes/ideation.py`: retrieve past winners as few-shot context. The feedback loop is
      now closed on real data (at least on YouTube).

### 3d. Operator console UI  (read frontend-design SKILL.md first)
- [ ] `api/main.py`: FastAPI streaming graph events (SSE/WebSocket) + endpoints to
      trigger a run and to approve/reject at the interrupt.
- [ ] `ui/` (Next.js + Tailwind): live graph view, checkpointed state, eval score + retry
      loop, approval panel, content library with real performance metrics.

**Done when:** you can trigger a run from the UI, watch the graph execute live, approve a
video, publish it to a real YouTube channel, and see its real retention feed back into the
memory that shapes the next video's ideation.

---

## Guardrails reminder (see CLAUDE.md §4)
Provider interfaces for all external calls · idempotent, non-auto publishing · typed +
checkpointed state · one shared eval rubric. If a task seems to require breaking one of
these, stop and flag it rather than working around it.