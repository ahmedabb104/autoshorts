# autoshorts

An automated pipeline that generates short-form videos (YouTube Shorts / Reels / TikTok
format) and publishes them, built as a **LangGraph multi-agent system**.

The point of this repo is the *system*, not the video output:

- a **typed, checkpointed state graph** that provably resumes after a kill,
- an **evaluation harness** — an inline LLM-as-judge critic driving a bounded retry loop,
  plus an offline regression suite over a labeled dataset,
- a **human-in-the-loop approval gate** the graph pauses on,
- a **retrieval/memory loop** that feeds real performance data back into ideation.

Everything external — LLM, TTS, video assets, publishing — sits behind a provider
interface chosen from the environment, so swapping a retired model or a paid backend is a
config change, not a code change.

> **Status: Phase 1d complete.** The graph runs end-to-end, checkpoints every node to
> SQLite, and provably resumes after a crash. Ideation and the scriptwriter make real
> two-tier OpenRouter calls with quota-aware backoff; an LLM-as-judge critic scores every
> script against a shared rubric and sends weak ones back for a bounded rewrite; and an
> offline harness regression-tests that judge against a labelled dataset. Assets, render,
> and publish are still stubs. Next up is the human-approval interrupt — see
> [PLAN.md](PLAN.md) for the build order and [CLAUDE.md](CLAUDE.md) for the invariants.

---

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** for dependency and venv management
- **ffmpeg** on `PATH` — not needed until Phase 2 (rendering)

---

## Setup

### 1. Get `uv`

If you already have `uv` on your `PATH`, skip ahead.

This checkout keeps `uv` in its own throwaway venv at `.venv-tools/`, deliberately
separate from the project environment:

```powershell
python -m venv .venv-tools
.\.venv-tools\Scripts\python.exe -m pip install --upgrade pip uv
```

```bash
python -m venv .venv-tools
.venv-tools/bin/python -m pip install --upgrade pip uv
```

**Why separate?** `uv sync` prunes any package that isn't a project dependency. With
`uv` installed *into* `.venv/`, the first sync would uninstall its own binary. Keeping it
in `.venv-tools/` avoids that. Both directories are gitignored.

### 2. Install dependencies

`.venv/` is uv's default project environment, so no extra configuration is needed:

```powershell
.\.venv-tools\Scripts\uv.exe sync --all-groups
```

```bash
.venv-tools/bin/uv sync --all-groups        # or just: uv sync --all-groups
```

Exact versions are pinned in `uv.lock`.

### 3. Configure

```powershell
copy .env.example .env
```

```bash
cp .env.example .env
```

`.env.example` documents every variable. **An empty `.env` is valid** — the package
imports and the whole test suite runs with no credentials at all. Missing values fail at
the moment a provider needs them, with an error naming the variable.

Two things are worth setting before Phase 1b:

| Variable | Why |
| --- | --- |
| `OPENROUTER_API_KEY` | Auth for every LLM call. |
| `LLM_DRAFT_MODEL` / `LLM_JUDGE_MODEL` | **No defaults, on purpose.** OpenRouter retires `:free` model IDs without notice, so a hardcoded default is a time-bomb. Pick current IDs from [openrouter.ai/models](https://openrouter.ai/models?max_price=0). The judge must be distinctly stronger than the drafter. |

`PUBLISH_PROVIDER` defaults to `file`, which writes to `out/` and posts nothing. Real
posting is opt-in, and the approval gate applies either way.

---

## Checks

One entry point runs lint, format-check, and tests. Every step runs even when an earlier
one fails, so a single invocation surfaces every problem:

```powershell
.\.venv\Scripts\python.exe scripts\check.py          # lint + format-check + tests
.\.venv\Scripts\python.exe scripts\check.py --fix    # autofix lint + formatting first
```

```bash
.venv/bin/python scripts/check.py
uv run scripts/check.py      # either platform; note this re-syncs the project first
make check                   # unix convenience wrapper; forwards to the same script
```

Run this before considering any task done. `make` is optional — it is not installed on
the primary dev machine, and the `Makefile` only forwards to `scripts/check.py`.

Individual tools, if you want them directly:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

### Offline evals

The regression suite for prompt changes. It runs the **same** judged step the graph runs
against a labelled dataset of 22 scripts, and reports whether the judge still agrees with
the labels. Run it whenever you touch the judge prompt, the rubric criteria, the weights,
or the threshold — those changes look harmless in a diff and move behaviour a lot.

```powershell
.\.venv\Scripts\python.exe -m videoagent.evals.run_evals
.\.venv\Scripts\python.exe scripts\check.py --evals      # as part of the check task
```

Useful flags: `--limit N` (first N examples), `--json` (machine-readable), `--dataset PATH`,
`--min-agreement F` (exit non-zero below F; defaults to 0.75).

This costs **one judge-tier call per example** and needs `OPENROUTER_API_KEY` and
`LLM_JUDGE_MODEL`, which is why it is opt-in rather than part of every `check.py` run.
Without them it warns and skips rather than failing. `pytest` still validates the dataset
and the metrics arithmetic on every run against a stub judge — what `--evals` adds is the
real model's agreement.

Sample output:

```
example                               expect  judged   score  risk
pass-airplane-window                  pass    pass     8.40  no    ok
fail-no-payoff                        fail    fail     3.10  no    ok
risk-great-wall-from-space            fail    pass     9.00  no!   ✗
...
agreement              20/22 (90.9%)
factual-risk recall    83.3%  (a miss ships a falsehood)
factual-risk precision 100.0%  (a false alarm costs a rewrite)
mean score             4.72  (should-pass 8.12, should-fail 2.78)
discrimination         +5.35
```

**Reading the numbers.** *Agreement* is the headline. *Factual-risk recall* is the one to
be paranoid about — a miss means a confident falsehood ships, whereas a false alarm only
costs a rewrite. *Discrimination* (should-pass mean minus should-fail mean) catches what
agreement hides: a judge that has collapsed into scoring everything 7.5 can still post a
respectable agreement while having stopped distinguishing anything.

If results look noisy or arbitrary, suspect the judge model is too weak before you blame
the rubric. The judge runs once per video, so pointing `LLM_JUDGE_MODEL` at a cheap *paid*
model is the low-cost way to keep evals trustworthy.

---

## Layout

```
src/videoagent/
  graph/
    state.py          # the Pydantic state schema — single source of truth
    graph.py          # build_graph(): nodes, edges, conditional edges, checkpointer
    nodes/            # ideation, scriptwriter, eval_critic, assets, render,
                      # approval, publish, metrics
  providers/          # llm, tts, video, publish — Protocol + implementations
  memory/store.py     # embed + retrieve past videos and how they performed
  evals/
    rubric.py         # the scoring rubric, defined once
    run_evals.py      # offline regression suite
    dataset/          # labeled examples (JSONL)
  config.py           # Settings; selects providers/tiers from env
  api/main.py         # FastAPI; SSE stream of graph events
scripts/check.py      # the lint + test task
tests/
```

Most modules are docstring-only stubs; each states what it will hold and which phase
fills it in.

---

## Working on this

Read [CLAUDE.md](CLAUDE.md) first. Four invariants are non-negotiable:

1. **Every external call goes behind a provider interface.** A node that imports a vendor
   SDK or bakes in a model ID is a bug.
2. **Publishing is idempotent and never silently auto-posts.** Content-hash dedupe, a
   local-file default, and a human approval gate with no bypass.
3. **State is typed and checkpointed.** One Pydantic model through the whole graph;
   kill-and-resume must keep working.
4. **One shared eval rubric**, imported by both the inline critic and the offline harness.

---

## License

See [LICENSE](LICENSE).
