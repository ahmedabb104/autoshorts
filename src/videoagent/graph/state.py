"""The graph state schema — the single source of truth for everything the graph carries.

Every node takes this model in and returns a partial update of it; no untyped dicts flow
through the graph (CLAUDE.md 4c). LangGraph merges each node's returned dict into the
model and hands the result to the checkpointer, so this file also defines *how* updates
combine: fields annotated with a reducer (`operator.add`) accumulate across nodes instead
of being overwritten, which is what makes `completed_nodes` and `costs` honest histories
rather than last-writer-wins.

Two accumulating fields exist on purpose:

* `completed_nodes` — an append-only execution trace. It is what proves, after a resumed
  run, that earlier nodes did not re-execute.
* `costs` — an append-only ledger. CLAUDE.md section 7 wants a truthful "cost per video"
  number, which means every provider call appends an entry rather than overwriting a
  running total.
"""

from __future__ import annotations

import hashlib
import operator
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final
from uuid import uuid4

from pydantic import BaseModel, Field

__all__ = [
    "CHECKPOINTED_TYPES",
    "CostEntry",
    "RunStatus",
    "Script",
    "VideoState",
]


class RunStatus(StrEnum):
    """Where a run has got to.

    Ordered roughly by progression, but this is a label rather than a state machine —
    the graph's edges decide what runs next, not this enum.
    """

    PENDING = "pending"
    IDEATED = "ideated"
    DRAFTED = "drafted"
    EVALUATED = "evaluated"
    ASSETS_READY = "assets_ready"
    RENDERED = "rendered"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"
    FAILED = "failed"


class Script(BaseModel):
    """The three parts of a short-form script.

    Kept as separate fields rather than one blob because the rubric scores them
    differently — hook strength is the single highest-leverage criterion in short-form,
    and the eval critic needs to address it on its own.
    """

    hook: str
    body: str
    cta: str

    @property
    def full_text(self) -> str:
        """The script as one narration-ready string."""
        return f"{self.hook}\n\n{self.body}\n\n{self.cta}"

    @property
    def word_count(self) -> int:
        """Rough length signal — a Short's narration has to fit in under a minute."""
        return len(self.full_text.split())


class CostEntry(BaseModel):
    """One charge incurred during a run.

    Appended, never overwritten, so a retry loop's true cost stays visible instead of
    being masked by a final total.
    """

    node: str = Field(description="Graph node that incurred the cost.")
    provider: str = Field(description="Provider that charged, e.g. 'openrouter', 'elevenlabs'.")
    usd: float = Field(ge=0.0, description="Amount in USD. Zero for free-tier calls.")
    detail: str = Field(default="", description="Model ID, character count, or similar.")


class VideoState(BaseModel):
    """Everything one video-generation run carries from START to END.

    Every field has a default so the graph can be invoked with a partial input (even
    `{}`); LangGraph validates the merged model after each node.
    """

    # --- Identity -------------------------------------------------------------------
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    status: RunStatus = RunStatus.PENDING

    # --- Ideation -------------------------------------------------------------------
    topic: str | None = None

    # --- Scripting ------------------------------------------------------------------
    script: Script | None = None
    #: Bounded by MAX_SCRIPT_RETRIES. The conditional retry edge lands in Phase 1c.
    retry_count: int = 0

    # --- Evaluation -----------------------------------------------------------------
    #: Rubric score, 0-10, compared against EVAL_SCORE_THRESHOLD.
    #: Phase 1c replaces this scalar with the structured score from `evals.rubric`.
    eval_score: float | None = None
    eval_notes: str | None = None

    # --- Assets and render ------------------------------------------------------------
    voiceover_path: Path | None = None
    clip_paths: list[Path] = Field(default_factory=list)
    video_path: Path | None = None

    # --- Human approval ---------------------------------------------------------------
    #: `None` means "not yet decided" and is distinct from `False` ("a human rejected
    #: this"). Publish refuses to act unless this is explicitly `True` (CLAUDE.md 4b).
    approved: bool | None = None
    approval_note: str | None = None

    # --- Publish ------------------------------------------------------------------------
    content_hash: str | None = None
    publish_receipt: str | None = None

    # --- Accumulators (reducer-backed: nodes append, they do not overwrite) -----------
    completed_nodes: Annotated[list[str], operator.add] = Field(default_factory=list)
    costs: Annotated[list[CostEntry], operator.add] = Field(default_factory=list)

    # --- Failure ------------------------------------------------------------------------
    error: str | None = None

    @property
    def total_cost_usd(self) -> float:
        """Sum of the cost ledger. A property, not a field, so it can never drift."""
        return sum(entry.usd for entry in self.costs)

    def content_fingerprint(self) -> str:
        """Stable content hash used to make publishing idempotent (CLAUDE.md 4b).

        Derived from what actually defines the video — its topic and script — so that a
        retry or a resumed run recomputes the *same* hash and the publish provider can
        recognise it as already published. Deliberately excludes `run_id`, timestamps,
        and file paths, all of which change between attempts at identical content.
        """
        parts = [self.topic or "", self.script.full_text if self.script else ""]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


#: Every custom type that can end up inside a checkpoint.
#:
#: LangGraph's msgpack serializer keeps an allowlist of types it will reconstruct when
#: reading a checkpoint back. Types outside it are currently deserialized with a warning
#: and will be *blocked* in a future release — which would silently break the resume
#: guarantee in CLAUDE.md 4c. Registering them explicitly (see `graph.build_serde`) means
#: the allowlist tightening is a non-event for us.
#:
#: Anything new that lands in `VideoState` and is not a plain builtin belongs here.
CHECKPOINTED_TYPES: Final = (VideoState, Script, CostEntry, RunStatus)
