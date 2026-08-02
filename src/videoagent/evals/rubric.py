"""The scoring rubric — defined exactly once.

Imported by both the `eval_critic` node (inline, once per video) and the offline
`run_evals.py` harness (Phase 1d, over a labelled dataset). Duplicating any part of it
elsewhere is a bug (CLAUDE.md 4d): if the inline judge and the offline harness could
drift apart, the offline metrics would stop describing the thing actually running in
production, and the whole eval story collapses.

That is why this module owns more than a list of criteria — it owns the *prompt*, the
*parser*, and the *pass/fail rule* too. Those are the parts that would silently diverge.

Deliberately independent of `graph.state`: this module knows how to score a script, not
how the graph stores one. Scripts arrive as a structural `ScriptLike`, which keeps the
dependency pointing one way (state -> rubric) and lets the offline harness feed rows
straight from a JSONL dataset without constructing graph objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from videoagent.config import LLMTier
from videoagent.providers.llm import LLMResponse, LLMResponseError, extract_json_object

if TYPE_CHECKING:
    from videoagent.providers.llm import LLMProvider

__all__ = [
    "CRITERIA",
    "SCORE_MAX",
    "SCORE_MIN",
    "Criterion",
    "CriterionScore",
    "RubricScore",
    "ScriptLike",
    "build_messages",
    "parse_score",
    "score_script",
]

#: Scores run 0-10, matching the `EVAL_SCORE_THRESHOLD` setting's bounds.
SCORE_MIN: Final = 0.0
SCORE_MAX: Final = 10.0


class ScriptLike(Protocol):
    """Anything with the three parts of a script. Structural on purpose — see module docstring."""

    hook: str
    body: str
    cta: str


class Criterion(BaseModel):
    """One scored dimension, including the text the judge is shown."""

    key: str
    name: str
    guidance: str
    #: Relative contribution to the overall score. Weights across CRITERIA sum to 1.0.
    weight: float = Field(gt=0.0, le=1.0)


CRITERIA: Final[tuple[Criterion, ...]] = (
    Criterion(
        key="hook_strength",
        name="Hook strength",
        guidance=(
            "Does the first sentence earn the next three seconds? A 9-10 opens on the "
            "surprise itself. A 5 is competent but generic. A 1-2 is throat-clearing: a "
            "greeting, a channel plug, 'in this video', or a question with an obvious "
            "answer. Judge only the hook text, not the topic's inherent interest."
        ),
        # Weighted hardest: in short-form, a weak hook means nothing else gets watched.
        weight=0.4,
    ),
    Criterion(
        key="clarity",
        name="Spoken clarity",
        guidance=(
            "Read it aloud in your head. A 9-10 is short sentences a listener follows "
            "with no rewinding. Penalise subordinate clauses, unexplained jargon, "
            "number-dense phrasing, and anything that only parses on the page."
        ),
        weight=0.3,
    ),
    Criterion(
        key="payoff",
        name="Payoff",
        guidance=(
            "Does the body deliver the specific thing the hook promised? A 9-10 closes "
            "the loop concretely. A 1-3 changes the subject, restates the hook without "
            "explaining it, or promises a reveal that never arrives."
        ),
        weight=0.3,
    ),
)


class CriterionScore(BaseModel):
    """The judge's verdict on one criterion."""

    key: str
    score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)
    reason: str = ""


class RubricScore(BaseModel):
    """A complete judgment of one script."""

    criteria: list[CriterionScore]
    #: True when the script asserts something that could be wrong. A separate flag, not a
    #: score, because it is not a matter of degree — and because Phase 1d reports
    #: precision on it specifically.
    factual_risk: bool = False
    factual_risk_reason: str = ""

    @property
    def overall(self) -> float:
        """Weighted mean across the criteria, on the same 0-10 scale as the threshold."""
        weights = {criterion.key: criterion.weight for criterion in CRITERIA}
        total_weight = sum(weights.get(entry.key, 0.0) for entry in self.criteria)
        if total_weight == 0.0:
            return 0.0
        weighted = sum(entry.score * weights.get(entry.key, 0.0) for entry in self.criteria)
        return weighted / total_weight

    def passes(self, threshold: float) -> bool:
        """Whether this script may proceed.

        A factual-risk flag fails the script outright, whatever the craft scores say. A
        beautifully written falsehood is worse than a dull truth: the craft scores measure
        how effectively it would be believed.
        """
        return not self.factual_risk and self.overall >= threshold

    def summary(self) -> str:
        """One-line-per-criterion feedback, fed back into a scriptwriter retry."""
        lines = [f"{entry.key}: {entry.score:.1f}/10 — {entry.reason}" for entry in self.criteria]
        if self.factual_risk:
            lines.append(f"FACTUAL RISK: {self.factual_risk_reason}")
        return "\n".join(lines)


def _criteria_block() -> str:
    return "\n".join(
        f'- "{criterion.key}" ({criterion.name}, weight {criterion.weight:g}): {criterion.guidance}'
        for criterion in CRITERIA
    )


def _schema_block() -> str:
    fields = ",\n".join(
        f'  "{criterion.key}": {{"score": <0-10>, "reason": "<one short sentence>"}}'
        for criterion in CRITERIA
    )
    return (
        "{\n"
        f"{fields},\n"
        '  "factual_risk": <true|false>,\n'
        '  "factual_risk_reason": "<one short sentence, or empty string if false>"\n'
        "}"
    )


def build_system_prompt() -> str:
    """The judge's instructions. Derived from `CRITERIA`, so the two cannot drift."""
    return f"""You grade scripts for 30-45 second vertical short-form videos.

Score each criterion from 0 to 10. Use the full range: 5 is mediocre, not average-good.
Be a harsh grader — an inflated score here means a bad script ships.

Criteria:
{_criteria_block()}

Separately, set "factual_risk" to true if the script states anything that is wrong,
overstated, or that you cannot verify. Judge the claims, not the writing. A confident
tone is not evidence. If in doubt, set it to true.

Reply with exactly this JSON object and nothing else:
{_schema_block()}"""


def build_messages(topic: str, script: ScriptLike) -> list[SystemMessage | HumanMessage]:
    """The full judge prompt for one script. Shared by the node and the harness."""
    return [
        SystemMessage(build_system_prompt()),
        HumanMessage(
            f"Topic: {topic}\n\nHOOK:\n{script.hook}\n\nBODY:\n{script.body}\n\nCTA:\n{script.cta}"
        ),
    ]


def parse_score(text: str) -> RubricScore:
    """Turn a judge reply into a `RubricScore`, or raise `LLMResponseError`.

    Out-of-range scores are rejected rather than clamped. A model answering on a 0-100
    scale would clamp to a perfect 10 and wave a bad script straight through — silently
    inverting the meaning of the result. Failing loudly is the safe direction.
    """
    payload = extract_json_object(text)
    scores: list[CriterionScore] = []

    for criterion in CRITERIA:
        entry = payload.get(criterion.key)
        if entry is None:
            raise LLMResponseError(f"Judge omitted criterion {criterion.key!r}: {payload!r}")
        raw = entry.get("score") if isinstance(entry, dict) else entry
        try:
            score = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise LLMResponseError(
                f"Judge gave a non-numeric score for {criterion.key!r}: {raw!r}"
            ) from error
        if not SCORE_MIN <= score <= SCORE_MAX:
            raise LLMResponseError(
                f"Judge scored {criterion.key!r} at {score}, outside {SCORE_MIN}-{SCORE_MAX}. "
                "Refusing to guess at the intended scale."
            )
        reason = entry.get("reason", "") if isinstance(entry, dict) else ""
        scores.append(CriterionScore(key=criterion.key, score=score, reason=str(reason).strip()))

    return RubricScore(
        criteria=scores,
        factual_risk=_as_bool(payload.get("factual_risk", False)),
        factual_risk_reason=str(payload.get("factual_risk_reason", "") or "").strip(),
    )


def _as_bool(value: Any) -> bool:
    """Accept the several ways a model spells a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}
    return bool(value)


async def score_script(
    llm: LLMProvider, *, topic: str, script: ScriptLike
) -> tuple[RubricScore, LLMResponse]:
    """Judge one script on the judge tier.

    The single entry point used by both the inline critic and the offline harness, so
    prompt, tier, sampling, and parsing are identical in both. Returns the response too,
    so callers can record what the judgment cost.
    """
    response = await llm.complete(
        build_messages(topic, script),
        tier=LLMTier.JUDGE,
        # Grading should be as close to deterministic as the model allows: the same
        # script scored twice should not straddle the threshold.
        temperature=0.0,
        max_tokens=800,
    )
    return parse_score(response.text), response
