"""Phase 1c tests for the shared rubric.

The rubric is imported by both the inline `eval_critic` node and the offline harness
(CLAUDE.md 4d), so a change here moves production behaviour and offline metrics together.
These tests pin the parts that would be quiet if they broke: the weighting, the pass rule,
and the parser's refusal to guess.
"""

from __future__ import annotations

import json

import pytest

from videoagent.config import LLMTier
from videoagent.evals.rubric import (
    CRITERIA,
    CriterionScore,
    RubricScore,
    build_messages,
    build_system_prompt,
    parse_score,
    score_script,
)
from videoagent.graph.state import Script
from videoagent.providers.llm import LLMResponseError

from .conftest import FakeLLM, rubric_reply

SCRIPT = Script(hook="a hook", body="a body", cta="a cta")


def score_of(**per_criterion: float) -> RubricScore:
    return RubricScore(
        criteria=[CriterionScore(key=key, score=value) for key, value in per_criterion.items()]
    )


# --------------------------------------------------------------------------------------
# The rubric definition itself
# --------------------------------------------------------------------------------------


def test_criteria_weights_sum_to_one() -> None:
    """Otherwise `overall` silently stops being on the 0-10 scale the threshold assumes."""
    assert sum(criterion.weight for criterion in CRITERIA) == pytest.approx(1.0)


def test_criteria_keys_are_unique() -> None:
    keys = [criterion.key for criterion in CRITERIA]
    assert len(keys) == len(set(keys))


def test_the_prompt_is_generated_from_the_criteria() -> None:
    """The prompt and the criteria cannot drift, because one is built from the other."""
    prompt = build_system_prompt()
    for criterion in CRITERIA:
        assert criterion.key in prompt
        assert criterion.guidance in prompt
    assert "factual_risk" in prompt


def test_build_messages_shows_the_judge_all_three_script_parts() -> None:
    system, human = build_messages("a topic", SCRIPT)
    assert "grade scripts" in str(system.content)
    body = str(human.content)
    assert "a topic" in body
    for part in ("a hook", "a body", "a cta"):
        assert part in body


# --------------------------------------------------------------------------------------
# Scoring arithmetic and the pass rule
# --------------------------------------------------------------------------------------


def test_overall_is_weighted_not_a_flat_mean() -> None:
    """Hook strength carries 0.4 — a weak hook must not be averaged away."""
    weak_hook = score_of(hook_strength=0.0, clarity=10.0, payoff=10.0)
    weak_payoff = score_of(hook_strength=10.0, clarity=10.0, payoff=0.0)

    assert weak_hook.overall == pytest.approx(6.0)
    assert weak_payoff.overall == pytest.approx(7.0)
    assert weak_hook.overall < weak_payoff.overall


def test_overall_of_an_empty_score_is_zero_not_a_crash() -> None:
    assert RubricScore(criteria=[]).overall == 0.0


@pytest.mark.parametrize(
    ("scores", "threshold", "expected"),
    [
        ({"hook_strength": 7.0, "clarity": 7.0, "payoff": 7.0}, 7.0, True),
        ({"hook_strength": 6.9, "clarity": 6.9, "payoff": 6.9}, 7.0, False),
        ({"hook_strength": 10.0, "clarity": 10.0, "payoff": 10.0}, 7.0, True),
    ],
    ids=["exactly-at-threshold", "just-below", "well-above"],
)
def test_passes_compares_against_the_threshold(
    scores: dict[str, float], threshold: float, expected: bool
) -> None:
    assert score_of(**scores).passes(threshold) is expected


def test_a_factual_risk_flag_fails_a_perfect_script() -> None:
    """A beautifully written falsehood is worse than a dull truth."""
    flagged = RubricScore(
        criteria=[CriterionScore(key=c.key, score=10.0) for c in CRITERIA],
        factual_risk=True,
        factual_risk_reason="the date is wrong",
    )
    assert flagged.overall == pytest.approx(10.0)
    assert flagged.passes(7.0) is False


def test_summary_is_feedback_a_retry_can_act_on() -> None:
    flagged = RubricScore(
        criteria=[CriterionScore(key="hook_strength", score=3.0, reason="opens on a greeting")],
        factual_risk=True,
        factual_risk_reason="the 1969 date is wrong",
    )
    summary = flagged.summary()
    assert "hook_strength: 3.0/10 — opens on a greeting" in summary
    assert "FACTUAL RISK: the 1969 date is wrong" in summary


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def test_parse_score_reads_a_well_formed_reply() -> None:
    parsed = parse_score(rubric_reply(hook_strength=8.0, clarity=6.0, payoff=7.0))
    assert parsed.overall == pytest.approx(0.4 * 8 + 0.3 * 6 + 0.3 * 7)
    assert parsed.factual_risk is False


def test_parse_score_tolerates_fenced_json() -> None:
    assert parse_score(f"```json\n{rubric_reply()}\n```").overall == pytest.approx(9.0)


@pytest.mark.parametrize(
    "raw_flag",
    [True, "true", "True", "yes", 1],
    ids=["bool", "lower", "capitalised", "yes", "one"],
)
def test_parse_score_accepts_the_several_ways_a_model_spells_true(raw_flag: object) -> None:
    payload = json.loads(rubric_reply())
    payload["factual_risk"] = raw_flag
    assert parse_score(json.dumps(payload)).factual_risk is True


def test_parse_score_accepts_a_bare_number_for_a_criterion() -> None:
    """Some models drop the {"score": ...} wrapper. Recoverable, so recover."""
    payload = {criterion.key: 7.0 for criterion in CRITERIA}
    payload["factual_risk"] = False  # type: ignore[assignment]
    assert parse_score(json.dumps(payload)).overall == pytest.approx(7.0)


def test_parse_score_rejects_a_missing_criterion() -> None:
    payload = json.loads(rubric_reply())
    del payload[CRITERIA[0].key]
    with pytest.raises(LLMResponseError, match=CRITERIA[0].key):
        parse_score(json.dumps(payload))


def test_parse_score_rejects_an_out_of_range_score_rather_than_clamping() -> None:
    """A judge answering 0-100 would clamp to a perfect 10 and wave a bad script through."""
    payload = json.loads(rubric_reply())
    payload[CRITERIA[0].key]["score"] = 85
    with pytest.raises(LLMResponseError, match="outside"):
        parse_score(json.dumps(payload))


def test_parse_score_rejects_a_non_numeric_score() -> None:
    payload = json.loads(rubric_reply())
    payload[CRITERIA[0].key]["score"] = "very good"
    with pytest.raises(LLMResponseError, match="non-numeric"):
        parse_score(json.dumps(payload))


def test_parse_score_rejects_prose() -> None:
    with pytest.raises(LLMResponseError):
        parse_score("I think this script is pretty good, maybe an 8?")


# --------------------------------------------------------------------------------------
# The shared entry point
# --------------------------------------------------------------------------------------


async def test_score_script_uses_the_judge_tier_at_zero_temperature() -> None:
    """The same script scored twice must not straddle the threshold on sampling noise."""
    llm = FakeLLM()
    parsed, response = await score_script(llm, topic="t", script=SCRIPT)

    call = llm.calls[0]
    assert call.tier is LLMTier.JUDGE
    assert call.temperature == 0.0
    assert parsed.overall == pytest.approx(9.0)
    assert response.model == "fake/judge-model"
