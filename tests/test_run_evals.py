"""Phase 1d tests for the offline eval harness.

Two jobs, and they are different:

1. **Validate the dataset.** A regression suite is worth exactly as much as its labels. A
   duplicated id, an unparseable row, or a set of examples that are all easy would make
   the harness report a comfortable number while measuring nothing.
2. **Validate the harness.** The metrics arithmetic, the error handling, and the
   agreement tripwire, driven by a stub judge with known behaviour.

What these tests deliberately do NOT do is call a real model. Twenty-odd judge calls per
test run would burn the free tier's daily quota within a few commits; the real run is
`python -m videoagent.evals.run_evals` (or `scripts/check.py --evals`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from langchain_core.messages import BaseMessage

from videoagent.config import LLMTier
from videoagent.evals.rubric import CRITERIA, RubricScore
from videoagent.evals.run_evals import (
    DEFAULT_DATASET,
    EvalExample,
    EvalReport,
    ExampleResult,
    format_report,
    load_dataset,
    run_dataset,
)
from videoagent.providers.llm import LLMError, LLMResponse, LLMUsage

from .conftest import rubric_reply

THRESHOLD = 7.0


@pytest.fixture(scope="module")
def dataset() -> list[EvalExample]:
    return load_dataset()


# --------------------------------------------------------------------------------------
# The dataset is an artifact in its own right
# --------------------------------------------------------------------------------------


def test_the_shipped_dataset_loads(dataset: list[EvalExample]) -> None:
    assert DEFAULT_DATASET.exists()
    assert len(dataset) >= 15, "PLAN.md asks for 15-25 labelled examples"
    assert len(dataset) <= 30


def test_every_example_is_fully_populated(dataset: list[EvalExample]) -> None:
    for example in dataset:
        assert example.topic.strip()
        assert example.hook.strip()
        assert example.body.strip()
        assert example.cta.strip()
        # The note is what makes a disagreement debuggable rather than mysterious.
        assert example.note.strip(), f"{example.id} has no note explaining its label"


def test_the_dataset_is_not_trivially_easy(dataset: list[EvalExample]) -> None:
    """Both classes must be well represented, or agreement is a meaningless number.

    A dataset that is 90% fails lets a judge that rejects everything score 90%.
    """
    passes = sum(example.expected_pass for example in dataset)
    fails = len(dataset) - passes
    assert passes >= 5, f"only {passes} should-pass examples"
    assert fails >= 5, f"only {fails} should-fail examples"
    assert 0.25 <= passes / len(dataset) <= 0.6


def test_the_dataset_exercises_the_factual_risk_flag(dataset: list[EvalExample]) -> None:
    """Recall on this flag is the metric worth being paranoid about, so it needs examples."""
    risky = [example for example in dataset if example.expected_factual_risk]
    assert len(risky) >= 4
    # Every risky script must also be labelled a fail — a factual risk fails outright.
    for example in risky:
        assert not example.expected_pass, f"{example.id} is risky but labelled a pass"


def test_the_dataset_separates_craft_failures_from_factual_ones(
    dataset: list[EvalExample],
) -> None:
    """Otherwise 'fail' and 'risky' are the same column and one of them is untested."""
    craft_failures = [
        example
        for example in dataset
        if not example.expected_pass and not example.expected_factual_risk
    ]
    assert len(craft_failures) >= 4


def test_examples_satisfy_the_rubrics_script_protocol(dataset: list[EvalExample]) -> None:
    """`EvalExample` feeds straight into the same scorer the graph uses (CLAUDE.md 4d)."""
    from videoagent.evals.rubric import build_messages

    _system, human = build_messages(dataset[0].topic, dataset[0])
    assert dataset[0].hook in str(human.content)


def test_load_dataset_rejects_a_duplicate_id(tmp_path: Path) -> None:
    row = {
        "id": "same",
        "topic": "t",
        "hook": "h",
        "body": "b",
        "cta": "c",
        "expected_pass": True,
        "expected_factual_risk": False,
    }
    path = tmp_path / "dupes.jsonl"
    path.write_text(f"{json.dumps(row)}\n{json.dumps(row)}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicates id"):
        load_dataset(path)


def test_load_dataset_rejects_a_malformed_row_rather_than_skipping_it(tmp_path: Path) -> None:
    """A silently skipped row would quietly shrink the suite."""
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id": "x", "topic": "t"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="not a valid example"):
        load_dataset(path)


def test_load_dataset_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="no examples"):
        load_dataset(path)


# --------------------------------------------------------------------------------------
# A stub judge with known behaviour
# --------------------------------------------------------------------------------------


class ScriptedJudge:
    """Answers from a per-example-id script, so metrics are exactly predictable."""

    def __init__(
        self,
        replies: dict[str, str],
        *,
        default: str | None = None,
        cost_usd: float = 0.0,
    ) -> None:
        self._replies = replies
        self._default = default or rubric_reply()
        self._cost = cost_usd
        self.calls = 0
        self.max_in_flight = 0
        self._in_flight = 0

    async def complete(
        self,
        messages: Sequence[BaseMessage],
        *,
        tier: LLMTier,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls += 1
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            body = str(messages[-1].content)
            for example_id, reply in self._replies.items():
                if example_id in body:
                    return self._response(reply)
            return self._response(self._default)
        finally:
            self._in_flight -= 1

    def _response(self, text: str) -> LLMResponse:
        if text == "__raise__":
            raise LLMError("judge unavailable")
        return LLMResponse(text=text, model="stub/judge", usage=LLMUsage(cost_usd=self._cost))


def perfect_replies(examples: Sequence[EvalExample]) -> dict[str, str]:
    """A judge that agrees with every label. Keyed by hook, which appears in the prompt."""
    return {
        example.hook: rubric_reply(
            hook_strength=9.0 if example.expected_pass else 2.0,
            clarity=9.0 if example.expected_pass else 3.0,
            payoff=9.0 if example.expected_pass else 2.0,
            factual_risk=example.expected_factual_risk,
        )
        for example in examples
    }


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


async def test_a_perfect_judge_scores_perfectly(dataset: list[EvalExample]) -> None:
    judge = ScriptedJudge(perfect_replies(dataset))
    report = await run_dataset(judge, dataset, threshold=THRESHOLD, judge_model="stub/judge")

    assert judge.calls == len(dataset)
    assert report.agreement == 1.0
    assert report.risk_recall == 1.0
    assert report.risk_precision == 1.0
    assert report.discrimination > 0
    assert report.errors == []


async def test_a_judge_that_passes_everything_is_caught(dataset: list[EvalExample]) -> None:
    """The failure agreement alone would hide if the dataset were unbalanced."""
    judge = ScriptedJudge({}, default=rubric_reply())
    report = await run_dataset(judge, dataset, threshold=THRESHOLD)

    expected_passes = sum(example.expected_pass for example in dataset)
    assert report.agreement == pytest.approx(expected_passes / len(dataset))
    assert report.risk_recall == 0.0
    # Everything scored identically, so the classes are indistinguishable.
    assert report.discrimination == pytest.approx(0.0)


async def test_a_collapsed_judge_shows_zero_discrimination(
    dataset: list[EvalExample],
) -> None:
    """Scoring everything 7.5 can look fine on agreement while measuring nothing."""
    judge = ScriptedJudge({}, default=rubric_reply(hook_strength=7.5, clarity=7.5, payoff=7.5))
    report = await run_dataset(judge, dataset, threshold=THRESHOLD)

    assert report.mean_score == pytest.approx(7.5)
    assert report.discrimination == pytest.approx(0.0)


async def test_an_unparseable_reply_counts_as_disagreement_not_as_neutral(
    dataset: list[EvalExample],
) -> None:
    replies = perfect_replies(dataset)
    replies[dataset[0].hook] = "the judge waffled instead of answering"
    report = await run_dataset(ScriptedJudge(replies), dataset, threshold=THRESHOLD)

    assert len(report.errors) == 1
    assert report.agreement == pytest.approx((len(dataset) - 1) / len(dataset))


async def test_a_judge_outage_is_recorded_not_raised(dataset: list[EvalExample]) -> None:
    """One dead call must not abort the whole run and lose the other results."""
    replies = perfect_replies(dataset)
    replies[dataset[0].hook] = "__raise__"
    report = await run_dataset(ScriptedJudge(replies), dataset, threshold=THRESHOLD)

    assert len(report.results) == len(dataset)
    assert report.errors[0].error == "judge unavailable"


async def test_concurrency_is_bounded_for_the_rate_limit(dataset: list[EvalExample]) -> None:
    """20 requests/minute on the free tier — fanning out the whole dataset would trip it."""
    judge = ScriptedJudge(perfect_replies(dataset))
    await run_dataset(judge, dataset, threshold=THRESHOLD, concurrency=3)
    assert judge.max_in_flight <= 3


async def test_cost_is_totalled_across_the_run(dataset: list[EvalExample]) -> None:
    judge = ScriptedJudge(perfect_replies(dataset), cost_usd=0.001)
    report = await run_dataset(judge, dataset, threshold=THRESHOLD)
    assert report.total_cost_usd == pytest.approx(0.001 * len(dataset))


def test_precision_and_recall_differ_where_it_matters() -> None:
    """Hand-built confusion matrix: 2 true positives, 1 false positive, 1 false negative."""

    def result(*, expected_risk: bool, judged_risk: bool) -> ExampleResult:
        example = EvalExample(
            id=f"{expected_risk}-{judged_risk}",
            topic="t",
            hook="h",
            body="b",
            cta="c",
            expected_pass=not expected_risk,
            expected_factual_risk=expected_risk,
        )
        score = RubricScore(criteria=[], factual_risk=judged_risk)  # criteria empty: overall is 0.0
        return ExampleResult(example=example, score=score, predicted_pass=False)

    report = EvalReport(
        results=[
            result(expected_risk=True, judged_risk=True),
            result(expected_risk=True, judged_risk=True),
            result(expected_risk=False, judged_risk=True),
            result(expected_risk=True, judged_risk=False),
        ]
    )
    assert report.risk_precision == pytest.approx(2 / 3)
    assert report.risk_recall == pytest.approx(2 / 3)


def test_empty_report_metrics_do_not_divide_by_zero() -> None:
    empty = EvalReport()
    assert empty.agreement == 0.0
    assert empty.risk_precision == 0.0
    assert empty.risk_recall == 0.0
    assert empty.mean_score == 0.0
    assert empty.discrimination == 0.0


# --------------------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------------------


async def test_the_report_names_every_example_and_the_headline_metrics(
    dataset: list[EvalExample],
) -> None:
    judge = ScriptedJudge(perfect_replies(dataset))
    report = await run_dataset(judge, dataset, threshold=THRESHOLD, judge_model="stub/judge")
    rendered = format_report(report)

    for example in dataset:
        assert example.id in rendered
    assert "agreement" in rendered
    assert "factual-risk recall" in rendered
    assert "discrimination" in rendered
    assert "stub/judge" in rendered


async def test_a_disagreement_is_visibly_marked(dataset: list[EvalExample]) -> None:
    replies = perfect_replies(dataset)
    wrong = next(example for example in dataset if example.expected_pass)
    replies[wrong.hook] = rubric_reply(hook_strength=1.0, clarity=1.0, payoff=1.0)

    report = await run_dataset(ScriptedJudge(replies), dataset, threshold=THRESHOLD)
    line = next(row for row in format_report(report).splitlines() if row.startswith(wrong.id))

    assert "✗" in line
    assert report.agreement < 1.0


async def test_the_json_summary_is_machine_readable(dataset: list[EvalExample]) -> None:
    judge = ScriptedJudge(perfect_replies(dataset))
    report = await run_dataset(judge, dataset, threshold=THRESHOLD, judge_model="stub/judge")

    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["agreement"] == 1.0
    assert payload["examples"] == len(dataset)
    assert payload["judge_model"] == "stub/judge"


def test_the_rubric_criteria_and_the_stub_replies_stay_in_step() -> None:
    """If a criterion is added, `rubric_reply` must be updated or these tests lie."""
    payload = json.loads(rubric_reply())
    for criterion in CRITERIA:
        assert criterion.key in payload
