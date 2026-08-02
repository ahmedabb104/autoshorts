"""Offline eval harness — the regression suite for prompt and rubric changes.

Runs the *same* judged step the graph runs (`rubric.score_script`) over a labelled
dataset, and reports whether the judge still agrees with the labels. Run it whenever you
touch the judge prompt, the criteria, the weights, or the threshold — those changes look
harmless in a diff and move behaviour a lot.

What the numbers mean, in rough order of how much they should worry you:

* **Agreement** — how often the judge's pass/fail matches the label. The headline number.
* **Factual-risk recall** — of the scripts that really do contain a false claim, how many
  did the judge flag? A miss here means a confident falsehood ships. This is the metric
  worth being paranoid about; precision matters less, because a false alarm only costs a
  rewrite.
* **Discrimination** — mean score of should-pass examples minus mean score of should-fail
  ones. It catches the failure agreement alone hides: a judge that has collapsed into
  scoring everything 7.5 can still look respectable on agreement while having stopped
  distinguishing anything.
* **Errors** — replies that could not be parsed at all.

Two ways to run it::

    # the real thing: hits LLM_JUDGE_MODEL, needs OPENROUTER_API_KEY
    python -m videoagent.evals.run_evals

    # and as part of the check task
    python scripts/check.py --evals

`pytest` deliberately does *not* invoke the real judge — see `tests/test_run_evals.py`,
which exercises this module against a stub. Twenty-odd judge calls on every test run would
burn the free tier's daily quota within a few commits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError

from videoagent.config import LLMTier, Settings, get_settings
from videoagent.evals.rubric import RubricScore, score_script
from videoagent.providers.llm import LLMError, LLMProvider, open_llm_provider

__all__ = [
    "DEFAULT_DATASET",
    "EvalExample",
    "EvalReport",
    "ExampleResult",
    "format_report",
    "load_dataset",
    "run_dataset",
]

DEFAULT_DATASET = Path(__file__).parent / "dataset" / "scripts.jsonl"

#: The free tier allows 20 requests/minute. Fanning the whole dataset out at once would
#: trip that immediately and spend the run in backoff, so keep a small window open.
DEFAULT_CONCURRENCY = 4

#: Below this, treat the judge as regressed. Not a target to tune against — a tripwire.
DEFAULT_MIN_AGREEMENT = 0.75


class EvalExample(BaseModel):
    """One labelled script.

    Structurally satisfies `rubric.ScriptLike` (it has `hook`, `body`, `cta`), which is
    why the harness can feed dataset rows straight to the same scoring function the graph
    uses without constructing any graph objects.
    """

    id: str
    topic: str
    hook: str
    body: str
    cta: str
    #: Whether a correct judge should let this script through.
    expected_pass: bool
    #: Whether this script contains a claim that is wrong or unverifiable.
    expected_factual_risk: bool
    #: Why it is labelled this way. For humans reading a failure, not used in scoring.
    note: str = ""


@dataclass
class ExampleResult:
    """What the judge did with one example."""

    example: EvalExample
    score: RubricScore | None = None
    error: str | None = None
    cost_usd: float = 0.0
    #: `None` when the judge failed to produce a usable score. Decided by `run_dataset`,
    #: which is the layer that knows the configured threshold.
    predicted_pass: bool | None = None

    @property
    def agreed(self) -> bool:
        """Unparseable replies count as disagreement, not as neutral."""
        return self.predicted_pass is not None and (
            self.predicted_pass == self.example.expected_pass
        )

    @property
    def risk_agreed(self) -> bool:
        return (
            self.score is not None and self.score.factual_risk == self.example.expected_factual_risk
        )


@dataclass
class EvalReport:
    """Aggregate metrics over a dataset run."""

    results: list[ExampleResult] = field(default_factory=list)
    threshold: float = 7.0
    judge_model: str = "unknown"

    @property
    def scored(self) -> list[ExampleResult]:
        return [result for result in self.results if result.score is not None]

    @property
    def errors(self) -> list[ExampleResult]:
        return [result for result in self.results if result.score is None]

    @property
    def agreement(self) -> float:
        """Fraction of examples where pass/fail matched the label.

        Unparseable replies count as disagreements. A judge that cannot answer is not
        neutral — it has failed at the job.
        """
        if not self.results:
            return 0.0
        return sum(result.agreed for result in self.results) / len(self.results)

    @property
    def risk_precision(self) -> float:
        """Of the scripts flagged as risky, how many really were."""
        flagged = [r for r in self.scored if r.score and r.score.factual_risk]
        if not flagged:
            return 0.0
        return sum(r.example.expected_factual_risk for r in flagged) / len(flagged)

    @property
    def risk_recall(self) -> float:
        """Of the scripts that really are risky, how many were caught.

        The metric to be paranoid about: a miss means a confident falsehood ships.
        """
        risky = [r for r in self.scored if r.example.expected_factual_risk]
        if not risky:
            return 0.0
        return sum(bool(r.score and r.score.factual_risk) for r in risky) / len(risky)

    @property
    def mean_score(self) -> float:
        if not self.scored:
            return 0.0
        return sum(r.score.overall for r in self.scored if r.score) / len(self.scored)

    def _mean_score_where(self, expected_pass: bool) -> float:
        subset = [r for r in self.scored if r.example.expected_pass is expected_pass and r.score]
        if not subset:
            return 0.0
        return sum(r.score.overall for r in subset if r.score) / len(subset)

    @property
    def mean_score_should_pass(self) -> float:
        return self._mean_score_where(True)

    @property
    def mean_score_should_fail(self) -> float:
        return self._mean_score_where(False)

    @property
    def discrimination(self) -> float:
        """Gap between the two classes' mean scores.

        Catches what agreement hides: a judge that scores everything 7.5 can still look
        acceptable on agreement while having stopped distinguishing good from bad.
        """
        return self.mean_score_should_pass - self.mean_score_should_fail

    @property
    def total_cost_usd(self) -> float:
        return sum(result.cost_usd for result in self.results)

    def as_dict(self) -> dict[str, object]:
        """Machine-readable summary, for piping somewhere or diffing across runs."""
        return {
            "judge_model": self.judge_model,
            "threshold": self.threshold,
            "examples": len(self.results),
            "errors": len(self.errors),
            "agreement": round(self.agreement, 4),
            "risk_precision": round(self.risk_precision, 4),
            "risk_recall": round(self.risk_recall, 4),
            "mean_score": round(self.mean_score, 3),
            "mean_score_should_pass": round(self.mean_score_should_pass, 3),
            "mean_score_should_fail": round(self.mean_score_should_fail, 3),
            "discrimination": round(self.discrimination, 3),
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


def load_dataset(path: Path = DEFAULT_DATASET) -> list[EvalExample]:
    """Read the JSONL dataset, failing loudly on a bad row.

    A silently skipped row would quietly shrink the suite, which is the one failure mode a
    regression harness must not have.
    """
    examples: list[EvalExample] = []
    seen: set[str] = set()

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            example = EvalExample.model_validate_json(line)
        except ValidationError as error:
            raise ValueError(f"{path.name}:{number} is not a valid example: {error}") from error
        if example.id in seen:
            raise ValueError(f"{path.name}:{number} duplicates id {example.id!r}")
        seen.add(example.id)
        examples.append(example)

    if not examples:
        raise ValueError(f"{path} contains no examples")
    return examples


async def run_dataset(
    llm: LLMProvider,
    examples: Sequence[EvalExample],
    *,
    threshold: float,
    concurrency: int = DEFAULT_CONCURRENCY,
    judge_model: str = "unknown",
) -> EvalReport:
    """Score every example, bounded by `concurrency`, and aggregate the results."""
    limiter = asyncio.Semaphore(concurrency)

    async def judge(example: EvalExample) -> ExampleResult:
        async with limiter:
            try:
                score, response = await score_script(llm, topic=example.topic, script=example)
            except LLMError as error:
                return ExampleResult(example=example, error=str(error))
            return ExampleResult(
                example=example,
                score=score,
                cost_usd=response.usage.cost_usd,
                predicted_pass=score.passes(threshold),
            )

    results = await asyncio.gather(*(judge(example) for example in examples))
    return EvalReport(results=list(results), threshold=threshold, judge_model=judge_model)


def format_report(report: EvalReport) -> str:
    """The human-facing summary table."""
    lines: list[str] = []
    width = max((len(r.example.id) for r in report.results), default=10)

    lines.append(f"{'example':<{width}}  expect  judged   score  risk  ")
    lines.append("-" * (width + 32))

    for result in report.results:
        expected = "pass" if result.example.expected_pass else "fail"
        if result.score is None:
            lines.append(f"{result.example.id:<{width}}  {expected:<6}  ERROR    ----  ----  ✗")
            continue
        judged = "pass" if result.predicted_pass else "fail"
        risk = "yes" if result.score.factual_risk else "no"
        expected_risk = "yes" if result.example.expected_factual_risk else "no"
        risk_cell = risk if risk == expected_risk else f"{risk}!"
        mark = "ok" if result.agreed else "✗"
        lines.append(
            f"{result.example.id:<{width}}  {expected:<6}  {judged:<6}  "
            f"{result.score.overall:5.2f}  {risk_cell:<4}  {mark}"
        )

    agreed = sum(result.agreed for result in report.results)
    lines.append("")
    lines.append(f"judge model            {report.judge_model}")
    lines.append(f"pass threshold         {report.threshold:.1f}")
    lines.append(f"agreement              {agreed}/{len(report.results)} ({report.agreement:.1%})")
    lines.append(f"factual-risk recall    {report.risk_recall:.1%}  (a miss ships a falsehood)")
    lines.append(
        f"factual-risk precision {report.risk_precision:.1%}  (a false alarm costs a rewrite)"
    )
    lines.append(
        f"mean score             {report.mean_score:.2f}  "
        f"(should-pass {report.mean_score_should_pass:.2f}, "
        f"should-fail {report.mean_score_should_fail:.2f})"
    )
    lines.append(f"discrimination         {report.discrimination:+.2f}")
    if report.errors:
        lines.append(f"unparseable replies    {len(report.errors)}")
    lines.append(f"cost                   ${report.total_cost_usd:.4f}")

    return "\n".join(lines)


async def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_evals",
        description="Score the labelled dataset with the judge-tier model and report metrics.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N examples")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=DEFAULT_MIN_AGREEMENT,
        help="exit non-zero below this agreement (a regression tripwire, not a target)",
    )
    parser.add_argument("--json", action="store_true", help="print metrics as JSON instead")
    args = parser.parse_args(argv)

    settings: Settings = get_settings()
    examples = load_dataset(args.dataset)[: args.limit]

    try:
        judge_model = settings.require_model(LLMTier.JUDGE)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    async with open_llm_provider(settings) as llm:
        report = await run_dataset(
            llm,
            examples,
            threshold=settings.eval_score_threshold,
            concurrency=args.concurrency,
            judge_model=judge_model,
        )

    print(json.dumps(report.as_dict(), indent=2) if args.json else format_report(report))

    if report.agreement < args.min_agreement:
        print(
            f"\nFAILED: agreement {report.agreement:.1%} is below "
            f"{args.min_agreement:.1%}. The judge has regressed, or the labels have.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
