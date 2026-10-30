from collections.abc import Sequence

try:
    from .models import EvaluationModel
except ImportError:
    try:
        from evaluation.models import EvaluationModel
    except ImportError:
        from models import EvaluationModel


def validity_rate(results: Sequence[EvaluationModel]) -> float:
    """Percentage of evaluation cases where structured generation succeeded."""
    if not results:
        return 0.0

    return sum(result.success for result in results) / len(results)


def _successful_results(
    results: Sequence[EvaluationModel],
) -> list[EvaluationModel]:
    """Return only cases where structured generation succeeded."""

    return [
        result
        for result in results
        if result.success
    ]


def category_accuracy(results: Sequence[EvaluationModel]) -> float:
    """Accuracy of the predicted category among successfully generated structured outputs."""

    successful = _successful_results(results)

    if not successful:
        return 0.0

    return sum(
        result.category_correct
        for result in successful
    ) / len(successful)


def sentiment_accuracy(results: Sequence[EvaluationModel]) -> float:
    """Accuracy of the predicted sentiment among successfully generated structured outputs."""

    successful = _successful_results(results)

    if not successful:
        return 0.0

    return sum(
        result.sentiment_correct
        for result in successful
    ) / len(successful)


def urgency_accuracy(results: Sequence[EvaluationModel]) -> float:
    """Accuracy of the predicted urgency among successfully generated structured outputs."""

    successful = _successful_results(results)

    if not successful:
        return 0.0

    return sum(
        result.urgency_correct
        for result in successful
    ) / len(successful)


def overall_success_rate(
    results: Sequence[EvaluationModel],
) -> float:
    """Percentage of cases where all predictions were correct."""

    if not results:
        return 0.0

    successful_cases = sum(
        result.success
        and result.category_correct
        and result.sentiment_correct
        and result.urgency_correct
        for result in results
    )

    return successful_cases / len(results)


def failure_count(
    results: Sequence[EvaluationModel],
) -> int:
    """Number of cases where structured generation failed."""

    return sum(
        not result.success
        for result in results
    )

def metric_evaluation(results:Sequence[EvaluationModel])->dict:
    """Compute all metrics for evaluation results."""
    return{
        "Validity Rate":validity_rate(results),
        "Category Accuracy":category_accuracy(results),
        "Sentiment Accuracy":sentiment_accuracy(results),
        "Urgency Accuracy":urgency_accuracy(results),
        "Overall Success Rate":overall_success_rate(results),
        "Failure Count":failure_count(results)
        }
    