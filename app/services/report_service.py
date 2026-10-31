"""Transparent, non-clinical comparison against intervals supplied by a report."""

from app.schemas.report import ReportEvaluationRequest, ReportEvaluationResponse, ReportItemEvaluation
from ml.preprocessing.biopro import FEATURE_METADATA


def evaluate_report(request: ReportEvaluationRequest) -> ReportEvaluationResponse:
    evaluations = [_evaluate_item(item) for item in request.items]
    return ReportEvaluationResponse(
        items=evaluations,
        limitation=(
            "Esta comparação usa exclusivamente o intervalo informado no próprio laudo. "
            "Ela não diagnostica condições, não explica causas e não substitui avaliação profissional."
        ),
    )


def _evaluate_item(item) -> ReportItemEvaluation:
    if item.result < item.reference_lower:
        position = "below"
        statement = "O resultado está abaixo do limite inferior informado no laudo."
    elif item.result > item.reference_upper:
        position = "above"
        statement = "O resultado está acima do limite superior informado no laudo."
    else:
        position = "within"
        statement = "O resultado está dentro do intervalo informado no laudo."

    return ReportItemEvaluation(
        feature_code=item.feature_code,
        test_name=FEATURE_METADATA[item.feature_code]["name"],
        result=item.result,
        unit=item.unit,
        reference_lower=item.reference_lower,
        reference_upper=item.reference_upper,
        position=position,
        statement=statement,
    )
