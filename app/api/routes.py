"""REST endpoints for LabInsight."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.analysis import AnalysisResponse, LaboratoryPanel
from app.schemas.explanation import ExplanationRequest, ExplanationResponse
from app.schemas.report import ReportEvaluationRequest, ReportEvaluationResponse
from app.services.anomaly_service import AnomalyService, ModelUnavailableError, get_anomaly_service
from app.services.rag_service import ExplanationUnavailableError, RagService, get_rag_service
from app.services.report_service import evaluate_report


router = APIRouter()


def get_available_service() -> AnomalyService:
    """Turn a missing or incompatible local artifact into an HTTP 503."""
    try:
        return get_anomaly_service()
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail="Model is temporarily unavailable") from error


def get_available_rag_service() -> RagService:
    try:
        return get_rag_service()
    except ValueError as error:
        raise HTTPException(status_code=503, detail="Knowledge base is temporarily unavailable") from error


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/v1/analyses",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a laboratory panel statistically",
)
def create_analysis(
    panel: LaboratoryPanel,
    service: AnomalyService = Depends(get_available_service),
) -> AnalysisResponse:
    """Return a statistical anomaly signal; never a clinical conclusion."""
    return service.analyze(panel)


@router.post(
    "/v1/report-evaluations",
    response_model=ReportEvaluationResponse,
    summary="Compare results with intervals supplied by a report",
)
def create_report_evaluation(request: ReportEvaluationRequest) -> ReportEvaluationResponse:
    """Classify position only, without interpreting health meaning or cause."""
    return evaluate_report(request)


@router.post(
    "/v1/explanations",
    response_model=ExplanationResponse,
    summary="Explain LabInsight methodology using retrieved context",
)
def create_explanation(
    request: ExplanationRequest,
    service: RagService = Depends(get_available_rag_service),
) -> ExplanationResponse:
    try:
        return service.explain(request)
    except ExplanationUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail="Gemini explanations are not configured or temporarily unavailable",
        ) from error
