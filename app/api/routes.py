"""REST endpoints for LabInsight."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.analysis import AnalysisResponse, LaboratoryPanel
from app.services.anomaly_service import AnomalyService, ModelUnavailableError, get_anomaly_service


router = APIRouter()


def get_available_service() -> AnomalyService:
    """Turn a missing or incompatible local artifact into an HTTP 503."""
    try:
        return get_anomaly_service()
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail="Model is temporarily unavailable") from error


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
