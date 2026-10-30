from fastapi.testclient import TestClient

from app.api.routes import get_available_service
from app.main import app


class FakeService:
    def analyze(self, panel):
        return {
            "model_version": "test-model",
            "anomaly_score": 0.42,
            "is_statistically_unusual": False,
            "observed_feature_count": 2,
            "missing_feature_count": 16,
            "interpretation": "Statistical comparison.",
            "limitation": "Not a diagnosis.",
        }


def test_healthcheck() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analysis_accepts_partial_panel_and_returns_non_clinical_response() -> None:
    app.dependency_overrides[get_available_service] = lambda: FakeService()
    client = TestClient(app)

    response = client.post("/v1/analyses", json={"LBXSATSI": 12.5, "LBDSALSI": 42.0})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["anomaly_score"] == 0.42
    assert body["missing_feature_count"] == 16
    assert "diagnosis" in body["limitation"].lower()


def test_analysis_rejects_empty_or_unknown_panels() -> None:
    client = TestClient(app)
    assert client.post("/v1/analyses", json={}).status_code == 422
    assert client.post("/v1/analyses", json={"not_a_feature": 12}).status_code == 422
