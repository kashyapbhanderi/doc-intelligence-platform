import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


# ── Model tests (no server needed) ───────────────────────

def test_query_request_model():
    """QueryRequest should validate correctly."""
    from api.models import QueryRequest
    req = QueryRequest(question="What is RAG?")
    assert req.question == "What is RAG?"
    assert req.top_k == 5


def test_query_request_default_top_k():
    """Default top_k should be 5."""
    from api.models import QueryRequest
    req = QueryRequest(question="test question")
    assert req.top_k == 5


def test_query_request_custom_top_k():
    """Custom top_k should be accepted."""
    from api.models import QueryRequest
    req = QueryRequest(
        question="test question",
        top_k=10
    )
    assert req.top_k == 10


def test_query_request_too_short():
    """Questions under 3 chars should fail."""
    from api.models import QueryRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        QueryRequest(question="hi")


def test_edit_request_model():
    """EditRequest should validate correctly."""
    from api.models import EditRequest
    req = EditRequest(
        instruction="Replace DRAFT with FINAL",
        file_path="data/test.docx"
    )
    assert req.instruction == \
        "Replace DRAFT with FINAL"


def test_health_response_model():
    """HealthResponse should accept all fields."""
    from api.models import HealthResponse
    resp = HealthResponse(
        status="healthy",
        version="1.0.0",
        weaviate="healthy",
        total_chunks=11089
    )
    assert resp.status == "healthy"
    assert resp.total_chunks == 11089


# ── FastAPI app tests (TestClient, no server) ─────────────

@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_root_endpoint(client):
    """Root endpoint should return app info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data


def test_health_endpoint_returns_200(client):
    """Health endpoint should always return 200."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_endpoint_has_status(client):
    """Health response must have status field."""
    response = client.get("/api/v1/health")
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_docs_endpoint_returns_200(client):
    """Swagger docs should be accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_list_tools_endpoint(client):
    """Tools list endpoint should return tools."""
    response = client.get("/api/v1/edit/tools")
    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    assert data["total"] >= 10


def test_ingest_status_endpoint(client):
    """Ingest status should return chunk count."""
    response = client.get("/api/v1/ingest/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_query_endpoint_validates_input(client):
    """Short question should return 422."""
    response = client.post(
        "/api/v1/query",
        json={"question": "hi"}
    )
    assert response.status_code == 422


def test_api_files_exist():
    """All API files must exist."""
    files = [
        "api/main.py",
        "api/models.py",
        "api/routers/query.py",
        "api/routers/ingest.py",
        "api/routers/edit.py",
    ]
    for f in files:
        assert os.path.exists(f), \
            f"Missing: {f}"


def test_dockerfile_exists():
    """Dockerfile must exist."""
    assert os.path.exists("Dockerfile")


def test_docker_compose_has_api_service():
    """docker-compose should define api service."""
    import yaml
    with open("docker-compose.yml") as f:
        config = yaml.safe_load(f)
    assert "api" in config.get("services", {}), \
        "api service missing from docker-compose"


def test_docker_compose_has_weaviate():
    """docker-compose should define weaviate service."""
    import yaml
    with open("docker-compose.yml") as f:
        config = yaml.safe_load(f)
    services = config.get("services", {})
    assert "weaviate" in services