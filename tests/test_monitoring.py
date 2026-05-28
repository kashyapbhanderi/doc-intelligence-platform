import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


# ── Prometheus config tests ───────────────────────────────

def test_prometheus_config_exists():
    """prometheus.yml must exist."""
    assert os.path.exists("prometheus.yml"), \
        "Create prometheus.yml first"


def test_prometheus_config_valid_yaml():
    """prometheus.yml must be valid YAML."""
    import yaml
    with open("prometheus.yml") as f:
        config = yaml.safe_load(f)
    assert "scrape_configs" in config
    assert "global" in config


def test_prometheus_config_has_api_target():
    """Prometheus must scrape the API service."""
    import yaml
    with open("prometheus.yml") as f:
        config = yaml.safe_load(f)
    jobs = [
        j["job_name"]
        for j in config["scrape_configs"]
    ]
    assert "doc-intelligence-api" in jobs, \
        "API scrape job missing from prometheus.yml"


def test_docker_compose_has_prometheus():
    """docker-compose should define prometheus service."""
    import yaml
    with open("docker-compose.yml") as f:
        config = yaml.safe_load(f)
    services = config.get("services", {})
    assert "prometheus" in services, \
        "prometheus service missing from docker-compose"


def test_docker_compose_prometheus_port():
    """Prometheus should expose port 9090."""
    import yaml
    with open("docker-compose.yml") as f:
        config = yaml.safe_load(f)
    prom = config["services"]["prometheus"]
    ports = prom.get("ports", [])
    assert any("9090" in str(p) for p in ports), \
        "Port 9090 not exposed for Prometheus"


# ── Metrics module tests ──────────────────────────────────

def test_metrics_module_imports():
    """All metrics should import without errors."""
    from api.metrics import (
        QUERY_TOTAL,
        QUERY_LATENCY,
        ACTIVE_QUERIES,
        WEAVIATE_CHUNKS,
        FAITHFUL_ANSWERS,
        UNFAITHFUL_ANSWERS,
    )
    assert QUERY_TOTAL   is not None
    assert QUERY_LATENCY is not None


def test_query_counter_has_labels():
    """QUERY_TOTAL must support status labels."""
    from api.metrics import QUERY_TOTAL
    # Should not raise
    QUERY_TOTAL.labels(status="success")
    QUERY_TOTAL.labels(status="error")


def test_latency_histogram_exists():
    """Latency histogram must be defined."""
    from api.metrics import QUERY_LATENCY
    # Observe a dummy value — should not raise
    QUERY_LATENCY.observe(1.5)


def test_active_queries_gauge():
    """Active queries gauge must support inc/dec."""
    from api.metrics import ACTIVE_QUERIES
    ACTIVE_QUERIES.inc()
    ACTIVE_QUERIES.dec()


def test_weaviate_chunks_gauge():
    """Weaviate chunks gauge must support set."""
    from api.metrics import WEAVIATE_CHUNKS
    WEAVIATE_CHUNKS.set(11089)


# ── Dashboard script test ─────────────────────────────────

def test_monitor_script_exists():
    """Monitor dashboard script must exist."""
    assert os.path.exists(
        "scripts/monitor_dashboard.py")


def test_metrics_endpoint_in_main():
    """api/main.py should include Instrumentator."""
    with open("api/main.py",
               encoding='utf-8') as f:
        content = f.read()
    assert "Instrumentator" in content, \
        "Prometheus Instrumentator not in main.py"
    assert "prometheus" in content.lower(), \
        "Prometheus not referenced in main.py"