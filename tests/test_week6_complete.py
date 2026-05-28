import pytest
import sys
import os
import json
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestWeek6Complete:
    """Integration tests verifying all Week 6 deliverables."""

    # ── API files ─────────────────────────────────────────

    def test_api_main_exists(self):
        assert os.path.exists("api/main.py")

    def test_api_models_exist(self):
        assert os.path.exists("api/models.py")

    def test_api_routers_exist(self):
        for f in ["api/routers/query.py",
                  "api/routers/ingest.py",
                  "api/routers/edit.py"]:
            assert os.path.exists(f), \
                f"Missing router: {f}"

    def test_metrics_module_exists(self):
        assert os.path.exists("api/metrics.py")

    def test_task_queue_exists(self):
        assert os.path.exists("api/task_queue.py")

    # ── Docker files ──────────────────────────────────────

    def test_dockerfile_exists(self):
        assert os.path.exists("Dockerfile")

    def test_docker_compose_has_four_services(self):
        import yaml
        with open("docker-compose.yml") as f:
            config = yaml.safe_load(f)
        services = config.get("services", {})
        required = [
            "weaviate", "api",
            "prometheus", "grafana"
        ]
        for svc in required:
            assert svc in services, \
                f"Missing Docker service: {svc}"

    # ── Monitoring files ──────────────────────────────────

    def test_prometheus_config_exists(self):
        assert os.path.exists("prometheus.yml")

    def test_grafana_datasource_exists(self):
        assert os.path.exists(
            "grafana/provisioning/datasources/"
            "prometheus.yml"
        )

    def test_grafana_dashboard_exists(self):
        assert os.path.exists(
            "grafana/provisioning/dashboards/"
            "rag_dashboard.json"
        )

    def test_grafana_dashboard_valid_json(self):
        path = ("grafana/provisioning/dashboards/"
                "rag_dashboard.json")
        with open(path) as f:
            dashboard = json.load(f)
        assert "title"  in dashboard
        assert "panels" in dashboard
        assert len(dashboard["panels"]) >= 4

    # ── Evaluation files ──────────────────────────────────

    def test_ragas_results_exist(self):
        assert os.path.exists(
            "eval/ragas_results.json"), \
            "Run python eval/ragas_eval.py first"

    def test_ragas_scores_acceptable(self):
        path = "eval/ragas_results.json"
        if not os.path.exists(path):
            pytest.skip("RAGAS not run yet")
        with open(path) as f:
            data = json.load(f)
        s = data["summary"]
        assert s["faithfulness"]      >= 0.3
        assert s["answer_relevancy"]  >= 0.3
        assert s["context_recall"]    >= 0.3
        assert s["context_precision"] >= 0.3

    # ── Scripts ───────────────────────────────────────────

    def test_benchmark_script_exists(self):
        assert os.path.exists(
            "scripts/benchmark_api.py")

    def test_monitor_dashboard_exists(self):
        assert os.path.exists(
            "scripts/monitor_dashboard.py")

    def test_docker_stack_test_exists(self):
        assert os.path.exists(
            "scripts/test_docker_stack.py")