import pytest
import sys
import os
import json
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


# ── RAGAS results tests ───────────────────────────────────

def test_ragas_results_file_exists():
    """RAGAS results file should exist after eval."""
    assert os.path.exists(
        "eval/ragas_results.json"), \
        "Run python eval/ragas_eval.py first"


def test_ragas_results_has_summary():
    """RAGAS file must have summary field."""
    path = "eval/ragas_results.json"
    if not os.path.exists(path):
        pytest.skip("RAGAS not run yet")
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    assert "summary" in data


def test_ragas_has_all_four_metrics():
    """All 4 RAGAS metrics must be present."""
    path = "eval/ragas_results.json"
    if not os.path.exists(path):
        pytest.skip("RAGAS not run yet")
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    summary  = data["summary"]
    required = [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]
    for metric in required:
        assert metric in summary, \
            f"Missing RAGAS metric: {metric}"


def test_ragas_scores_between_0_and_1():
    """All RAGAS scores must be between 0 and 1."""
    path = "eval/ragas_results.json"
    if not os.path.exists(path):
        pytest.skip("RAGAS not run yet")
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    scores = data["summary"]
    for metric in [
        "faithfulness", "answer_relevancy",
        "context_recall", "context_precision"
    ]:
        val = scores[metric]
        assert 0.0 <= val <= 1.0, \
            f"{metric} out of range: {val}"


def test_ragas_faithfulness_above_threshold():
    """Faithfulness should be above 0.5."""
    path = "eval/ragas_results.json"
    if not os.path.exists(path):
        pytest.skip("RAGAS not run yet")
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    score = data["summary"]["faithfulness"]
    print(f"\nFaithfulness: {score:.4f}")
    assert score >= 0.3, \
        f"Faithfulness too low: {score:.4f}"


def test_ragas_answer_relevancy_above_threshold():
    """Answer relevancy should be above 0.5."""
    path = "eval/ragas_results.json"
    if not os.path.exists(path):
        pytest.skip("RAGAS not run yet")
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    score = data["summary"]["answer_relevancy"]
    print(f"\nAnswer relevancy: {score:.4f}")
    assert score >= 0.3, \
        f"Answer relevancy too low: {score:.4f}"


def test_ragas_evaluated_minimum_samples():
    """Should have evaluated at least 10 samples."""
    path = "eval/ragas_results.json"
    if not os.path.exists(path):
        pytest.skip("RAGAS not run yet")
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    n = data["summary"].get("num_samples", 0)
    assert n >= 10, \
        f"Too few samples evaluated: {n}"


# ── Docker stack tests ────────────────────────────────────

def test_dockerfile_exists():
    """Dockerfile must exist."""
    assert os.path.exists("Dockerfile")


def test_docker_compose_has_api():
    """docker-compose should have api service."""
    import yaml
    with open("docker-compose.yml") as f:
        config = yaml.safe_load(f)
    assert "api" in config.get("services", {})


def test_docker_test_script_exists():
    """Docker stack test script must exist."""
    assert os.path.exists(
        "scripts/test_docker_stack.py")


def test_benchmark_script_exists():
    """Benchmark script must exist."""
    assert os.path.exists(
        "scripts/benchmark_api.py")