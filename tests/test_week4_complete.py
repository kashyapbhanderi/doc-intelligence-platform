import pytest
import sys
import os
import json
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestWeek4Complete:
    """Full Week 4 completion checklist."""

    # ── File existence ────────────────────────────────

    def test_state_file_exists(self):
        assert os.path.exists("agents/state.py")

    def test_planner_file_exists(self):
        assert os.path.exists("agents/planner.py")

    def test_executor_file_exists(self):
        assert os.path.exists("agents/executor.py")

    def test_critic_file_exists(self):
        assert os.path.exists("agents/critic.py")

    def test_graph_file_exists(self):
        assert os.path.exists("agents/graph.py")

    def test_agent_eval_results_exist(self):
        assert os.path.exists(
            "eval/agent_eval_results.json"), \
            "Run python eval/agent_eval.py first"

    # ── Agent eval results quality ────────────────────

    def test_eval_has_summary(self):
        path = "eval/agent_eval_results.json"
        if not os.path.exists(path):
            pytest.skip("Agent eval not run yet")
        with open(path) as f:
            data = json.load(f)
        assert "summary" in data
        assert "results" in data

    def test_eval_ran_20_questions(self):
        path = "eval/agent_eval_results.json"
        if not os.path.exists(path):
            pytest.skip("Agent eval not run yet")
        with open(path) as f:
            data = json.load(f)
        assert data["summary"]["total_questions"] == 20

    def test_faithfulness_rate_above_50_pct(self):
        """At least 50% of answers should be faithful."""
        path = "eval/agent_eval_results.json"
        if not os.path.exists(path):
            pytest.skip("Agent eval not run yet")
        with open(path) as f:
            data = json.load(f)
        rate = data["summary"]["faithfulness_rate"]
        assert rate >= 0.50, \
            f"Faithfulness rate too low: {rate:.1%}"

    def test_avg_latency_under_15s(self):
        """Average latency should be under 15 seconds."""
        path = "eval/agent_eval_results.json"
        if not os.path.exists(path):
            pytest.skip("Agent eval not run yet")
        with open(path) as f:
            data = json.load(f)
        latency = data["summary"]["avg_latency"]
        assert latency < 15.0, \
            f"Avg latency too high: {latency:.1f}s"

    def test_each_result_has_sources(self):
        """Every answer should have at least 1 source."""
        path = "eval/agent_eval_results.json"
        if not os.path.exists(path):
            pytest.skip("Agent eval not run yet")
        with open(path) as f:
            data = json.load(f)
        for r in data["results"]:
            assert r["num_sources"] >= 1, \
                f"No sources for: {r['question'][:50]}"

    def test_each_result_has_sub_queries(self):
        """Planner should generate sub-queries for all questions."""
        path = "eval/agent_eval_results.json"
        if not os.path.exists(path):
            pytest.skip("Agent eval not run yet")
        with open(path) as f:
            data = json.load(f)
        for r in data["results"]:
            assert len(r["sub_queries"]) >= 1, \
                f"No sub-queries for: {r['question'][:50]}"

    # ── Graph structure ───────────────────────────────

    def test_graph_compiles(self):
        from agents.graph import build_agent_graph
        app = build_agent_graph()
        assert app is not None

    def test_graph_entry_is_planner(self):
        from agents.graph import build_agent_graph
        app   = build_agent_graph()
        graph = app.get_graph()
        # Entry node should connect to planner
        assert "planner" in graph.nodes

    def test_all_agents_importable(self):
        from agents.planner  import planner_node
        from agents.executor import executor_node
        from agents.critic   import critic_node
        assert all([planner_node,
                    executor_node,
                    critic_node])

    # ── LangSmith config ──────────────────────────────

    def test_langsmith_env_vars_set(self):
        """LangSmith env vars should be configured."""
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("LANGCHAIN_API_KEY", "")
        assert len(key) > 0, \
            "LANGCHAIN_API_KEY not set in .env"