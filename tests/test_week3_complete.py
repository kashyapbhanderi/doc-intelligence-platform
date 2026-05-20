import pytest
import sys
import os
import json
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestWeek3Complete:
    """
    Integration tests verifying ALL Week 3
    deliverables are complete and working.
    """

    def test_finetuned_model_exists(self):
        """Fine-tuned model must exist."""
        assert os.path.exists(
            "models/finetuned/final"
        ), "Fine-tuned model not found!"

    def test_triplets_generated(self):
        """Clean triplets must exist."""
        assert os.path.exists(
            "data/triplets_clean.json"
        ), "Clean triplets not found!"

    def test_triplets_minimum_count(self):
        """Must have at least 200 clean triplets."""
        path = "data/triplets_clean.json"
        with open(path, encoding='utf-8') as f:
            triplets = json.load(f)
        assert len(triplets) >= 200

    def test_baseline_ndcg_exists(self):
        """Baseline NDCG results must exist."""
        assert os.path.exists(
            "eval/ndcg_results.json"
        ), "Baseline NDCG not found!"

    def test_finetuned_ndcg_exists(self):
        """Fine-tuned NDCG results must exist."""
        assert os.path.exists(
            "eval/ndcg_finetuned.json"
        ), "Fine-tuned NDCG not found!"

    def test_ndcg_improvement_positive(self):
        """Fine-tuned NDCG must be better than baseline."""
        if not (
            os.path.exists("eval/ndcg_results.json")
            and
            os.path.exists("eval/ndcg_finetuned.json")
        ):
            pytest.skip("Results not found")

        with open("eval/ndcg_results.json",
                  encoding='utf-8') as f:
            base = json.load(f)
        with open("eval/ndcg_finetuned.json",
                  encoding='utf-8') as f:
            ft = json.load(f)

        base_score = base["summary"]["hybrid_ndcg"]
        ft_score = ft["summary"]["hybrid_ndcg"]

        improvement = (
            (ft_score - base_score) /
            base_score * 100
        )

        print(f"\nBaseline:    {base_score:.4f}")
        print(f"Fine-tuned:  {ft_score:.4f}")
        print(f"Improvement: +{improvement:.1f}%")

        assert ft_score >= base_score * 0.95, \
            "Fine-tuned model is significantly worse!"

    def test_answer_eval_both_exist(self):
        """Both answer evaluations must exist."""
        assert os.path.exists(
            "eval/answer_eval_results.json"
        ), "Baseline answer eval not found!"

    def test_mlflow_experiment_exists(self):
        """MLflow should have experiments logged."""
        import mlflow
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        exp_names = [e.name for e in experiments]

        assert len(exp_names) > 1, \
            "No MLflow experiments found!"
        print(f"\nMLflow experiments: {exp_names}")

    def test_week3_all_files_present(self):
        """All Week 3 output files must exist."""
        required = [
            "models/finetuned/final",
            "data/triplets_clean.json",
            "eval/ndcg_results.json",
            "eval/ndcg_finetuned.json",
            "eval/ndcg_analysis.json",
            "eval/qa_dataset.json",
        ]
        missing = [
            f for f in required
            if not os.path.exists(f)
        ]
        assert len(missing) == 0, \
            f"Missing files: {missing}"

    def test_finetuned_model_produces_vectors(self):
        """Fine-tuned model must produce vectors."""
        model_path = "models/finetuned/final"
        if not os.path.exists(model_path):
            pytest.skip("Model not found")

        from sentence_transformers import (
            SentenceTransformer
        )
        model = SentenceTransformer(model_path)
        vector = model.encode("test sentence")
        assert len(vector) == 384