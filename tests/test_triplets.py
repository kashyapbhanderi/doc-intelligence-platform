import pytest
import sys
import os
import json
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def test_triplets_file_exists():
    """Raw triplets file should exist."""
    assert os.path.exists("data/triplets.json"), \
        "Run python embeddings/generate_triplets.py"


def test_clean_triplets_file_exists():
    """Clean triplets file should exist."""
    assert os.path.exists(
        "data/triplets_clean.json"), \
        "Run python scripts/validate_triplets.py"


def test_triplets_count():
    """Should have at least 200 clean triplets."""
    path = "data/triplets_clean.json"
    if not os.path.exists(path):
        pytest.skip("Clean triplets not generated")

    with open(path, encoding='utf-8') as f:
        triplets = json.load(f)
    assert len(triplets) >= 200, \
        f"Only {len(triplets)} triplets, need 200+"


def test_triplets_have_required_fields():
    """Every triplet needs anchor, positive, negative."""
    path = "data/triplets_clean.json"
    if not os.path.exists(path):
        pytest.skip("Clean triplets not generated")

    with open(path, encoding='utf-8') as f:
        triplets = json.load(f)

    for t in triplets[:20]:
        assert "anchor" in t
        assert "positive" in t
        assert "negative" in t
        assert "positive_source" in t
        assert "negative_source" in t


def test_triplets_sources_are_different():
    """Positive and negative must come from different sources."""
    path = "data/triplets_clean.json"
    if not os.path.exists(path):
        pytest.skip("Clean triplets not generated")

    with open(path, encoding='utf-8') as f:
        triplets = json.load(f)

    for t in triplets[:50]:
        assert t["positive_source"] != \
               t["negative_source"], \
            "Positive and negative from same source!"


def test_triplets_minimum_length():
    """All fields must meet minimum length."""
    path = "data/triplets_clean.json"
    if not os.path.exists(path):
        pytest.skip("Clean triplets not generated")

    with open(path, encoding='utf-8') as f:
        triplets = json.load(f)

    for t in triplets[:50]:
        assert len(t["anchor"]) >= 10
        assert len(t["positive"]) >= 50
        assert len(t["negative"]) >= 50


def test_triplets_no_duplicates():
    """Anchors should be unique across triplets."""
    path = "data/triplets_clean.json"
    if not os.path.exists(path):
        pytest.skip("Clean triplets not generated")

    with open(path, encoding='utf-8') as f:
        triplets = json.load(f)

    anchors = [t["anchor"][:50].lower()
               for t in triplets]
    unique = set(anchors)
    assert len(anchors) == len(unique), \
        "Duplicate anchors found!"


def test_positive_longer_than_anchor():
    """Positive chunks should be longer than questions."""
    path = "data/triplets_clean.json"
    if not os.path.exists(path):
        pytest.skip("Clean triplets not generated")

    with open(path, encoding='utf-8') as f:
        triplets = json.load(f)

    for t in triplets[:20]:
        assert len(t["positive"]) > len(t["anchor"])