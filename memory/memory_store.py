"""
Memory — memory_store.py
=========================
Two-tier persistent memory for your LangGraph agent.

TIER 1 — Episodic Memory
  What happened in past conversations.
  "On Jan 15, user spent 5 exchanges analysing Infosys Q3 cash flow."

TIER 2 — Semantic Memory
  Distilled, deduplicated facts about the user and domain.
  "User prefers bullet-point answers."
  "User is analysing Infosys financial reports 2020–2024."
  "User corrected: revenue figure is ₹146B not ₹144B."

Storage: Uses your existing Weaviate instance. Falls back to a local
JSON file if Weaviate is not running (useful during development).

WHERE TO PLACE THIS FILE: memory/memory_store.py

INSTALL DEPENDENCIES (already in your project):
    pip install sentence-transformers weaviate-client
"""

import json
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

try:
    import weaviate
    _WEAVIATE_AVAILABLE = True
except ImportError:
    _WEAVIATE_AVAILABLE = False


class MemoryStore:
    """
    Persistent memory store with vector-similarity retrieval.

    Two Weaviate classes are created automatically on first use:
      - EpisodicMemory : conversation summaries
      - SemanticMemory : distilled facts / preferences

    File fallback: if Weaviate is unavailable, stores in data/memories.json
    using numpy cosine similarity (no extra dependency).
    """

    _EPISODIC = "EpisodicMemory"
    _SEMANTIC  = "SemanticMemory"

    def __init__(
        self,
        weaviate_url:  str = "http://localhost:8080",
        model_name:    str = "all-MiniLM-L6-v2",     # already used by your embedder
        fallback_path: str = "data/memories.json",
    ):
        self.model = SentenceTransformer(model_name)
        self._fallback = Path(fallback_path)
        self._fallback.parent.mkdir(parents=True, exist_ok=True)

        self._use_weaviate = False
        self.client: Optional[object] = None

        if _WEAVIATE_AVAILABLE:
            try:
                self.client = weaviate.Client(weaviate_url)
                self.client.schema.get()       # connectivity check
                self._setup_schemas()
                self._use_weaviate = True
                print("[MemoryStore] Using Weaviate")
            except Exception as exc:
                print(f"[MemoryStore] Weaviate unavailable ({exc}), using file fallback")

        if not self._use_weaviate:
            self._db = self._load_file()

    # ────────────────────────────────────────────────
    # WEAVIATE SCHEMA SETUP
    # ────────────────────────────────────────────────

    def _setup_schemas(self) -> None:
        existing = {c["class"] for c in self.client.schema.get().get("classes", [])}

        if self._EPISODIC not in existing:
            self.client.schema.create_class({
                "class":      self._EPISODIC,
                "vectorizer": "none",
                "properties": [
                    {"name": "summary",       "dataType": ["text"]},
                    {"name": "user_id",       "dataType": ["text"]},
                    {"name": "session_id",    "dataType": ["text"]},
                    {"name": "timestamp",     "dataType": ["text"]},
                    {"name": "message_count", "dataType": ["int"]},
                    {"name": "topics",        "dataType": ["text[]"]},
                ],
            })
            print(f"[MemoryStore] Created Weaviate class: {self._EPISODIC}")

        if self._SEMANTIC not in existing:
            self.client.schema.create_class({
                "class":      self._SEMANTIC,
                "vectorizer": "none",
                "properties": [
                    {"name": "fact",           "dataType": ["text"]},
                    {"name": "user_id",        "dataType": ["text"]},
                    {"name": "fact_type",      "dataType": ["text"]},   # preference|fact|context|correction
                    {"name": "confidence",     "dataType": ["number"]},
                    {"name": "created_at",     "dataType": ["text"]},
                    {"name": "last_confirmed", "dataType": ["text"]},
                ],
            })
            print(f"[MemoryStore] Created Weaviate class: {self._SEMANTIC}")

    # ────────────────────────────────────────────────
    # FILE FALLBACK HELPERS
    # ────────────────────────────────────────────────

    def _load_file(self) -> dict:
        if self._fallback.exists():
            with open(self._fallback) as f:
                return json.load(f)
        return {self._EPISODIC: [], self._SEMANTIC: []}

    def _save_file(self) -> None:
        with open(self._fallback, "w") as f:
            json.dump(self._db, f, indent=2)

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        import math
        dot  = sum(x * y for x, y in zip(a, b))
        norm = math.sqrt(sum(x ** 2 for x in a)) * math.sqrt(sum(y ** 2 for y in b))
        return dot / (norm + 1e-9)

    # ────────────────────────────────────────────────
    # EPISODIC MEMORY — WRITE
    # ────────────────────────────────────────────────

    def store_episodic(
        self,
        summary:       str,
        topics:        list[str],
        user_id:       str = "default",
        session_id:    Optional[str] = None,
        message_count: int = 0,
    ) -> str:
        """
        Store a conversation episode summary.
        Call this at the end of each user session via MemoryEnabledAgent.end_session().
        """
        mem_id     = str(uuid.uuid4())
        session_id = session_id or mem_id
        timestamp  = datetime.utcnow().isoformat()
        vector     = self.model.encode(summary).tolist()

        if self._use_weaviate:
            self.client.data_object.create(
                {
                    "summary":       summary,
                    "user_id":       user_id,
                    "session_id":    session_id,
                    "timestamp":     timestamp,
                    "message_count": message_count,
                    "topics":        topics,
                },
                self._EPISODIC,
                uuid=mem_id,
                vector=vector,
            )
        else:
            self._db[self._EPISODIC].append({
                "id":            mem_id,
                "summary":       summary,
                "user_id":       user_id,
                "session_id":    session_id,
                "timestamp":     timestamp,
                "message_count": message_count,
                "topics":        topics,
                "vector":        vector,
            })
            self._save_file()

        return mem_id

    # ────────────────────────────────────────────────
    # EPISODIC MEMORY — READ
    # ────────────────────────────────────────────────

    def retrieve_episodic(
        self,
        query:    str,
        user_id:  str = "default",
        top_k:    int = 3,
        days_back: int = 90,
    ) -> list[dict]:
        """
        Return the most relevant past episodes for this query.
        """
        cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
        vector = self.model.encode(query).tolist()

        if self._use_weaviate:
            where_filter = {
                "operator": "And",
                "operands": [
                    {"path": ["user_id"],  "operator": "Equal",       "valueText":  user_id},
                    {"path": ["timestamp"], "operator": "GreaterThan", "valueText":  cutoff},
                ],
            }
            result = (
                self.client.query
                .get(self._EPISODIC, ["summary", "session_id", "timestamp", "topics", "message_count"])
                .with_near_vector({"vector": vector})
                .with_where(where_filter)
                .with_limit(top_k)
                .do()
            )
            return result.get("data", {}).get("Get", {}).get(self._EPISODIC, [])

        # File fallback: cosine similarity
        scored = []
        for mem in self._db.get(self._EPISODIC, []):
            if mem.get("user_id") != user_id:
                continue
            if mem.get("timestamp", "") < cutoff:
                continue
            sim = self._cosine_sim(vector, mem.get("vector", []))
            scored.append({**mem, "_similarity": sim})

        scored.sort(key=lambda x: x.get("_similarity", 0), reverse=True)
        return scored[:top_k]

    # ────────────────────────────────────────────────
    # SEMANTIC MEMORY — WRITE
    # ────────────────────────────────────────────────

    def store_semantic(
        self,
        fact:       str,
        fact_type:  str   = "fact",     # preference | fact | context | correction
        confidence: float = 0.9,
        user_id:    str   = "default",
    ) -> str:
        """
        Store a long-term semantic fact about a user.
        Auto-skips near-duplicate facts (cosine sim > 0.92).
        """
        # Dedup check
        existing = self.retrieve_semantic(fact, user_id=user_id, top_k=1)
        if existing:
            top_sim = existing[0].get("_similarity", 0)
            if top_sim > 0.92:
                return existing[0].get("id", "")   # already known

        mem_id = str(uuid.uuid4())
        now    = datetime.utcnow().isoformat()
        vector = self.model.encode(fact).tolist()

        if self._use_weaviate:
            self.client.data_object.create(
                {
                    "fact":           fact,
                    "user_id":        user_id,
                    "fact_type":      fact_type,
                    "confidence":     float(confidence),
                    "created_at":     now,
                    "last_confirmed": now,
                },
                self._SEMANTIC,
                uuid=mem_id,
                vector=vector,
            )
        else:
            self._db[self._SEMANTIC].append({
                "id":             mem_id,
                "fact":           fact,
                "user_id":        user_id,
                "fact_type":      fact_type,
                "confidence":     float(confidence),
                "created_at":     now,
                "last_confirmed": now,
                "vector":         vector,
            })
            self._save_file()

        return mem_id

    # ────────────────────────────────────────────────
    # SEMANTIC MEMORY — READ
    # ────────────────────────────────────────────────

    def retrieve_semantic(
        self,
        query:          str,
        user_id:        str   = "default",
        top_k:          int   = 5,
        min_confidence: float = 0.5,
    ) -> list[dict]:
        """Return the most relevant semantic facts for this query."""
        vector = self.model.encode(query).tolist()

        if self._use_weaviate:
            result = (
                self.client.query
                .get(self._SEMANTIC, ["fact", "fact_type", "confidence", "created_at"])
                .with_near_vector({"vector": vector})
                .with_where({
                    "operator": "And",
                    "operands": [
                        {"path": ["user_id"],    "operator": "Equal",              "valueText":  user_id},
                        {"path": ["confidence"], "operator": "GreaterThanEqual",   "valueNumber": min_confidence},
                    ],
                })
                .with_limit(top_k)
                .do()
            )
            return result.get("data", {}).get("Get", {}).get(self._SEMANTIC, [])

        # File fallback
        scored = []
        for mem in self._db.get(self._SEMANTIC, []):
            if mem.get("user_id") != user_id:
                continue
            if mem.get("confidence", 1.0) < min_confidence:
                continue
            sim = self._cosine_sim(vector, mem.get("vector", []))
            scored.append({**mem, "_similarity": sim})

        scored.sort(key=lambda x: x.get("_similarity", 0), reverse=True)
        return scored[:top_k]

    def get_all_semantic(self, user_id: str = "default") -> list[dict]:
        """Get all stored facts for a user — useful for debugging & display."""
        if self._use_weaviate:
            result = (
                self.client.query
                .get(self._SEMANTIC, ["fact", "fact_type", "confidence", "created_at"])
                .with_where({"path": ["user_id"], "operator": "Equal", "valueText": user_id})
                .with_limit(200)
                .do()
            )
            return result.get("data", {}).get("Get", {}).get(self._SEMANTIC, [])

        return [m for m in self._db.get(self._SEMANTIC, []) if m.get("user_id") == user_id]

    def memory_summary(self, user_id: str = "default") -> dict:
        """Quick stats — log to MLflow after each session."""
        semantic = self.get_all_semantic(user_id)
        episodic = self._db.get(self._EPISODIC, []) if not self._use_weaviate else []
        return {
            "user_id":          user_id,
            "semantic_facts":   len(semantic),
            "episodic_count":   len(episodic),
            "preference_count": sum(1 for f in semantic if f.get("fact_type") == "preference"),
            "fact_count":       sum(1 for f in semantic if f.get("fact_type") == "fact"),
        }
