"""
tests/test_memory.py
Tests for long-term memory components.

Run: pytest tests/test_memory.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


# ────────────────────────────────────────────────
# FIXTURES
# ────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path):
    """MemoryStore backed by a temp file (no Weaviate needed)."""
    from memory.memory_store import MemoryStore
    store = MemoryStore(
        weaviate_url="http://localhost:19999",   # intentionally wrong → forces fallback
        fallback_path=str(tmp_path / "memories.json"),
    )
    return store


@pytest.fixture
def sample_messages():
    return [
        {"role": "user",      "content": "I am analysing Infosys financial reports from 2020 to 2024."},
        {"role": "assistant", "content": "I can help with that. What would you like to know?"},
        {"role": "user",      "content": "Please always give me answers as bullet points."},
        {"role": "assistant", "content": "Understood. I will use bullet points from now on."},
        {"role": "user",      "content": "What was the Q3 net profit?"},
        {"role": "assistant", "content": "• Net profit: ₹6,586 crore\n• YoY growth: 3.2%"},
    ]


# ────────────────────────────────────────────────
# MEMORY STORE TESTS
# ────────────────────────────────────────────────

class TestMemoryStore:

    def test_file_fallback_initialises(self, tmp_store):
        """Store should initialise without Weaviate."""
        assert tmp_store._use_weaviate is False
        assert "EpisodicMemory" in tmp_store._db
        assert "SemanticMemory" in tmp_store._db

    def test_store_and_retrieve_episodic(self, tmp_store):
        mem_id = tmp_store.store_episodic(
            summary="User asked about Q3 revenue.",
            topics=["revenue", "Q3"],
            user_id="user_test",
        )
        assert mem_id is not None

        results = tmp_store.retrieve_episodic("revenue question", user_id="user_test")
        assert len(results) >= 1
        assert any("Q3" in r.get("summary", "") for r in results)

    def test_store_and_retrieve_semantic(self, tmp_store):
        tmp_store.store_semantic(
            fact="User prefers bullet-point answers.",
            fact_type="preference",
            confidence=0.95,
            user_id="user_test",
        )

        results = tmp_store.retrieve_semantic("answer format", user_id="user_test")
        assert len(results) >= 1
        assert any("bullet" in r.get("fact", "") for r in results)

    def test_semantic_dedup_skips_similar(self, tmp_store):
        """Storing a very similar fact twice should not create duplicate."""
        fact = "User is analysing Infosys annual reports."
        tmp_store.store_semantic(fact=fact, user_id="u1")
        tmp_store.store_semantic(fact=fact, user_id="u1")   # same again

        all_facts = tmp_store.get_all_semantic("u1")
        # At most 1 copy
        matching = [f for f in all_facts if "Infosys" in f.get("fact", "")]
        assert len(matching) <= 1

    def test_user_isolation(self, tmp_store):
        """Facts stored for user A should not appear for user B."""
        tmp_store.store_semantic("A specific fact for user A.", user_id="user_A")
        results_B = tmp_store.retrieve_semantic("specific fact", user_id="user_B")
        texts = [r.get("fact", "") for r in results_B]
        assert not any("user A" in t for t in texts)

    def test_get_all_semantic_returns_list(self, tmp_store):
        tmp_store.store_semantic("Fact one.",   user_id="u_all")
        tmp_store.store_semantic("Fact two.",   user_id="u_all")
        all_facts = tmp_store.get_all_semantic("u_all")
        assert isinstance(all_facts, list)
        assert len(all_facts) == 2

    def test_memory_summary_keys(self, tmp_store):
        summary = tmp_store.memory_summary("u_summary")
        assert "semantic_facts" in summary
        assert "episodic_count" in summary
        assert "preference_count" in summary

    def test_persistence_across_reload(self, tmp_path):
        """Facts written to file should survive store reload."""
        from memory.memory_store import MemoryStore

        store1 = MemoryStore(
            weaviate_url="http://localhost:19999",
            fallback_path=str(tmp_path / "mem.json"),
        )
        store1.store_semantic("Persistent fact.", user_id="reload_user")

        store2 = MemoryStore(
            weaviate_url="http://localhost:19999",
            fallback_path=str(tmp_path / "mem.json"),
        )
        facts = store2.get_all_semantic("reload_user")
        assert any("Persistent" in f.get("fact", "") for f in facts)


# ────────────────────────────────────────────────
# MEMORY EXTRACTOR TESTS (mock LLM)
# ────────────────────────────────────────────────

class TestMemoryExtractor:

    def _mock_openai(self, monkeypatch, response_text: str):
        """Patch OpenAI to return a fixed response."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = response_text

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        monkeypatch.setattr("memory.memory_extractor.OpenAI", lambda *args, **kwargs: mock_client)

    def test_episode_extraction(self, monkeypatch, sample_messages):
        episode_json = json.dumps({
            "summary":       "User asked about Infosys Q3 profit and requested bullet-point format.",
            "topics":        ["Infosys", "Q3", "profit"],
            "key_documents": ["Infosys Q3 report"],
            "user_intent":   "Understand Infosys quarterly financial performance",
        })
        self._mock_openai(monkeypatch, episode_json)

        from memory.memory_extractor import MemoryExtractor
        extractor = MemoryExtractor()
        result = extractor.extract_episode_summary(sample_messages)

        assert "summary" in result
        assert "topics"  in result
        assert result["message_count"] == len(sample_messages)

    def test_semantic_fact_extraction(self, monkeypatch, sample_messages):
        facts_json = json.dumps([
            {"fact": "User is analysing Infosys reports 2020-2024.", "fact_type": "context",    "confidence": 0.95},
            {"fact": "User prefers bullet-point format.",            "fact_type": "preference", "confidence": 0.98},
        ])
        self._mock_openai(monkeypatch, facts_json)

        from memory.memory_extractor import MemoryExtractor
        extractor = MemoryExtractor()
        facts = extractor.extract_semantic_facts(sample_messages)

        assert isinstance(facts, list)
        assert len(facts) == 2
        assert all("fact" in f for f in facts)
        assert all("fact_type" in f for f in facts)

    def test_process_session_combines_both(self, monkeypatch, sample_messages):
        """process_session should return episode AND new_facts."""
        episode_json = json.dumps({
            "summary": "Session about Infosys Q3.",
            "topics":  ["Infosys"],
            "key_documents": [],
            "user_intent": "Financial analysis",
        })
        # Two calls expected — mock returns same JSON for simplicity
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = episode_json
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        monkeypatch.setattr("memory.memory_extractor.OpenAI", lambda *args, **kwargs: mock_client)

        from memory.memory_extractor import MemoryExtractor
        extractor = MemoryExtractor()
        result = extractor.process_session(sample_messages)

        assert "episode"   in result
        assert "new_facts" in result

    def test_short_conversation_returns_empty(self, monkeypatch):
        """Single message should not trigger extraction."""
        from memory.memory_extractor import MemoryExtractor
        extractor = MemoryExtractor()
        result = extractor.process_session([{"role": "user", "content": "hi"}])
        assert result["episode"] is None
        assert result["new_facts"] == []


# ────────────────────────────────────────────────
# MEMORY ENABLED AGENT TESTS
# ────────────────────────────────────────────────

class TestMemoryEnabledAgent:

    def test_invoke_injects_memory(self, tmp_store):
        """Pre-stored facts should appear in the augmented question."""
        from memory.memory_agent import MemoryEnabledAgent

        # Pre-store a fact
        tmp_store.store_semantic(
            "User prefers concise answers.",
            fact_type="preference",
            user_id="agent_user",
        )

        # Fake base agent that echoes the question
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"answer": "mock answer", "question": ""}

        agent = MemoryEnabledAgent(fake_agent, memory_store=tmp_store)
        result = agent.invoke("What is the revenue?", user_id="agent_user")

        # Check that the base agent was called with memory context prepended
        call_args = fake_agent.invoke.call_args[0][0]
        assert "Long-term memory" in call_args.get("question", "")
        assert result["memory_context_used"] is True

    def test_invoke_no_memory_flag_false(self, tmp_store):
        """With no stored memories, memory_context_used should be False."""
        from memory.memory_agent import MemoryEnabledAgent

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"answer": "mock", "question": ""}

        agent = MemoryEnabledAgent(fake_agent, memory_store=tmp_store)
        result = agent.invoke("hello", user_id="new_user")
        assert result["memory_context_used"] is False

    def test_end_session_stores_memories(self, monkeypatch, tmp_store, sample_messages):
        """end_session should persist episode and facts."""
        from memory.memory_agent import MemoryEnabledAgent

        # Patch extractor to return fixed data without LLM calls
        mock_extractor = MagicMock()
        mock_extractor.process_session.return_value = {
            "episode": {
                "summary":       "User explored Infosys Q3 data.",
                "topics":        ["Infosys"],
                "key_documents": [],
                "message_count": len(sample_messages),
            },
            "new_facts": [
                {"fact": "User works with Infosys data.", "fact_type": "context", "confidence": 0.9},
            ],
        }

        fake_agent = MagicMock()
        agent = MemoryEnabledAgent(fake_agent, memory_store=tmp_store, memory_extractor=mock_extractor)

        # Simulate session messages
        agent._sessions["sess_user"] = sample_messages

        counts = agent.end_session("sess_user")
        assert counts["episodic_stored"] == 1
        assert counts["semantic_stored"] == 1

    def test_end_session_clears_buffer(self, tmp_store):
        """Session buffer should be empty after end_session."""
        from memory.memory_agent import MemoryEnabledAgent

        mock_extractor = MagicMock()
        mock_extractor.process_session.return_value = {"episode": None, "new_facts": []}

        fake_agent = MagicMock()
        agent = MemoryEnabledAgent(fake_agent, memory_store=tmp_store, memory_extractor=mock_extractor)
        agent._sessions["buf_user"] = [
            {"role": "user",      "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        agent.end_session("buf_user")
        assert "buf_user" not in agent._sessions
