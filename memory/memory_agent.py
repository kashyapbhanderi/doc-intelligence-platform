"""
Memory — memory_agent.py
=========================
Drop-in wrapper that gives your existing LangGraph agent long-term memory.

HOW IT WORKS:
  1. Before each query → retrieve relevant past memories → inject into question
  2. Agent answers the memory-augmented question (no changes to agent code)
  3. After session ends → extract & store new memories

WHERE TO PLACE THIS FILE: memory/memory_agent.py

INTEGRATION — replace your existing agent call in api/main.py:
──────────────────────────────────────────────────────────────
# BEFORE:
from agents.graph import agent_graph

@app.post("/query")
async def query(request: QueryRequest):
    result = agent_graph.invoke({"question": request.question})
    return result

# AFTER:
from agents.graph import agent_graph
from memory.memory_agent import MemoryEnabledAgent

memory_agent = MemoryEnabledAgent(agent_graph)

@app.post("/query")
async def query(request: QueryRequest):
    result = memory_agent.invoke(request.question, user_id=request.user_id)
    return result

@app.post("/session/end")
async def end_session(user_id: str):
    counts = memory_agent.end_session(user_id)
    return {"memories_stored": counts}
──────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import Optional

from memory.memory_store import MemoryStore
from memory.memory_extractor import MemoryExtractor


# ────────────────────────────────────────────────
# MEMORY CONTEXT BUILDER
# ────────────────────────────────────────────────

def build_memory_context(
    query:        str,
    user_id:      str,
    memory_store: MemoryStore,
    max_episodic: int = 2,
    max_semantic: int = 5,
) -> str:
    """
    Build a formatted memory block to prepend to the agent's question.

    Returns empty string if no memories exist yet (first-time user).

    Output format injected into the question:
    ┌─ Long-term memory ──────────────────────────────┐
    │ Known facts and preferences:                    │
    │   [preference] User prefers bullet-point format │
    │   [context]    Analysing Infosys 2020–2024 data │
    │                                                 │
    │ Past relevant conversations:                    │
    │   [2024-01-15] Discussed Q3 revenue breakdown … │
    └─────────────────────────────────────────────────┘
    """
    episodic = memory_store.retrieve_episodic(query, user_id=user_id, top_k=max_episodic)
    semantic = memory_store.retrieve_semantic(query, user_id=user_id, top_k=max_semantic)

    if not episodic and not semantic:
        return ""

    lines = ["--- Long-term memory ---"]

    if semantic:
        lines.append("Known facts and preferences:")
        for fact in semantic:
            ftype = fact.get("fact_type", "fact")
            ftext = fact.get("fact", "")
            lines.append(f"  [{ftype}] {ftext}")

    if episodic:
        lines.append("Past relevant conversations:")
        for ep in episodic:
            date    = ep.get("timestamp", "")[:10]   # YYYY-MM-DD
            summary = ep.get("summary", "")
            lines.append(f"  [{date}] {summary}")

    lines.append("------------------------")
    return "\n".join(lines)


# ────────────────────────────────────────────────
# MEMORY-ENABLED AGENT WRAPPER
# ────────────────────────────────────────────────

class MemoryEnabledAgent:
    """
    Wraps your compiled LangGraph agent with long-term memory.
    No changes needed to your existing agent code.

    Args:
        base_agent       : your compiled LangGraph graph (the object you call .invoke() on)
        memory_store     : MemoryStore instance (created automatically if None)
        memory_extractor : MemoryExtractor instance (created automatically if None)

    Example:
        from agents.graph import agent_graph          # your existing agent
        from memory.memory_agent import MemoryEnabledAgent

        agent = MemoryEnabledAgent(agent_graph)
        result = agent.invoke("What is the Q3 net profit?", user_id="user_123")
        # ... at end of session:
        stored = agent.end_session("user_123")
        # → {"episodic_stored": 1, "semantic_stored": 3}
    """

    def __init__(
        self,
        base_agent,
        memory_store:     Optional[MemoryStore]     = None,
        memory_extractor: Optional[MemoryExtractor] = None,
    ):
        self.agent     = base_agent
        self.store     = memory_store     or MemoryStore()
        self.extractor = memory_extractor or MemoryExtractor()

        # Per-user session message history (cleared on end_session)
        self._sessions: dict[str, list[dict]] = {}

    # ────────────────────────────────────────────────
    # INVOKE (main entry point)
    # ────────────────────────────────────────────────

    def invoke(
        self,
        question:   str,
        user_id:    str           = "default",
        session_id: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Invoke the agent with long-term memory context injected.

        Args:
            question   : raw user question
            user_id    : unique user identifier (use auth token or email hash)
            session_id : optional session identifier
            **kwargs   : passed through to base_agent.invoke()

        Returns:
            agent result dict with an extra key "memory_context_used" (bool)
        """
        # ── 1. Retrieve relevant memories ─────────────
        memory_context = build_memory_context(question, user_id, self.store)

        # ── 2. Augment question ────────────────────────
        augmented = question
        if memory_context:
            augmented = f"{memory_context}\n\nCurrent question: {question}"

        # ── 3. Track for session-end extraction ───────
        if user_id not in self._sessions:
            self._sessions[user_id] = []
        self._sessions[user_id].append({"role": "user", "content": question})

        # ── 4. Invoke base agent ───────────────────────
        result = self.agent.invoke({"question": augmented}, **kwargs)

        # ── 5. Track agent response ────────────────────
        answer = result.get("answer", "")
        if answer:
            self._sessions[user_id].append({"role": "assistant", "content": answer})

        result["memory_context_used"] = bool(memory_context)
        return result

    # ────────────────────────────────────────────────
    # STREAMING VARIANT
    # ────────────────────────────────────────────────

    async def astream(self, question: str, user_id: str = "default", **kwargs):
        """
        Async streaming version — use with your FastAPI SSE endpoint.
        Injects memory then streams tokens from the base agent.
        """
        memory_context = build_memory_context(question, user_id, self.store)
        augmented = f"{memory_context}\n\nCurrent question: {question}" if memory_context else question

        if user_id not in self._sessions:
            self._sessions[user_id] = []
        self._sessions[user_id].append({"role": "user", "content": question})

        full_answer = []
        async for token in self.agent.astream({"question": augmented}, **kwargs):
            full_answer.append(str(token))
            yield token

        if full_answer:
            self._sessions[user_id].append({
                "role":    "assistant",
                "content": "".join(full_answer),
            })

    # ────────────────────────────────────────────────
    # END SESSION (extract & store memories)
    # ────────────────────────────────────────────────

    def end_session(self, user_id: str = "default") -> dict:
        """
        Extract and persist memories from the completed session.
        Call this when:
          - User logs out
          - Session timeout
          - After POST /session/end API endpoint

        Returns:
            {"episodic_stored": 1, "semantic_stored": 3}
        """
        messages = self._sessions.get(user_id, [])
        if len(messages) < 2:
            self._sessions.pop(user_id, None)
            return {"episodic_stored": 0, "semantic_stored": 0}

        # Get existing facts to avoid duplicates
        existing = self.store.get_all_semantic(user_id=user_id)

        # Extract memories via LLM
        extracted = self.extractor.process_session(messages, existing)

        counts = {"episodic_stored": 0, "semantic_stored": 0}

        # Store episode
        episode = extracted.get("episode")
        if episode and episode.get("summary"):
            self.store.store_episodic(
                summary=episode["summary"],
                topics=episode.get("topics", []),
                user_id=user_id,
                message_count=episode.get("message_count", 0),
            )
            counts["episodic_stored"] = 1

        # Store semantic facts
        for fact_item in extracted.get("new_facts", []):
            if fact_item.get("fact") and float(fact_item.get("confidence", 0)) > 0.5:
                self.store.store_semantic(
                    fact=fact_item["fact"],
                    fact_type=fact_item.get("fact_type", "fact"),
                    confidence=float(fact_item.get("confidence", 0.8)),
                    user_id=user_id,
                )
                counts["semantic_stored"] += 1

        # Clear session buffer
        self._sessions.pop(user_id, None)

        print(f"[MemoryEnabledAgent] Session ended for '{user_id}': {counts}")
        return counts

    # ────────────────────────────────────────────────
    # DEBUGGING / DISPLAY HELPERS
    # ────────────────────────────────────────────────

    def get_user_memories(self, user_id: str = "default") -> dict:
        """
        Retrieve all memories for a user.
        Useful for a /memories endpoint in your API.
        """
        return {
            "semantic_facts": self.store.get_all_semantic(user_id),
            "recent_episodes": self.store.retrieve_episodic("", user_id=user_id, top_k=10),
            "stats": self.store.memory_summary(user_id),
        }

    def clear_user_memories(self, user_id: str = "default") -> None:
        """
        Wipe all memories for a user (GDPR compliance / testing).
        Only supported with file-based storage currently.
        """
        if not self.store._use_weaviate:
            self.store._db["EpisodicMemory"] = [
                m for m in self.store._db.get("EpisodicMemory", [])
                if m.get("user_id") != user_id
            ]
            self.store._db["SemanticMemory"] = [
                m for m in self.store._db.get("SemanticMemory", [])
                if m.get("user_id") != user_id
            ]
            self.store._save_file()
            print(f"[MemoryEnabledAgent] Cleared all memories for '{user_id}'")
        else:
            print("[MemoryEnabledAgent] Weaviate delete-by-filter not yet implemented. "
                  "Use Weaviate console to delete objects manually.")
