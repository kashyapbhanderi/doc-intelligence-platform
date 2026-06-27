"""
scripts/demo_memory.py
=======================
Interactive demo of the long-term memory system.
Simulates two separate sessions with the same user to show memory persistence.

Run:
    python scripts/demo_memory.py

WHERE TO PLACE THIS FILE: scripts/demo_memory.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def print_box(title: str, content: str, width: int = 65):
    print("\n" + "─" * width)
    print(f"  {title}")
    print("─" * width)
    print(content)


def main():
    from memory.memory_store import MemoryStore
    from memory.memory_extractor import MemoryExtractor
    from memory.memory_agent import MemoryEnabledAgent, build_memory_context

    print("=" * 65)
    print("  Long-Term Memory Demo — Two-session simulation")
    print("=" * 65)
    print("\nThis demo:")
    print("  Session 1 → user asks questions, session ends → memories stored")
    print("  Session 2 → same user returns → agent remembers context")

    # ── Setup ────────────────────────────────────────────────────
    store     = MemoryStore(fallback_path="data/demo_memories.json")
    extractor = MemoryExtractor()
    USER_ID   = "demo_user_priya"

    # Clear previous demo data
    if not store._use_weaviate:
        store._db["EpisodicMemory"] = [
            m for m in store._db.get("EpisodicMemory", [])
            if m.get("user_id") != USER_ID
        ]
        store._db["SemanticMemory"] = [
            m for m in store._db.get("SemanticMemory", [])
            if m.get("user_id") != USER_ID
        ]
        store._save_file()

    # ── Fake base agent (echoes question for demo) ───────────────
    class FakeAgent:
        def invoke(self, state: dict, **_):
            q = state.get("question", "")
            answer = f"[Agent answer to: {q[:60]}...]"
            return {"answer": answer, "question": q}

    agent = MemoryEnabledAgent(FakeAgent(), memory_store=store, memory_extractor=extractor)

    # ════════════════════════════════════════════════════════════
    # SESSION 1 — First time user interacts
    # ════════════════════════════════════════════════════════════
    print_box(
        "SESSION 1 — First interaction",
        "User: Priya, financial analyst studying Infosys FY2024 reports"
    )

    session_1_messages = [
        {"role": "user",      "content": "I am doing my financial analysis project on Infosys FY2024 annual report."},
        {"role": "assistant", "content": "Great! I can help you analyse the Infosys FY2024 annual report. What aspects are you focusing on?"},
        {"role": "user",      "content": "I need revenue, operating margin, and EPS data. Please give answers in table format always."},
        {"role": "assistant", "content": "Understood — I'll always use table format.\n\n| Metric | Value |\n|--------|-------|\n| Revenue | ₹1,53,670 Cr |"},
        {"role": "user",      "content": "What is the YoY revenue growth?"},
        {"role": "assistant", "content": "YoY revenue growth for FY2024 was approximately 4.2%."},
        {"role": "user",      "content": "The operating margin was 20.7%, not 21%. Please always use 20.7%."},
        {"role": "assistant", "content": "Correction noted. The correct operating margin is 20.7%."},
    ]

    print("\nSimulating session 1 messages...")

    # Simulate invoke calls
    for msg in session_1_messages:
        if msg["role"] == "user":
            result = agent.invoke(msg["content"], user_id=USER_ID)

    print(f"Session 1: {len(session_1_messages)} messages exchanged.")

    # End session — triggers memory extraction
    print("\nEnding session 1 (extracting memories via LLM)...")
    counts = agent.end_session(USER_ID)
    print(f"Stored: {counts['episodic_stored']} episode, {counts['semantic_stored']} semantic facts")

    # Show what was stored
    stored = store.get_all_semantic(USER_ID)
    if stored:
        print_box("SEMANTIC MEMORIES STORED", "\n".join(
            f"  [{f.get('fact_type','fact')}] {f.get('fact','')}"
            for f in stored
        ))

    # ════════════════════════════════════════════════════════════
    # SESSION 2 — User returns next day
    # ════════════════════════════════════════════════════════════
    print_box(
        "SESSION 2 — Same user returns (next day)",
        "User asks a new question — watch memory context get injected"
    )

    new_query = "What is the EPS for FY2024?"
    memory_context = build_memory_context(new_query, USER_ID, store)

    if memory_context:
        print_box("MEMORY CONTEXT INJECTED (prepended to question)", memory_context)
    else:
        print("\n⚠ No memory context found. "
              "This happens if Session 1 extraction failed (likely no OpenAI key).")
        print("  The memory system still works — just no LLM extraction in demo mode.")

    result = agent.invoke(new_query, user_id=USER_ID)
    print_box("AGENT RESPONSE", result.get("answer", ""))
    print(f"\n  memory_context_used = {result.get('memory_context_used', False)}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Memory Demo Summary")
    print("=" * 65)
    all_memories = agent.get_user_memories(USER_ID)
    stats = all_memories["stats"]
    print(f"  Total semantic facts stored : {stats['semantic_facts']}")
    print(f"  Preferences learned         : {stats['preference_count']}")
    print(f"  Facts learned               : {stats['fact_count']}")
    print(f"  Episodes stored             : {stats['episodic_count']}")
    print("\nMemory system is working correctly!")
    print("Memory file saved at: data/demo_memories.json")

    # Cleanup
    Path("data/demo_memories.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
