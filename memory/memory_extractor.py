"""
Memory — memory_extractor.py
=============================
Uses GPT-4o-mini to distil two types of memory from a conversation:

1. Episode summary — what happened in this session (2–3 sentences)
2. Semantic facts  — reusable facts, preferences, corrections to store long-term

This is called ONCE at the end of each session by MemoryEnabledAgent.end_session().

WHERE TO PLACE THIS FILE: memory/memory_extractor.py
"""

import json
from typing import Optional
from openai import OpenAI


class MemoryExtractor:
    """
    Extracts structured memories from a conversation history.

    Input:  list of {"role": "user"|"assistant", "content": "..."} dicts
    Output: {episode: dict, new_facts: list[dict]}
    """

   
    def __init__(self, model: str = None):
        from config.llm_config import get_llm_config
        cfg = get_llm_config()
        self.client = OpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"]
        )
        self.model = model or cfg["model"]

    def _call(self, prompt: str, max_tokens: int = 500) -> str:
        """Thin wrapper around OpenAI chat completion."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if the LLM adds them
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        return raw

    # ────────────────────────────────────────────────
    # EPISODE SUMMARY
    # ────────────────────────────────────────────────

    def extract_episode_summary(
        self,
        messages:     list[dict],
        max_messages: int = 20,
    ) -> dict:
        """
        Summarise a conversation into an episodic memory entry.

        Returns:
            {
              "summary":       "2-3 sentence summary",
              "topics":        ["topic1", "topic2"],
              "key_documents": ["doc names mentioned"],
              "user_intent":   "what the user was trying to achieve",
              "message_count": N
            }
        """
        recent = messages[-max_messages:]
        convo  = "\n".join(
            f"{m.get('role','user').upper()}: {str(m.get('content',''))[:300]}"
            for m in recent
        )

        prompt = f"""Summarise this conversation into a concise episodic memory for an AI assistant.

CONVERSATION:
{convo}

Return ONLY valid JSON with these exact keys — no explanation:
{{
  "summary":       "2-3 sentence summary of what was discussed and what was found or accomplished",
  "topics":        ["up to 4 topic keywords"],
  "key_documents": ["names of documents or files mentioned, if any"],
  "user_intent":   "one sentence — what was the user ultimately trying to do"
}}"""

        try:
            result = json.loads(self._call(prompt))
            result["message_count"] = len(messages)
            return result
        except Exception:
            return {
                "summary":       f"Conversation with {len(messages)} messages.",
                "topics":        [],
                "key_documents": [],
                "user_intent":   "unknown",
                "message_count": len(messages),
            }

    # ────────────────────────────────────────────────
    # SEMANTIC FACTS
    # ────────────────────────────────────────────────

    def extract_semantic_facts(
        self,
        messages:       list[dict],
        existing_facts: Optional[list[dict]] = None,
        max_messages:   int = 20,
    ) -> list[dict]:
        """
        Extract durable, reusable facts from the conversation.
        Skips facts that are already known (dedup via `existing_facts`).

        Returns list of:
            {"fact": "...", "fact_type": "preference|fact|context|correction", "confidence": 0.0-1.0}

        Rules applied in the prompt:
          - Only extract explicitly stated or strongly implied facts
          - Do NOT include transient / one-time questions
          - Max 5 facts per conversation to keep signal-to-noise high
          - Corrections are prioritised (confidence = 1.0)
        """
        recent = messages[-max_messages:]
        convo  = "\n".join(
            f"{m.get('role','user').upper()}: {str(m.get('content',''))[:300]}"
            for m in recent
        )

        existing_str = ""
        if existing_facts:
            existing_str = "ALREADY KNOWN (do NOT repeat these):\n" + "\n".join(
                f"- {f.get('fact','')}" for f in existing_facts[:15]
            ) + "\n\n"

        prompt = f"""Extract long-term facts worth remembering from this conversation.

WHAT TO EXTRACT:
  - User preferences (format, detail level, language)
  - Domain context (what documents / domain the user works in)
  - Corrections (user corrected a previous answer)
  - Important facts explicitly stated

WHAT TO SKIP:
  - Transient questions that won't recur
  - Greetings, acknowledgements
  - Vague or uncertain statements

{existing_str}CONVERSATION:
{convo}

Return ONLY a valid JSON array. Each element:
{{"fact": "specific concrete statement", "fact_type": "preference|fact|context|correction", "confidence": 0.5-1.0}}

Rules:
- Return [] if no new facts worth storing
- Maximum 5 facts
- Corrections always get confidence 1.0
- JSON array only, no explanation:"""

        try:
            result = json.loads(self._call(prompt, max_tokens=600))
            return result if isinstance(result, list) else []
        except Exception:
            return []

    # ────────────────────────────────────────────────
    # COMBINED SESSION PROCESSING
    # ────────────────────────────────────────────────

    def process_session(
        self,
        messages:       list[dict],
        existing_facts: Optional[list[dict]] = None,
    ) -> dict:
        """
        Full session processing — call this ONCE at session end.

        Args:
            messages       : full conversation history for this session
            existing_facts : already-stored semantic facts (for dedup)

        Returns:
            {
              "episode":    { summary, topics, key_documents, user_intent, message_count },
              "new_facts":  [ {fact, fact_type, confidence}, ... ]
            }
        """
        if len(messages) < 2:
            return {"episode": None, "new_facts": []}

        episode   = self.extract_episode_summary(messages)
        new_facts = self.extract_semantic_facts(messages, existing_facts)

        print(f"[MemoryExtractor] Episode: {episode.get('summary','')[:80]}…")
        print(f"[MemoryExtractor] {len(new_facts)} new fact(s) extracted")

        return {"episode": episode, "new_facts": new_facts}
