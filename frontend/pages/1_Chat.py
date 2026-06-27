"""
frontend/pages/1_Chat.py
Chat interface — full Planner → Executor → Critic pipeline
"""
import streamlit as st
import requests
import time
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Chat — Doc Intelligence",
    page_icon="💬",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    #MainMenu, footer, header { visibility: hidden; }
    .chat-user {
        background: #2d3748; color: #e2e8f0;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0; max-width: 75%;
        margin-left: auto; font-size: 0.95rem;
    }
    .chat-bot {
        background: #1a365d; color: #bee3f8;
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 10px 0; max-width: 85%;
        font-size: 0.95rem; line-height: 1.6;
    }
    .source-chip {
        display: inline-block;
        background: #2a4365; color: #90cdf4;
        padding: 4px 12px; border-radius: 12px;
        font-size: 0.75rem; margin: 4px 4px 0 0;
        border: 1px solid #3182ce;
    }
    .agent-step {
        background: #1a1f2e;
        border-left: 3px solid #63b3ed;
        padding: 8px 14px; margin: 6px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.83rem; color: #a0aec0;
    }
    .faithful-badge {
        background: #276749; color: #9ae6b4;
        padding: 3px 12px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600;
    }
    .unfaithful-badge {
        background: #742a2a; color: #feb2b2;
        padding: 3px 12px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_pipeline" not in st.session_state:
    st.session_state.show_pipeline = True

# ── Header ────────────────────────────────────────────────
col_title, col_controls = st.columns([3, 1])
with col_title:
    st.markdown("""
    <h2 style='color:#e2e8f0; margin:0;'>
        💬 Intelligent Q&amp;A
    </h2>
    <p style='color:#718096; font-size:0.9rem; margin:4px 0 0 0;'>
        3-agent pipeline: Planner → Executor → Critic
    </p>
    """, unsafe_allow_html=True)

with col_controls:
    st.session_state.show_pipeline = st.toggle(
        "Show agent steps", value=True)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.divider()

# ── Suggested questions ───────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        "<p style='color:#a0aec0; font-size:0.9rem;'>"
        "✨ Try one of these questions:</p>",
        unsafe_allow_html=True)

    suggestions = [
        "What is retrieval augmented generation?",
        "How does LoRA reduce memory during fine-tuning?",
        "Compare dense and sparse retrieval methods",
        "What is chain of thought prompting?",
        "How does RLHF improve language models?",
        "What are the components of a RAG system?",
    ]
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(
                suggestion[:45] + "…"
                if len(suggestion) > 45 else suggestion,
                key=f"sug_{i}",
                use_container_width=True
            ):
                st.session_state.pending_question = suggestion
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

# ── Render existing messages ──────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='chat-user'>{msg['content']}</div>",
            unsafe_allow_html=True)
    else:
        data = msg["data"]
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        sub_q  = data.get("sub_queries", [])
        faithful = data.get("is_faithful", True)
        latency  = data.get("latency_seconds", 0)

        # Agent steps
        if st.session_state.show_pipeline and sub_q:
            with st.expander(
                f"🔍 Agent pipeline "
                f"({len(sub_q)} sub-queries, "
                f"{latency}s)",
                expanded=False
            ):
                st.markdown(
                    "<div class='agent-step'>"
                    "🗺️ <b>Planner</b> — decomposed question"
                    "</div>",
                    unsafe_allow_html=True)
                for q in sub_q:
                    st.markdown(
                        f"<div class='agent-step'>"
                        f"&nbsp;&nbsp;&nbsp;→ {q}</div>",
                        unsafe_allow_html=True)
                st.markdown(
                    f"<div class='agent-step'>"
                    f"⚙️ <b>Executor</b> — retrieved "
                    f"{len(sources)} sources</div>",
                    unsafe_allow_html=True)
                badge = (
                    "<span class='faithful-badge'>"
                    "✅ Faithful</span>"
                    if faithful else
                    "<span class='unfaithful-badge'>"
                    "⚠️ Flagged</span>"
                )
                st.markdown(
                    f"<div class='agent-step'>"
                    f"🔍 <b>Critic</b> — {badge}</div>",
                    unsafe_allow_html=True)

        # Answer bubble
        st.markdown(
            f"<div class='chat-bot'>{answer}</div>",
            unsafe_allow_html=True)

        # Sources
        if sources:
            chips = "".join(
                f"<span class='source-chip'>"
                f"📄 {s['source']} p{s['page']}</span>"
                for s in sources[:5]
            )
            st.markdown(
                f"<div style='margin:6px 0 16px 0;'>"
                f"{chips}</div>",
                unsafe_allow_html=True)

# ── Input box ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

# Handle suggestion clicks
default_q = st.session_state.pop(
    "pending_question", "")

with st.form("chat_form", clear_on_submit=True):
    cols = st.columns([6, 1])
    with cols[0]:
        question = st.text_input(
            "Ask anything about your documents…",
            value=default_q,
            label_visibility="collapsed",
            placeholder="e.g. What is the attention mechanism?")
    with cols[1]:
        submitted = st.form_submit_button(
            "Send ➤", use_container_width=True,
            type="primary")

# ── Handle submission ─────────────────────────────────────
if submitted and question.strip():
    st.session_state.messages.append(
        {"role": "user", "content": question})

    with st.spinner(
        "🤔 Planner → Executor → Critic…"
    ):
        try:
            start = time.time()
            r = requests.post(
                f"{API_URL}/api/v1/query",
                json={"question": question,
                      "top_k": 5},
                timeout=60
            )
            elapsed = round(time.time() - start, 1)

            if r.status_code == 200:
                data = r.json()
                data["latency_seconds"] = elapsed
                st.session_state.messages.append(
                    {"role": "assistant", "data": data})
            else:
                error_data = {
                    "answer": (
                        f"⚠️ API error "
                        f"(HTTP {r.status_code}). "
                        f"Is the backend running at "
                        f"`{API_URL}`?"),
                    "sources": [],
                    "sub_queries": [],
                    "is_faithful": False,
                    "latency_seconds": elapsed
                }
                st.session_state.messages.append(
                    {"role": "assistant",
                     "data": error_data})

        except requests.Timeout:
            st.session_state.messages.append({
                "role": "assistant",
                "data": {
                    "answer": (
                        "⏱️ Request timed out (>60s). "
                        "The LLM might be slow. Try again."),
                    "sources": [], "sub_queries": [],
                    "is_faithful": False,
                    "latency_seconds": 60
                }
            })
        except requests.ConnectionError:
            st.session_state.messages.append({
                "role": "assistant",
                "data": {
                    "answer": (
                        f"❌ Cannot connect to API at "
                        f"`{API_URL}`. "
                        f"Run `docker-compose up -d` first."),
                    "sources": [], "sub_queries": [],
                    "is_faithful": False,
                    "latency_seconds": 0
                }
            })

    st.rerun()