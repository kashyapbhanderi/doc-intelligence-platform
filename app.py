"""
frontend/app.py
Doc Intelligence Platform — Web Interface
Multi-page Streamlit app connecting to FastAPI backend.

Run locally:
  streamlit run frontend/app.py

Deploy free:
  streamlit cloud → connect GitHub repo → set API_URL secret
"""
import streamlit as st
import requests
import os

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Doc Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%);
        border-right: 1px solid #2d3748;
    }

    /* Cards */
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #63b3ed;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 4px;
    }

    /* Status badges */
    .badge-green {
        background: #276749; color: #9ae6b4;
        padding: 3px 10px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600;
    }
    .badge-red {
        background: #742a2a; color: #feb2b2;
        padding: 3px 10px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600;
    }

    /* Feature cards on home */
    .feature-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        border-left: 4px solid #63b3ed;
    }
    .feature-card h3 { color: #e2e8f0; margin: 0 0 8px 0; }
    .feature-card p  { color: #a0aec0; margin: 0; font-size: 0.9rem; }

    /* Chat bubbles */
    .chat-user {
        background: #2d3748; color: #e2e8f0;
        padding: 12px 16px; border-radius: 18px 18px 4px 18px;
        margin: 8px 0; max-width: 80%; margin-left: auto;
        font-size: 0.95rem;
    }
    .chat-bot {
        background: #1a365d; color: #bee3f8;
        padding: 12px 16px; border-radius: 18px 18px 18px 4px;
        margin: 8px 0; max-width: 85%;
        font-size: 0.95rem; line-height: 1.6;
    }
    .source-chip {
        display: inline-block;
        background: #2a4365; color: #90cdf4;
        padding: 3px 10px; border-radius: 12px;
        font-size: 0.75rem; margin: 3px;
    }

    /* Hide default Streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── API URL ───────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

@st.cache_data(ttl=30)
def get_health():
    try:
        r = requests.get(
            f"{API_URL}/api/v1/health", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <span style='font-size:2.5rem;'>🧠</span>
        <h2 style='color:#e2e8f0; margin:8px 0 4px 0;
                   font-size:1.1rem;'>
            Doc Intelligence
        </h2>
        <p style='color:#718096; font-size:0.8rem; margin:0;'>
            Agentic RAG Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Live health check
    health = get_health()
    is_up  = health.get("status") == "healthy"
    wv_ok  = health.get("weaviate") == "healthy"
    chunks = health.get("total_chunks", 0)

    st.markdown("**System Status**")
    col1, col2 = st.columns(2)
    with col1:
        badge = "badge-green" if is_up else "badge-red"
        text  = "Online" if is_up else "Offline"
        st.markdown(
            f'<span class="{badge}">API {text}</span>',
            unsafe_allow_html=True)
    with col2:
        badge2 = "badge-green" if wv_ok else "badge-red"
        text2  = "Healthy" if wv_ok else "Down"
        st.markdown(
            f'<span class="{badge2}">DB {text2}</span>',
            unsafe_allow_html=True)

    st.caption(f"📚 {chunks:,} chunks indexed")
    st.divider()

    st.markdown("**Navigation**")
    st.page_link("app.py",             label="🏠 Home",      icon="🏠")
    st.page_link("pages/1_Chat.py",    label="💬 Chat",      icon="💬")
    st.page_link("pages/2_Upload.py",  label="📤 Upload",    icon="📤")
    st.page_link("pages/3_Editor.py",  label="✏️ Editor",   icon="✏️")
    st.page_link("pages/4_Analytics.py", label="📊 Analytics", icon="📊")
    st.page_link("pages/5_Monitor.py", label="📡 Monitor",   icon="📡")

    st.divider()
    st.caption(f"API: `{API_URL}`")
    st.caption("v1.0.0 · Week 8 Complete")


# ── Home page ─────────────────────────────────────────────
st.markdown("""
<h1 style='color:#e2e8f0; margin-bottom:4px;'>
    🧠 Doc Intelligence Platform
</h1>
<p style='color:#718096; font-size:1rem; margin-bottom:28px;'>
    Production-grade Multimodal Agentic RAG with
    Document Editing — 52 AI/ML papers indexed
</p>
""", unsafe_allow_html=True)

# Stats row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='value'>{chunks:,}</div>
        <div class='label'>Chunks Indexed</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class='metric-card'>
        <div class='value'>52</div>
        <div class='label'>Research Papers</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class='metric-card'>
        <div class='value'>3</div>
        <div class='label'>AI Agents</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class='metric-card'>
        <div class='value'>12</div>
        <div class='label'>Editor Tools</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Feature cards
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    <div class='feature-card'>
        <h3>💬 Intelligent Q&A</h3>
        <p>Ask any question about your documents.
        Planner → Executor → Critic agents deliver
        verified, source-cited answers.</p>
    </div>

    <div class='feature-card'>
        <h3>✏️ Document Editor</h3>
        <p>Edit Word docs, watermark PDFs, resize images
        using natural language instructions.
        12 AI-powered tools.</p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class='feature-card'>
        <h3>📤 Smart Ingestion</h3>
        <p>Upload PDFs, Word docs, or images.
        OCR + GPT-4o Vision extracts text, tables,
        charts and diagrams automatically.</p>
    </div>

    <div class='feature-card'>
        <h3>📊 Full Analytics</h3>
        <p>RAGAS evaluation scores, NDCG@10 retrieval
        metrics, MLflow experiment history, and live
        Prometheus monitoring.</p>
    </div>
    """, unsafe_allow_html=True)

# Quick-start
st.markdown("---")
st.markdown("### 🚀 Quick Start")
q1, q2, q3 = st.columns(3)
with q1:
    if st.button("💬 Ask a Question",
                 use_container_width=True,
                 type="primary"):
        st.switch_page("pages/1_Chat.py")
with q2:
    if st.button("📤 Upload Document",
                 use_container_width=True):
        st.switch_page("pages/2_Upload.py")
with q3:
    if st.button("✏️ Edit a Document",
                 use_container_width=True):
        st.switch_page("pages/3_Editor.py")