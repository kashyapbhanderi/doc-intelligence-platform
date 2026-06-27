"""
frontend/pages/5_Monitor.py
Live system monitoring — polls Prometheus directly,
embeds Grafana dashboard.
"""
import streamlit as st
import requests
import os
import time

API_URL        = os.getenv("API_URL", "http://localhost:8000")
PROMETHEUS_URL = os.getenv(
    "PROMETHEUS_URL", "http://localhost:9090")
GRAFANA_URL    = os.getenv(
    "GRAFANA_URL", "http://localhost:3000")

st.set_page_config(
    page_title="Monitor — Doc Intelligence",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    #MainMenu, footer, header { visibility: hidden; }
    .status-pill {
        display: inline-flex; align-items: center;
        gap: 6px; padding: 6px 14px;
        border-radius: 20px; font-size: 0.85rem;
        font-weight: 600;
    }
    .pill-up   { background: #276749; color: #9ae6b4; }
    .pill-down { background: #742a2a; color: #feb2b2; }
    .metric-box {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 18px; text-align: center;
    }
    .metric-box .val {
        font-size: 1.9rem; font-weight: 700;
        color: #63b3ed;
    }
    .metric-box .lbl {
        font-size: 0.78rem; color: #a0aec0;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

col_t, col_r = st.columns([4, 1])
with col_t:
    st.markdown("""
    <h2 style='color:#e2e8f0; margin:0;'>📡 Live Monitor</h2>
    <p style='color:#718096; font-size:0.9rem; margin:4px 0 0 0;'>
        Real-time system health via Prometheus
    </p>
    """, unsafe_allow_html=True)
with col_r:
    auto_refresh = st.toggle("Auto-refresh (5s)", value=False)


def query_prom(promql: str):
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": promql}, timeout=3
        )
        results = r.json().get(
            "data", {}).get("result", [])
        return float(results[0]["value"][1]) \
            if results else 0.0
    except Exception:
        return None


def get_health():
    try:
        r = requests.get(
            f"{API_URL}/api/v1/health", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


st.divider()

# ── Service status row ────────────────────────────────────
health  = get_health()
api_ok  = health.get("status") == "healthy"
wv_ok   = health.get("weaviate") == "healthy"
prom_ok = query_prom("up") is not None

c1, c2, c3 = st.columns(3)
with c1:
    pill = "pill-up" if api_ok else "pill-down"
    icon = "🟢" if api_ok else "🔴"
    st.markdown(
        f"<span class='status-pill {pill}'>"
        f"{icon} FastAPI</span>",
        unsafe_allow_html=True)
with c2:
    pill = "pill-up" if wv_ok else "pill-down"
    icon = "🟢" if wv_ok else "🔴"
    st.markdown(
        f"<span class='status-pill {pill}'>"
        f"{icon} Weaviate</span>",
        unsafe_allow_html=True)
with c3:
    pill = "pill-up" if prom_ok else "pill-down"
    icon = "🟢" if prom_ok else "🔴"
    st.markdown(
        f"<span class='status-pill {pill}'>"
        f"{icon} Prometheus</span>",
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Metrics grid ───────────────────────────────────────────
if prom_ok:
    total_req  = query_prom("sum(http_requests_total)") or 0
    req_rate   = query_prom(
        "sum(rate(http_requests_total[1m]))") or 0
    chunks     = query_prom("weaviate_chunks_total") or 0
    queries_ok = query_prom(
        'rag_queries_total{status="success"}') or 0
    queries_err = query_prom(
        'rag_queries_total{status="error"}') or 0
    faithful   = query_prom("faithful_answers_total") or 0
    unfaithful = query_prom(
        "unfaithful_answers_total") or 0
    active_q   = query_prom("active_rag_queries") or 0

    row1 = st.columns(4)
    with row1[0]:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='val'>{int(total_req):,}</div>
            <div class='lbl'>Total Requests</div>
        </div>""", unsafe_allow_html=True)
    with row1[1]:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='val'>{req_rate:.2f}</div>
            <div class='lbl'>Requests / sec</div>
        </div>""", unsafe_allow_html=True)
    with row1[2]:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='val'>{int(chunks):,}</div>
            <div class='lbl'>Chunks in Weaviate</div>
        </div>""", unsafe_allow_html=True)
    with row1[3]:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='val'>{int(active_q)}</div>
            <div class='lbl'>Active Queries</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    row2 = st.columns(3)

    total_q = queries_ok + queries_err
    success_pct = (
        queries_ok / total_q * 100 if total_q else 0)
    total_f = faithful + unfaithful
    faith_pct = (
        faithful / total_f * 100 if total_f else 0)

    with row2[0]:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='val'>{success_pct:.0f}%</div>
            <div class='lbl'>
                Query Success Rate<br>
                ({int(queries_ok)} ok / {int(queries_err)} err)
            </div>
        </div>""", unsafe_allow_html=True)
    with row2[1]:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='val'>{faith_pct:.0f}%</div>
            <div class='lbl'>
                Faithfulness Rate<br>
                ({int(faithful)} faithful / {int(unfaithful)} flagged)
            </div>
        </div>""", unsafe_allow_html=True)
    with row2[2]:
        p95 = query_prom(
            "histogram_quantile(0.95, "
            "rate(http_request_duration_seconds_bucket[5m]))"
        ) or 0
        st.markdown(f"""
        <div class='metric-box'>
            <div class='val'>{p95*1000:.0f}ms</div>
            <div class='lbl'>P95 Latency</div>
        </div>""", unsafe_allow_html=True)

else:
    st.warning(
        f"⚠️ Cannot reach Prometheus at "
        f"`{PROMETHEUS_URL}`. "
        f"Start the stack with `docker-compose up -d`.")

# ── Embedded Grafana ───────────────────────────────────────
st.divider()
st.markdown("### 📈 Full Grafana Dashboard")
st.caption(
    f"Live dashboard embedded below. "
    f"Open directly: {GRAFANA_URL}")

try:
    st.components.v1.iframe(
        f"{GRAFANA_URL}/d/rag-monitor/"
        f"doc-intelligence-platform"
        f"?orgId=1&refresh=10s&theme=dark",
        height=600
    )
except Exception:
    st.info(
        f"Grafana not reachable. "
        f"Open manually: {GRAFANA_URL}")

# ── Auto-refresh ───────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()