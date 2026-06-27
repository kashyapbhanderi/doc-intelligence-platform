"""
frontend/pages/2_Upload.py
Document upload + ingestion with live task tracking.
"""
import streamlit as st
import requests
import time
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Upload — Doc Intelligence",
    page_icon="📤",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    #MainMenu, footer, header { visibility: hidden; }
    .upload-zone {
        background: #1a1f2e;
        border: 2px dashed #2d3748;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
    }
    .task-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-left: 4px solid #63b3ed;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .task-card.done   { border-left-color: #48bb78; }
    .task-card.failed { border-left-color: #f56565; }
    .task-card.pending,
    .task-card.processing { border-left-color: #ed8936; }
    .status-chip {
        padding: 2px 10px; border-radius: 10px;
        font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase;
    }
    .status-done       { background:#276749; color:#9ae6b4; }
    .status-failed      { background:#742a2a; color:#feb2b2; }
    .status-processing { background:#7b341e; color:#fbd38d; }
    .status-pending     { background:#2d3748; color:#cbd5e0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style='color:#e2e8f0; margin:0;'>📤 Upload Documents</h2>
<p style='color:#718096; font-size:0.9rem; margin:4px 0 0 0;'>
    PDF, DOCX, or images — OCR + Vision extraction +
    automatic chunking + embedding
</p>
""", unsafe_allow_html=True)
st.divider()

col_upload, col_options = st.columns([2, 1])

with col_upload:
    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=["pdf", "docx", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

with col_options:
    st.markdown("**Processing options**")
    use_vision = st.checkbox(
        "🔍 Use GPT-4o Vision",
        value=False,
        help="Extracts charts, tables, and diagrams. "
             "Costs ~$0.01-0.05 per page.")
    auto_embed = st.checkbox(
        "⚡ Auto-embed after extraction",
        value=True,
        help="Immediately index into Weaviate "
             "so it's searchable in Chat.")

if "upload_tasks" not in st.session_state:
    st.session_state.upload_tasks = []

if uploaded_files:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(
        f"🚀 Process {len(uploaded_files)} file(s)",
        type="primary", use_container_width=True
    ):
        for file in uploaded_files:
            try:
                files_payload = {
                    "file": (file.name, file.getvalue(),
                            file.type)
                }
                r = requests.post(
                    f"{API_URL}/api/v1/ingest",
                    files=files_payload,
                    timeout=30
                )
                if r.status_code == 200:
                    data = r.json()
                    st.session_state.upload_tasks.insert(
                        0, {
                            "filename": file.name,
                            "task_id": data.get("task_id", ""),
                            "status": "pending"
                        })
                    st.toast(
                        f"✅ {file.name} queued",
                        icon="📤")
                else:
                    st.toast(
                        f"❌ {file.name} failed: "
                        f"HTTP {r.status_code}",
                        icon="⚠️")
            except requests.ConnectionError:
                st.error(
                    f"Cannot reach API at {API_URL}. "
                    f"Is the backend running?")
                break
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ── Task tracking ──────────────────────────────────────────
if st.session_state.upload_tasks:
    col_h, col_r = st.columns([4, 1])
    with col_h:
        st.markdown("### 📋 Processing Queue")
    with col_r:
        if st.button("🔄 Refresh status",
                     use_container_width=True):
            st.rerun()

    for task in st.session_state.upload_tasks:
        task_id = task["task_id"]
        try:
            r = requests.get(
                f"{API_URL}/api/v1/ingest/tasks/{task_id}",
                timeout=5)
            if r.status_code == 200:
                live = r.json()
                task["status"] = live.get(
                    "status", task["status"])
                task["chunks"] = live.get("chunks", 0)
                task["elapsed"] = live.get("elapsed", 0)
                task["error"]   = live.get("error", "")
        except Exception:
            pass

        status = task.get("status", "pending")
        chunks = task.get("chunks", 0)
        elapsed = task.get("elapsed", 0)

        icon = {
            "pending":    "⏳",
            "processing": "⚙️",
            "done":       "✅",
            "failed":     "❌"
        }.get(status, "⏳")

        st.markdown(f"""
        <div class='task-card {status}'>
            <div style='display:flex; justify-content:space-between;
                       align-items:center;'>
                <div>
                    <span style='font-size:1.1rem;'>{icon}</span>
                    <b style='color:#e2e8f0; margin-left:8px;'>
                        {task['filename']}
                    </b>
                </div>
                <span class='status-chip status-{status}'>
                    {status}
                </span>
            </div>
            <div style='color:#718096; font-size:0.8rem;
                       margin-top:6px;'>
                {f"📚 {chunks} chunks indexed · " if chunks else ""}
                ⏱️ {elapsed}s elapsed
                {f" · ⚠️ {task.get('error','')}"
                  if status == "failed" else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Auto-refresh while anything still processing
    active = any(
        t.get("status") in ("pending", "processing")
        for t in st.session_state.upload_tasks
    )
    if active:
        time.sleep(3)
        st.rerun()

else:
    st.markdown("""
    <div class='upload-zone'>
        <span style='font-size:3rem;'>📂</span>
        <p style='color:#a0aec0; margin-top:12px;'>
            No files uploaded yet.<br>
            Drag PDFs, Word docs, or images above to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Currently indexed docs ────────────────────────────────
st.divider()
st.markdown("### 📚 Currently Indexed")
try:
    r = requests.get(
        f"{API_URL}/api/v1/ingest/status", timeout=5)
    if r.status_code == 200:
        data = r.json()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Chunks",
                     f"{data.get('total_chunks', 0):,}")
        with c2:
            st.metric("Status",
                     data.get("status", "unknown").title())
except Exception:
    st.warning("Could not fetch index status.")