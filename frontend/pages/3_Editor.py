"""
frontend/pages/3_Editor.py
Document editor — natural language instructions
trigger python-docx / PyMuPDF / Pillow tools.
"""
import streamlit as st
import requests
import os

API_URL    = os.getenv("API_URL", "http://localhost:8000")
SERVER_DIR = os.getenv("SERVER_FILES_DIR", "data/test_docs")

st.set_page_config(
    page_title="Editor — Doc Intelligence",
    page_icon="✏️",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    #MainMenu, footer, header { visibility: hidden; }
    .tool-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: border-color 0.15s;
    }
    .tool-card:hover { border-color: #63b3ed; }
    .result-box {
        background: #1a365d;
        border-left: 4px solid #63b3ed;
        border-radius: 0 8px 8px 0;
        padding: 16px;
        color: #bee3f8;
        font-size: 0.92rem;
        margin-top: 12px;
    }
    .error-box {
        background: #742a2a;
        border-left: 4px solid #f56565;
        border-radius: 0 8px 8px 0;
        padding: 16px;
        color: #feb2b2;
        font-size: 0.92rem;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style='color:#e2e8f0; margin:0;'>✏️ Document Editor</h2>
<p style='color:#718096; font-size:0.9rem; margin:4px 0 0 0;'>
    Tell the Editor Agent what to do in plain English —
    it picks the right tool automatically
</p>
""", unsafe_allow_html=True)
st.divider()

col_main, col_tools = st.columns([2, 1])

with col_main:
    st.markdown("**1. Upload the file to edit**")
    edit_file = st.file_uploader(
        "Choose a .docx, .pdf, or image file",
        type=["docx", "pdf", "png", "jpg", "jpeg"],
        key="edit_upload"
    )

    file_path_on_server = None
    if edit_file:
        # Upload to server's working directory via API
        try:
            files = {
                "file": (edit_file.name,
                         edit_file.getvalue(),
                         edit_file.type)
            }
            r = requests.post(
                f"{API_URL}/api/v1/edit/upload",
                files=files, timeout=20
            )
            if r.status_code == 200:
                file_path_on_server = r.json().get("path")
                st.success(
                    f"✅ Uploaded: `{file_path_on_server}`")
            else:
                # Fallback — let user type path manually
                file_path_on_server = st.text_input(
                    "Server file path",
                    value=f"data/test_docs/{edit_file.name}")
        except requests.ConnectionError:
            file_path_on_server = st.text_input(
                "Server file path "
                "(upload endpoint unreachable — enter manually)",
                value=f"data/test_docs/{edit_file.name}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**2. Describe the edit you want**")

    example_instructions = [
        "Replace the word DRAFT with FINAL",
        "Add a CONFIDENTIAL watermark",
        "Resize this image to 800px width",
        "Insert a table comparing Method, Score, Speed",
        "Convert this PDF to a Word document",
        "Add a heading called 'Appendix'",
    ]

    chips = st.columns(3)
    for i, ex in enumerate(example_instructions):
        with chips[i % 3]:
            if st.button(ex, key=f"ex_{i}",
                        use_container_width=True):
                st.session_state.edit_instruction = ex

    instruction = st.text_area(
        "Instruction",
        value=st.session_state.get(
            "edit_instruction", ""),
        placeholder=(
            "e.g. Replace 'DRAFT' with 'FINAL' in "
            "the document, or add a watermark saying "
            "CONFIDENTIAL..."),
        height=100,
        label_visibility="collapsed"
    )

    run_clicked = st.button(
        "✨ Run Editor Agent",
        type="primary",
        use_container_width=True,
        disabled=not (edit_file and instruction.strip())
    )

    if run_clicked:
        with st.spinner(
            "🤖 Editor Agent selecting tool and "
            "executing…"
        ):
            try:
                payload = {
                    "instruction": instruction,
                    "file_path": (
                        file_path_on_server or
                        f"data/test_docs/{edit_file.name}")
                }
                r = requests.post(
                    f"{API_URL}/api/v1/edit",
                    json=payload, timeout=45
                )
                if r.status_code == 200:
                    data = r.json()
                    st.markdown(
                        f"<div class='result-box'>"
                        f"✅ <b>Result:</b><br>"
                        f"{data.get('result', '')}"
                        f"</div>",
                        unsafe_allow_html=True)

                    # Offer download if a new file was made
                    out_path = (
                        file_path_on_server or "")
                    if os.path.exists(out_path):
                        with open(out_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download edited file",
                                data=f.read(),
                                file_name=os.path.basename(
                                    out_path),
                                use_container_width=True
                            )
                else:
                    st.markdown(
                        f"<div class='error-box'>"
                        f"❌ HTTP {r.status_code}: "
                        f"{r.text[:200]}</div>",
                        unsafe_allow_html=True)
            except requests.ConnectionError:
                st.markdown(
                    f"<div class='error-box'>"
                    f"❌ Cannot reach API at {API_URL}"
                    f"</div>",
                    unsafe_allow_html=True)
            except requests.Timeout:
                st.markdown(
                    "<div class='error-box'>"
                    "⏱️ Request timed out</div>",
                    unsafe_allow_html=True)

with col_tools:
    st.markdown("**Available Tools**")
    try:
        r = requests.get(
            f"{API_URL}/api/v1/edit/tools", timeout=5)
        if r.status_code == 200:
            tools = r.json().get("tools", [])
            for t in tools:
                icon = (
                    "📄" if "docx" in t["name"] else
                    "📑" if "pdf"  in t["name"] else
                    "🖼️"
                )
                st.markdown(f"""
                <div class='tool-card'>
                    <b style='color:#e2e8f0; font-size:0.85rem;'>
                        {icon} {t['name'].replace('tool_','').replace('_',' ').title()}
                    </b>
                    <p style='color:#718096; font-size:0.75rem;
                              margin:4px 0 0 0;'>
                        {t['description'][:80]}…
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Could not load tool list.")
    except requests.ConnectionError:
        st.caption(f"⚠️ API offline at {API_URL}")