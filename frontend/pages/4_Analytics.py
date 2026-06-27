"""
frontend/pages/4_Analytics.py
Evaluation analytics — RAGAS, NDCG, fine-tuning results.
Reads JSON eval files directly (no API roundtrip needed)
when running on the same machine, falls back to API.
"""
import streamlit as st
import requests
import json
import os

API_URL  = os.getenv("API_URL", "http://localhost:8000")
EVAL_DIR = os.getenv("EVAL_DIR", "eval")

st.set_page_config(
    page_title="Analytics — Doc Intelligence",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    #MainMenu, footer, header { visibility: hidden; }
    .metric-box {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .metric-box .val {
        font-size: 1.8rem; font-weight: 700;
    }
    .metric-box .lbl {
        font-size: 0.78rem; color: #a0aec0;
        margin-top: 4px;
    }
    .grade-excellent { color: #68d391; }
    .grade-good      { color: #63b3ed; }
    .grade-fair       { color: #f6ad55; }
    .grade-poor       { color: #fc8181; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style='color:#e2e8f0; margin:0;'>📊 Evaluation Analytics</h2>
<p style='color:#718096; font-size:0.9rem; margin:4px 0 0 0;'>
    RAGAS scores, NDCG@10 retrieval quality,
    and fine-tuning improvement metrics
</p>
""", unsafe_allow_html=True)
st.divider()


def load_json(path: str) -> dict | None:
    """Load a JSON eval file if it exists."""
    full = os.path.join(EVAL_DIR, path)
    if os.path.exists(full):
        with open(full, encoding="utf-8") as f:
            return json.load(f)
    return None


def grade(score: float) -> tuple[str, str]:
    if score >= 0.8:
        return "Excellent", "grade-excellent"
    if score >= 0.6:
        return "Good", "grade-good"
    if score >= 0.4:
        return "Fair", "grade-fair"
    return "Needs work", "grade-poor"


tab1, tab2, tab3, tab4 = st.tabs(
    ["🏆 RAGAS Scores",
     "📈 Retrieval Quality (NDCG)",
     "🔬 Fine-tuning Improvement",
     "🕸️ GraphRAG + Memory"])

# ── Tab 1: RAGAS ───────────────────────────────────────────
with tab1:
    ragas = load_json("ragas_results.json")
    if ragas:
        s = ragas["summary"]
        st.caption(
            f"Evaluated on {s.get('num_samples', 0)} "
            f"Q&A samples")

        cols = st.columns(4)
        metrics = [
            ("Faithfulness", s.get("faithfulness", 0)),
            ("Answer Relevancy",
             s.get("answer_relevancy", 0)),
            ("Context Recall",
             s.get("context_recall", 0)),
            ("Context Precision",
             s.get("context_precision", 0)),
        ]
        for col, (name, val) in zip(cols, metrics):
            label, cls = grade(val)
            with col:
                st.markdown(f"""
                <div class='metric-box'>
                    <div class='val {cls}'>{val:.3f}</div>
                    <div class='lbl'>{name}</div>
                    <div class='lbl {cls}'>{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        try:
            import pandas as pd
            df = pd.DataFrame({
                "Metric": [m[0] for m in metrics],
                "Score":  [m[1] for m in metrics],
            })
            st.bar_chart(
                df.set_index("Metric"),
                height=280)
        except ImportError:
            st.caption("Install pandas for chart view")
    else:
        st.info(
            "No RAGAS results found yet. "
            "Run `python eval/ragas_eval.py` first.")

# ── Tab 2: NDCG ────────────────────────────────────────────
with tab2:
    ndcg_base = load_json("ndcg_results.json")
    ndcg_ft   = load_json("ndcg_finetuned.json")

    if ndcg_base:
        bs = ndcg_base["summary"]
        fs = ndcg_ft["summary"] if ndcg_ft else None

        st.markdown("**NDCG@10 by Search Method**")

        methods = ["bm25_ndcg", "vector_ndcg", "hybrid_ndcg"]
        labels  = ["BM25", "Vector", "Hybrid RRF"]

        cols = st.columns(3)
        for col, m, lbl in zip(cols, methods, labels):
            base_val = bs.get(m, 0)
            ft_val   = fs.get(m, 0) if fs else None
            with col:
                if ft_val:
                    delta = ft_val - base_val
                    st.metric(lbl, f"{ft_val:.4f}",
                            delta=f"{delta:+.4f}")
                    st.caption(f"Baseline: {base_val:.4f}")
                else:
                    st.metric(lbl, f"{base_val:.4f}")

        st.markdown("<br>", unsafe_allow_html=True)
        try:
            import pandas as pd
            chart_data = {"Method": labels}
            chart_data["Baseline"] = [
                bs.get(m, 0) for m in methods]
            if fs:
                chart_data["Fine-tuned"] = [
                    fs.get(m, 0) for m in methods]
            df = pd.DataFrame(chart_data).set_index(
                "Method")
            st.bar_chart(df, height=300)
        except ImportError:
            pass
    else:
        st.info(
            "No NDCG results found yet. "
            "Run `python eval/ndcg_eval.py` first.")

# ── Tab 3: Fine-tuning ─────────────────────────────────────
with tab3:
    comparison = load_json("model_comparison.json")
    analysis   = load_json("ndcg_analysis.json")

    if comparison:
        st.markdown("**Model Comparison**")
        scores = comparison.get("all_scores", {})

        cols = st.columns(len(scores) if scores else 1)
        for col, (name, val) in zip(cols, scores.items()):
            is_winner = name == comparison.get("winner")
            with col:
                st.metric(
                    f"{'🏆 ' if is_winner else ''}{name}",
                    f"{val:.4f}")

        st.success(
            f"Winner: **{comparison.get('winner')}** "
            f"with gap score "
            f"{comparison.get('winner_score', 0):.4f}")

    if analysis:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Per-Query Improvement**")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Queries Improved",
                     f"{analysis.get('improved_count', 0)}/"
                     f"{analysis.get('total_queries', 0)}")
        with c2:
            st.metric("Improvement Rate",
                     f"{analysis.get('improvement_rate', 0):.0f}%")
        with c3:
            st.metric("Got Worse",
                     analysis.get('worse_count', 0))

        details = analysis.get("details", [])
        if details:
            st.markdown(
                "**Top improved queries**")
            for item in sorted(
                details,
                key=lambda x: x.get(
                    "hybrid_improvement", 0),
                reverse=True
            )[:5]:
                imp = item.get("hybrid_improvement", 0)
                icon = "🟢" if imp > 0 else "🔴"
                st.markdown(
                    f"{icon} `{imp:+.4f}` — "
                    f"{item.get('question', '')[:70]}")

    if not comparison and not analysis:
        st.info(
            "No fine-tuning comparison found yet. "
            "Run `python scripts/pick_best_model.py` "
            "or `python eval/ndcg_analysis.py` first.")

# ── Tab 4: GraphRAG + Memory ──────────────────────────────
with tab4:
    col_g, col_m = st.columns(2)

    # ── GraphRAG stats (live from API) ────────────────────
    with col_g:
        st.markdown("**🕸️ Knowledge Graph**")
        try:
            r = requests.get(
                f"{API_URL}/graphrag/stats", timeout=5)
            if r.status_code == 200:
                stats = r.json()
                gc1, gc2 = st.columns(2)
                with gc1:
                    st.metric("Total Nodes",
                             f"{stats.get('total_nodes', 0):,}")
                    st.metric("Entity Nodes",
                             f"{stats.get('entity_nodes', 0):,}")
                    st.metric("Avg Degree",
                             stats.get("avg_degree", 0))
                with gc2:
                    st.metric("Total Edges",
                             f"{stats.get('total_edges', 0):,}")
                    st.metric("Chunk Nodes",
                             f"{stats.get('chunk_nodes', 0):,}")
                    st.metric("Graph Density",
                             f"{stats.get('graph_density', 0):.5f}")
            elif r.status_code == 503:
                st.warning(
                    "⚠️ GraphRAG not initialised yet. "
                    "Run `python scripts/build_graph.py` "
                    "then restart the API.")
            else:
                st.info(f"GraphRAG returned HTTP {r.status_code}")
        except requests.ConnectionError:
            st.warning(f"⚠️ Cannot reach API at {API_URL}")
        except Exception as e:
            st.info(f"GraphRAG stats unavailable: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🔍 Inspect an entity**")
        entity_name = st.text_input(
            "Entity name (e.g. Infosys)",
            placeholder="Type an entity from your documents…",
            key="entity_lookup")
        if entity_name and st.button(
            "Look up entity", use_container_width=True):
            try:
                r = requests.get(
                    f"{API_URL}/graphrag/entity/{entity_name}",
                    timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    st.json(data)
                elif r.status_code == 404:
                    st.warning(
                        f"Entity '{entity_name}' not found "
                        f"in the knowledge graph.")
                else:
                    st.info(f"HTTP {r.status_code}")
            except Exception as e:
                st.error(str(e))

    # ── Memory stats (live from API) ──────────────────────
    with col_m:
        st.markdown("**🧠 Long-Term Memory**")
        user_id = st.text_input(
            "User ID to inspect",
            value="default",
            key="memory_user_lookup")

        if st.button("Load memories",
                     use_container_width=True,
                     type="primary"):
            try:
                r = requests.get(
                    f"{API_URL}/memory/user/{user_id}",
                    timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    stats = data.get("stats", {})

                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.metric(
                            "Semantic Facts",
                            len(data.get(
                                "semantic_facts", [])))
                    with mc2:
                        st.metric(
                            "Recent Episodes",
                            len(data.get(
                                "recent_episodes", [])))

                    facts = data.get("semantic_facts", [])
                    if facts:
                        st.markdown("**Stored facts**")
                        for f in facts[:10]:
                            st.markdown(
                                f"• {f.get('fact', f)}")

                    episodes = data.get(
                        "recent_episodes", [])
                    if episodes:
                        st.markdown(
                            "**Recent episode summaries**")
                        for ep in episodes[:5]:
                            st.markdown(
                                f"📝 {ep.get('summary', ep)}")

                    if not facts and not episodes:
                        st.info(
                            "No memories stored yet for "
                            "this user. They appear after "
                            "calling POST /memory/session/end.")

                elif r.status_code == 503:
                    st.warning(
                        "⚠️ Memory agent not initialised. "
                        "Check api/main.py wiring.")
                else:
                    st.info(f"HTTP {r.status_code}")
            except requests.ConnectionError:
                st.warning(f"⚠️ Cannot reach API at {API_URL}")
            except Exception as e:
                st.error(str(e))

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            "🗑️ Clear this user's memories",
            use_container_width=True
        ):
            try:
                r = requests.delete(
                    f"{API_URL}/memory/user/{user_id}",
                    timeout=5)
                if r.status_code == 200:
                    st.success("Memories cleared.")
                else:
                    st.info(f"HTTP {r.status_code}")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.caption(
        "🕸️ GraphRAG fuses entity-graph traversal with "
        "vector search via Reciprocal Rank Fusion for "
        "multi-hop questions. "
        "🧠 Memory persists facts and episode summaries "
        "across sessions so the agent recalls prior "
        "context for a returning user."
    )

# ── MLflow link ────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='color:#718096; font-size:0.85rem;'>
    📈 For full experiment history (training runs,
    hyperparameters, all metrics over time), open
    <a href='http://localhost:5000' target='_blank'
       style='color:#63b3ed;'>MLflow UI</a>
</p>
""", unsafe_allow_html=True)