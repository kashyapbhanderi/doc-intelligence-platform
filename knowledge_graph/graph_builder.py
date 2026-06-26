"""
GraphRAG — graph_builder.py
===========================
Builds a knowledge graph from your document chunks.

Nodes:
  - chunk   → each text chunk from your ingestion pipeline
  - entity  → named entities (person, org, location, etc.)

Edges:
  - chunk → entity  : "this chunk mentions this entity"
  - entity → entity : "these entities are related" (LLM-extracted)

WHERE TO PLACE THIS FILE: knowledge_graph/graph_builder.py

INSTALL DEPENDENCIES:
    pip install spacy networkx
    python -m spacy download en_core_web_sm
"""

import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

from openai import OpenAI


# Entity types worth extracting for document intelligence
_USEFUL_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "LOC",
    "PRODUCT", "EVENT", "LAW",
    "MONEY", "PERCENT", "DATE", "NORP"
}


class GraphBuilder:
    """
    Builds and persists a NetworkX knowledge graph from document chunks.

    Typical usage (run ONCE after Week 2 ingestion):
    ─────────────────────────────────────────────────
    from knowledge_graph.graph_builder import GraphBuilder

    gb = GraphBuilder()
    if not gb.load():                  # try loading saved graph
        with open("data/chunks.json") as f:
            chunks = json.load(f)
        gb.build_from_chunks(chunks)   # ~5-10 min for 500 docs
    """

    def __init__(self, graph_path: str = "data/knowledge_graph.pkl"):
        self.graph_path = Path(graph_path)
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)

        self.G: nx.Graph = nx.Graph()
        self.entity_to_chunks: dict = defaultdict(list)
        from config.llm_config import get_llm_config
        cfg = get_llm_config()
        self.client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        self.model  = cfg["model"]

        # Load spaCy model
        self.nlp = None
        if _SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("GraphBuilder: spaCy loaded (en_core_web_sm)")
            except OSError:
                print("GraphBuilder: spaCy model not found — run: python -m spacy download en_core_web_sm")

    # ────────────────────────────────────────────────
    # ENTITY EXTRACTION
    # ────────────────────────────────────────────────

    def _extract_entities_spacy(self, text: str) -> list[dict]:
        """Fast NER with spaCy. Returns list of {text, type}."""
        if not self.nlp:
            return []
        try:
            doc = self.nlp(text[:8000])   # spaCy hard cap
            seen = set()
            entities = []
            for ent in doc.ents:
                if ent.label_ not in _USEFUL_ENTITY_TYPES:
                    continue
                key = ent.text.strip().lower()
                if key in seen or len(key) < 2:
                    continue
                seen.add(key)
                entities.append({"text": ent.text.strip(), "type": ent.label_})
            return entities
        except Exception:
            return []

    def _extract_relations_llm(self, text: str, entities: list[dict]) -> list[dict]:
        """
        Ask GPT-4o-mini to find relationships between entities.
        Only called when >= 2 entities found in chunk (cost control).
        Rate-limited to 1 call per 0.5s.
        """
        if len(entities) < 2:
            return []

        names = [e["text"] for e in entities[:8]]   # cap to 8 entities

        prompt = (
            f"Text:\n{text[:1200]}\n\n"
            f"Entities: {', '.join(names)}\n\n"
            "Extract relationships between these entities from the text.\n"
            "Return ONLY a JSON array. Each element:\n"
            '{"from": "entity", "to": "entity", "relation": "short description"}\n'
            "Return [] if no clear relationships. JSON only:"
        )
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
            )
            raw = resp.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            relations = json.loads(raw)
            time.sleep(0.5)         # respect rate limit
            return relations if isinstance(relations, list) else []
        except Exception:
            return []

    # ────────────────────────────────────────────────
    # GRAPH CONSTRUCTION
    # ────────────────────────────────────────────────

    def add_chunk(self, chunk_id: str, text: str, source: str = "", page: int = 0) -> None:
        """
        Process one chunk: add to graph and link to extracted entities.
        """
        # Chunk node — store short preview to keep graph small
        self.G.add_node(
            chunk_id,
            node_type="chunk",
            text=text[:600],
            source=source,
            page=page,
        )

        entities = self._extract_entities_spacy(text)

        for entity in entities:
            eid = f"entity::{entity['text'].lower()}"

            # Entity node
            if eid not in self.G:
                self.G.add_node(
                    eid,
                    node_type="entity",
                    text=entity["text"],
                    entity_type=entity["type"],
                )

            # chunk ↔ entity edge
            if not self.G.has_edge(chunk_id, eid):
                self.G.add_edge(chunk_id, eid, edge_type="mentions", weight=1.0)

            self.entity_to_chunks[eid].append(chunk_id)

        # Entity ↔ entity edges (LLM, only when ≥ 2 entities)
        if len(entities) >= 2:
            relations = self._extract_relations_llm(text, entities)
            for rel in relations:
                fid = f"entity::{str(rel.get('from', '')).lower()}"
                tid = f"entity::{str(rel.get('to', '')).lower()}"
                if fid in self.G and tid in self.G and not self.G.has_edge(fid, tid):
                    self.G.add_edge(
                        fid, tid,
                        edge_type="relation",
                        relation=rel.get("relation", "related_to"),
                        weight=2.0,    # explicit relations score higher
                    )

    def build_from_chunks(self, chunks: list[dict], log_every: int = 50) -> None:
        """
        Build graph from your existing chunks.json list.
        Each chunk needs: id, text, and optionally source / page.

        Args:
            chunks    : list of chunk dicts from data/chunks.json
            log_every : print progress every N chunks
        """
        print(f"[GraphBuilder] Building from {len(chunks)} chunks…")
        for i, chunk in enumerate(chunks):
            if i % log_every == 0:
                print(f"  {i}/{len(chunks)} chunks processed")
            self.add_chunk(
                chunk_id=str(chunk.get("id", i)),
                text=chunk.get("text", ""),
                source=chunk.get("source", ""),
                page=int(chunk.get("page", 0)),
            )

        self.save()
        stats = self.get_stats()
        print(f"[GraphBuilder] Done — {stats['entity_nodes']} entities, "
              f"{stats['chunk_nodes']} chunks, {stats['total_edges']} edges")

    # ────────────────────────────────────────────────
    # PERSISTENCE
    # ────────────────────────────────────────────────

    def save(self) -> None:
        """Pickle the graph to disk."""
        with open(self.graph_path, "wb") as f:
            pickle.dump({"graph": self.G, "entity_to_chunks": dict(self.entity_to_chunks)}, f)
        print(f"[GraphBuilder] Graph saved → {self.graph_path}")

    def load(self) -> bool:
        """Load persisted graph. Returns True on success."""
        if not self.graph_path.exists():
            print(f"[GraphBuilder] No saved graph at {self.graph_path}")
            return False
        try:
            with open(self.graph_path, "rb") as f:
                data = pickle.load(f)
            self.G = data["graph"]
            self.entity_to_chunks = defaultdict(list, data["entity_to_chunks"])
            print(f"[GraphBuilder] Loaded — {self.G.number_of_nodes()} nodes, "
                  f"{self.G.number_of_edges()} edges")
            return True
        except Exception as e:
            print(f"[GraphBuilder] Load failed: {e}")
            return False

    def get_stats(self) -> dict:
        """Summary statistics for README / MLflow logging."""
        nodes = dict(self.G.nodes(data=True))
        entity_nodes = [n for n, d in nodes.items() if d.get("node_type") == "entity"]
        chunk_nodes  = [n for n, d in nodes.items() if d.get("node_type") == "chunk"]
        degrees = [d for _, d in self.G.degree()]
        return {
            "total_nodes":             self.G.number_of_nodes(),
            "total_edges":             self.G.number_of_edges(),
            "entity_nodes":            len(entity_nodes),
            "chunk_nodes":             len(chunk_nodes),
            "avg_degree":              round(sum(degrees) / max(len(degrees), 1), 2),
            "max_degree":              max(degrees) if degrees else 0,
            "graph_density":           round(nx.density(self.G), 6),
        }
