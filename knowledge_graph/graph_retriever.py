"""
GraphRAG — graph_retriever.py
==============================
Retrieves document chunks via knowledge graph traversal.

Multi-hop reasoning example:
  Query: "What did the CEO say about the acquisition that was mentioned in Q3?"
  Step 1 → find entity "CEO" and "Q3" in graph
  Step 2 → traverse 2 hops → finds "acquisition" entity connected to both
  Step 3 → returns chunks that mention all three

WHERE TO PLACE THIS FILE: knowledge_graph/graph_retriever.py
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge_graph.graph_builder import GraphBuilder


class GraphRetriever:
    """
    Retrieves relevant chunks from the knowledge graph for a given query.

    Works in three stages:
    1. Match query words → entity nodes in graph
    2. Traverse graph N hops from those entities
    3. Score and rank the connected chunk nodes
    """

    def __init__(self, graph_builder: "GraphBuilder"):
        self.gb = graph_builder
        self._entity_cache: list[tuple[str, str]] = []   # (node_id, display_text)
        self._refresh_cache()

    def _refresh_cache(self) -> None:
        """Rebuild the entity lookup cache from the current graph."""
        self._entity_cache = [
            (nid, data.get("text", ""))
            for nid, data in self.gb.G.nodes(data=True)
            if data.get("node_type") == "entity"
        ]

    # ────────────────────────────────────────────────
    # STEP 1 — ENTITY MATCHING
    # ────────────────────────────────────────────────

    def _find_query_entities(self, query: str, top_k: int = 6) -> list[str]:
        """
        Find entity nodes that are relevant to the query.
        Uses fast substring + word-overlap scoring (no extra API call).
        """
        q_lower = query.lower()
        q_words = set(re.findall(r"\b\w{3,}\b", q_lower))
        scores: dict[str, float] = {}

        for eid, etext in self._entity_cache:
            e_lower = etext.lower()
            e_words = set(re.findall(r"\b\w{3,}\b", e_lower))

            # Exact substring match → highest score
            if e_lower in q_lower or q_lower in e_lower:
                scores[eid] = 1.0
                continue

            # Jaccard word overlap
            union = q_words | e_words
            if not union:
                continue
            overlap = len(q_words & e_words) / len(union)
            if overlap >= 0.25:
                scores[eid] = overlap

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in ranked[:top_k]]

    # ────────────────────────────────────────────────
    # STEP 2 — GRAPH TRAVERSAL (multi-hop)
    # ────────────────────────────────────────────────

    def _traverse(self, seed_entities: list[str], hops: int = 2, max_chunks: int = 40) -> set[str]:
        """
        BFS from seed entities up to `hops` depth.
        Collects chunk nodes along the way.
        """
        visited_entities = set(seed_entities)
        visited_chunks: set[str] = set()
        frontier = set(seed_entities)

        for _ in range(hops):
            if len(visited_chunks) >= max_chunks:
                break
            next_frontier: set[str] = set()

            for eid in frontier:
                if eid not in self.gb.G:
                    continue
                for neighbor in self.gb.G.neighbors(eid):
                    ndata = self.gb.G.nodes[neighbor]
                    if ndata.get("node_type") == "chunk":
                        visited_chunks.add(neighbor)
                    elif ndata.get("node_type") == "entity" and neighbor not in visited_entities:
                        visited_entities.add(neighbor)
                        next_frontier.add(neighbor)

            frontier = next_frontier
            if not frontier:
                break

        return visited_chunks

    # ────────────────────────────────────────────────
    # STEP 3 — SCORING & RANKING
    # ────────────────────────────────────────────────

    def _score_chunks(self, chunk_ids: set[str], query: str) -> list[dict]:
        """
        Rank retrieved chunks by word overlap with query.
        Returns standardised dicts compatible with your existing search results.
        """
        q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        results = []

        for cid in chunk_ids:
            if cid not in self.gb.G:
                continue
            cdata = self.gb.G.nodes[cid]
            c_words = set(re.findall(r"\b\w{3,}\b", cdata.get("text", "").lower()))
            union = q_words | c_words
            score = len(q_words & c_words) / len(union) if union else 0.0

            results.append({
                "chunk_id":         cid,
                "text":             cdata.get("text", ""),
                "source":           cdata.get("source", ""),
                "page":             cdata.get("page", 0),
                "graph_score":      round(score, 4),
                "retrieval_method": "graphrag",
            })

        results.sort(key=lambda x: x["graph_score"], reverse=True)
        return results

    # ────────────────────────────────────────────────
    # PUBLIC API
    # ────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5, hops: int = 2) -> list[dict]:
        """
        Full GraphRAG retrieval pipeline.

        Args:
            query  : natural language question
            top_k  : number of chunks to return
            hops   : traversal depth (1 = direct neighbours, 2 = friends-of-friends)

        Returns:
            list of chunk dicts with text, source, page, graph_score
        """
        if self.gb.G.number_of_nodes() == 0:
            return []

        seeds = self._find_query_entities(query)
        if not seeds:
            return []

        chunk_ids = self._traverse(seeds, hops=hops)
        if not chunk_ids:
            return []

        return self._score_chunks(chunk_ids, query)[:top_k]

    def explain(self, query: str) -> dict:
        """
        Debug helper: shows which entities were matched and what was traversed.
        Useful during development — log to LangSmith or print to console.

        Example output:
            {
              "query": "...",
              "matched_entities": ["entity::infosys", "entity::ceo"],
              "traversal_depth": 2,
              "chunks_found": 12,
              "top_result_preview": "..."
            }
        """
        seeds = self._find_query_entities(query)
        seeds_text = [
            self.gb.G.nodes[s].get("text", s)
            for s in seeds
            if s in self.gb.G
        ]
        chunk_ids = self._traverse(seeds) if seeds else set()
        results = self._score_chunks(chunk_ids, query) if chunk_ids else []

        return {
            "query":               query,
            "matched_entities":    seeds_text,
            "traversal_depth":     2,
            "chunks_found":        len(chunk_ids),
            "top_result_preview":  results[0]["text"][:200] if results else "—",
        }

    def get_entity_neighbors(self, entity_name: str) -> dict:
        """
        Inspect all entities connected to a given entity.
        Useful for graph debugging and visualisation.
        """
        eid = f"entity::{entity_name.lower()}"
        if eid not in self.gb.G:
            return {"entity": entity_name, "found": False}

        connections = []
        for nbr in self.gb.G.neighbors(eid):
            edata = self.gb.G.edges[eid, nbr]
            ndata = self.gb.G.nodes[nbr]
            if ndata.get("node_type") == "entity":
                connections.append({
                    "entity":    ndata.get("text", nbr),
                    "type":      ndata.get("entity_type", ""),
                    "relation":  edata.get("relation", "co-occurs with"),
                })

        return {
            "entity":      entity_name,
            "found":       True,
            "entity_type": self.gb.G.nodes[eid].get("entity_type", ""),
            "degree":      self.gb.G.degree(eid),
            "connections": connections,
        }
