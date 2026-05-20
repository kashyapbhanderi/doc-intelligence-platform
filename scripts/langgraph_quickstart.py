"""
LangGraph Quickstart — understand the basics
before building the real system.

LangGraph = a graph where:
- NODES = functions (agents or tools)
- EDGES = connections between nodes
- STATE = shared data flowing through the graph

Think of it like:
Node A → Node B → Node C
         ↓
        END
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator


# ── Step 1: Define State ────────────────────────
# State is the shared memory between all nodes.
# Every node can read and write to state.

class SimpleState(TypedDict):
    """
    State shared between all agents.
    Every field here is accessible by every node.
    """
    input: str           # original user input
    messages: list       # conversation history
    result: str          # final result


# ── Step 2: Define Nodes (functions) ───────────
# Each node takes state, does something, returns
# updated state.

def node_a(state: SimpleState) -> dict:
    """First node — processes input."""
    print(f"Node A received: {state['input']}")
    return {
        "messages": [f"Node A processed: {state['input']}"]
    }


def node_b(state: SimpleState) -> dict:
    """Second node — transforms the message."""
    last_msg = state["messages"][-1]
    print(f"Node B received: {last_msg}")
    return {
        "messages": state["messages"] + [
            f"Node B transformed: {last_msg.upper()}"
        ]
    }


def node_c(state: SimpleState) -> dict:
    """Third node — creates final result."""
    all_messages = state["messages"]
    print(f"Node C creating result from "
          f"{len(all_messages)} messages")
    return {
        "result": f"Final: {all_messages[-1]}"
    }


# ── Step 3: Build the Graph ─────────────────────
def build_simple_graph():
    # Create graph with our state type
    graph = StateGraph(SimpleState)

    # Add nodes
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_node("node_c", node_c)

    # Add edges (connections)
    graph.set_entry_point("node_a")  # start here
    graph.add_edge("node_a", "node_b")  # a → b
    graph.add_edge("node_b", "node_c")  # b → c
    graph.add_edge("node_c", END)       # c → done

    # Compile
    return graph.compile()


# ── Step 4: Run the Graph ───────────────────────
if __name__ == "__main__":
    print("Building simple LangGraph...")
    app = build_simple_graph()

    print("\nRunning graph with test input:")
    print("-" * 40)

    result = app.invoke({
        "input": "hello world",
        "messages": [],
        "result": ""
    })

    print("\nFinal state:")
    print(f"  Input:    {result['input']}")
    print(f"  Messages: {result['messages']}")
    print(f"  Result:   {result['result']}")
    print("\nLangGraph working correctly!")