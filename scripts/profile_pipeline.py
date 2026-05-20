"""
scripts/profile_pipeline.py
Measures time spent in each agent stage.
Run before and after optimisations to compare.
"""
import os
import sys
import time
sys.path.insert(0, os.path.abspath('.'))

from agents.state import create_initial_state
from agents.planner import planner_node
from agents.executor import executor_node
from agents.critic import critic_node

QUESTION = "What is retrieval augmented generation?"

print("Pipeline Latency Profile")
print("=" * 50)
print(f"Question: {QUESTION}\n")

state = create_initial_state(QUESTION)

# Time each agent individually
stages = [
    ("Planner",  planner_node),
    ("Executor", executor_node),
    ("Critic",   critic_node),
]

timings = {}
for name, fn in stages:
    t0     = time.time()
    update = fn(state)
    elapsed = time.time() - t0
    state.update(update)
    timings[name] = elapsed
    print(f"  {name:<10} {elapsed:.2f}s")

total = sum(timings.values())
print(f"\n  {'TOTAL':<10} {total:.2f}s")
print("\nBreakdown:")
for name, t in timings.items():
    pct = t / total * 100
    bar = "█" * int(pct / 5)
    print(f"  {name:<10} {bar:<20} {pct:.0f}%")

print("\nTarget: total < 5.0s")
print(f"Status: {'✅ PASS' if total < 5.0 else '⚠️  OVER TARGET'}")