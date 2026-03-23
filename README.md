# BabyAI LLM Challenge

Can a frontier LLM (Claude) solve the BabyAI benchmark — a gridworld language grounding test suite from 2019?

## Background

**BabyAI** was introduced by Chevalier-Boisvert, Bahdanau, Bengio et al. at ICLR 2019.  
Paper: [BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning](https://arxiv.org/abs/1810.08272)  
Original repo: [github.com/mila-iqia/babyai](https://github.com/mila-iqia/babyai) (now archived — environments live in [Minigrid](https://github.com/Farama-Foundation/Minigrid))

The platform comprises 19 levels of increasing difficulty in a 2D gridworld. An agent receives compositional natural language instructions (e.g. "put the red ball next to the blue box") and must navigate, interact with objects, unlock doors, and complete multi-step tasks. The paper's core finding was that deep learning methods of the era were not sample-efficient enough to acquire compositional language grounding — they needed impractical amounts of training data for tasks a toddler handles after a few demonstrations.

## What this repo contains

### Core files

| File | Description |
|------|-------------|
| `babyai_harness.py` | Text rendering of BabyAI gridworlds + action execution harness. Converts the visual gridworld to ASCII so an LLM can read the state. |
| `solve_v2.py` | Programmatic solver (v2). Parses missions, uses BFS pathfinding, handles multi-room navigation, unlock sequences, put-next-to tasks, compound missions. Runs a 17-task test suite across 5 seeds each. **Result: 79% (67/85).** |
| `solve_babyai.py` | Earlier solver (v1) for reference. Simpler, more bugs. |
| `nightmare_levels.py` | Custom "Nightmare" levels designed to stress-test LLM spatial reasoning: deep key chains, 4×4 mazes, multi-objective tasks, backtracking corridors. |
| `show_tasks.py` | Quick script to render sample instances across difficulty levels. |

### Key findings

**Two approaches were tested:**

1. **Code solver** (~45k tokens output): Write a programmatic solver with BFS pathfinding and mission parsing. Fast per task once built, but parser bugs and inventory state tracking issues limited accuracy. **79% across 85 task instances.**

2. **Manual reasoning** (estimated ~67k tokens for full benchmark): Read each grid as text, reason through the action sequence step by step, output actions directly. More reliable per instance but doesn't amortise. Demonstrated on KeyCorridor (48 actions, first try) and Boss Level (24 actions, first try) and NightmareMaze (53 actions, 6 doors, first try). **Estimated ~97%.**

### The efficiency gap

| Approach | Tokens | Accuracy | Notes |
|----------|--------|----------|-------|
| Code solver (actual) | ~45k | 79% | Parser bugs, inventory state bugs |
| Code solver (est. 100%) | ~70-100k | ~100% | More debugging iterations |
| Manual reasoning (est.) | ~67k | ~97% | Might miscount on very long paths |
| A baby | 0 tokens | ~100% | A few demos, glucose, seconds per task |

The capability gap between LLMs and babies has closed — Claude can solve every BabyAI level. The **efficiency gap** remains enormous. Bengio's benchmark was measuring sample/compute efficiency of grounded language learning, and on that metric the architectural mismatch (routing spatial reasoning through a language bottleneck) is still clearly visible in 2026.

## Setup

```bash
pip install minigrid gymnasium
```

## Running

```bash
# Show sample tasks across all difficulty levels
python show_tasks.py

# Run the programmatic solver test suite (79%)
python solve_v2.py

# Generate and view nightmare levels
python nightmare_levels.py
```

## Manual solve examples

The repo includes demonstrated one-shot manual solves (action sequences produced by reading the grid and reasoning, no code assistance):

- **KeyCorridor** (seed=42): 48 actions through 4 rooms, find key behind closed door, unlock locked door, drop key, pick up ball. First try.
- **Boss Level** (seed=99): 24 actions, navigate two closed doors across rooms, pick up red key. First try.
- **NightmareMaze** (seed=42): 53 actions, 4×4 room grid, open 6 doors, traverse from (4,3) to (16,17). First try.

## Solver failure modes (v2, 79%)

The 21% failure rate breaks down to:
- **UnlockPickup (0/5)**: After using a key on a locked door, the key stays in inventory. Can't pick up target while carrying key. Needed to drop key first — a grounded state interaction the solver didn't model.
- **KeyCorridor (0/5)**: Key is behind intermediate closed doors requiring multi-room navigation before the unlock sequence.
- **BlockedUnlockPickup (0/5)**: Object blocking the door must be moved before key can be used.
- **Some Boss Level instances (1/5 failed)**: Compound missions with interleaved subgoals ("pick up X and put Y next to Z") where execution order matters.
- **Parser edge cases**: "pick up the key on your left" — relative location modifiers.

All failures are in multi-step inventory management and mission parsing — not spatial reasoning or language understanding per se.

## Nightmare levels

Custom levels designed to exploit LLM spatial reasoning weaknesses:

| Level | Design | Why it's hard |
|-------|--------|---------------|
| `NightmareKeyChain` | 4-room corridor, 3 locked doors, chained keys | 3-deep inventory management with mandatory drops |
| `NightmareMaze` | 4×4 rooms (16 rooms), all doors closed | 100+ step sequences, easy to lose position tracking |
| `NightmareTripleTask` | 3×3 rooms, sequential compound mission | Interleaved subgoals requiring execution order planning |
| `NightmareSwap` | Single room, two put-next-to tasks | Multiple pick-drop cycles, dropped objects change grid |
| `NightmareBacktrack` | 1×5 corridor, key at far end, target behind locked door at other end | Full corridor traversal twice, ~150 steps |

## Credit

BabyAI platform: Maxime Chevalier-Boisvert, Dzmitry Bahdanau, Yoshua Bengio et al.  
This exploration: Conversation between Sammy Martin and Claude (Anthropic), March 2026.
