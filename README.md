# BabyAI Solver Challenge

A challenge benchmark for AI agents: build a solver for [BabyAI](https://github.com/Farama-Foundation/Minigrid) gridworld levels, from simple navigation up to a 36-room labyrinth.

**[Web Visualizer →](https://sammydmartin.github.io/baby/)** (see previous solver results animated step-by-step)

## The Challenge

22 levels across 5 difficulty tiers. Write a solver that produces action sequences to complete each level. Tested on 3 seeds per level (66 instances).

| Tier | Levels | What's Involved |
|------|--------|-----------------|
| **Easy** (D1-3) | GoTo, Open Door | Navigate to objects, open doors. Single room. |
| **Medium** (D4-6) | Pickup, Unlock, Maze | Pick up objects, use keys, multi-room navigation. |
| **Hard** (D7-9) | KeyCorridor, BlockedUnlock, Boss | Keys behind doors, compound missions ("X and Y"). |
| **Nightmare** (D10-13) | KeyChain, MegaMaze, Backtrack, Compound | Chained key deps, 16-room mazes, 150+ step solutions. |
| **Impossible** (D15) | Labyrinth | 6x6 rooms (36 rooms), 25x25 grid, 60 doors, 200-500+ steps. |

## Quick Start

```bash
pip install minigrid gymnasium

# List all levels
python challenge.py list

# View a level
python challenge.py show BabyAI-GoToRedBallNoDists-v0 42

# Verify an action sequence
python challenge.py verify BabyAI-GoToRedBallNoDists-v0 42 "right right forward"

# Run your solver against all levels
python challenge.py suite my_solver
```

**Full instructions are in [CLAUDE.md](CLAUDE.md)** — read that to understand the grid, actions, and how to write a solver.

## Project Structure

```
CLAUDE.md              # Challenge instructions (read this first)
challenge.py           # View levels, verify solutions, run test suite
engine/                # Grid rendering + level definitions (fair game to read)
  grid.py              # Grid state extraction, rendering, constants
  levels.py            # Standard + Nightmare level definitions
  impossible.py        # Impossible Labyrinth definition
docs/                  # Web visualizer (GitHub Pages)
  index.html           # Interactive grid replay
  data.json            # Pre-generated results
solvers/               # ⚠️  REFERENCE SOLUTIONS — don't read if attempting the challenge
  DO_NOT_READ.md       # Explains what's in here
  solver.py            # Full BFS solver (100% solve rate)
  pathfinding.py       # BFS pathfinding algorithms
  ...                  # Other solver versions and notes
```

### Legacy files (from original exploration)
```
babyai_harness.py      # Original text rendering harness
show_tasks.py          # Quick task viewer
nightmare_levels.py    # Original nightmare level definitions
```

## Writing Your Solver

Create a module with a `solve(env_name, seed) -> list[str]` function:

```python
# my_solver.py
import gymnasium as gym
from engine.grid import get_grid_info, find_objects, ACTION_MAP

def solve(env_name, seed):
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    # ... your logic ...
    env.close()
    return ["right", "forward", "toggle", ...]
```

Run: `python challenge.py suite my_solver`

## Credits

- BabyAI platform: Chevalier-Boisvert, Bahdanau, Bengio et al. ([paper](https://arxiv.org/abs/1810.08272))
- Challenge design: Built by Claude (Anthropic) in conversation with Sammy Martin, March 2026
