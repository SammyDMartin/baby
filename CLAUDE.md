# BabyAI Solver Challenge

## Your Task

Build a solver that can beat every level of the BabyAI gridworld challenge, including 4 custom "Nightmare" levels and a final "Impossible Labyrinth."

There are **22 levels** total across 5 difficulty tiers. Your solver will be tested on 3 random seeds per level (66 instances total).

## Rules

1. **Do NOT read any files in the `solvers/` directory.** That folder contains reference solutions. Reading them defeats the purpose of the challenge.
2. You may read everything else: `engine/`, `challenge.py`, `docs/`, old files like `babyai_harness.py`, `show_tasks.py`, `nightmare_levels.py`.
3. You should write your own solver from scratch. You can use any approach: BFS, heuristics, manual reasoning, or anything else.
4. Use `challenge.py` to view levels, test action sequences, and run the full suite.

## Setup

```bash
pip install minigrid gymnasium
```

## How It Works

Each level is a 2D gridworld. Your agent receives a natural language mission (e.g., "go to the red ball", "pick up the blue key", "put the green box next to the yellow ball") and must produce a sequence of actions to complete it.

**Actions:** `left`, `right`, `forward`, `pickup`, `drop`, `toggle`
- `left` / `right` — turn in place (rotate 90 degrees)
- `forward` — move one cell in the direction you're facing
- `toggle` — open/close/unlock the door in front of you
- `pickup` — pick up the object in front of you (can only carry one thing)
- `drop` — drop carried object in front of you

**Grid conventions:**
- Directions: 0=right(+x), 1=down(+y), 2=left(-x), 3=up(-y)
- Turn left = (dir-1)%4, turn right = (dir+1)%4
- Walls block movement. Closed doors block movement (toggle to open, then walk through).
- Locked doors require the matching colored key (carry key, face door, toggle).

## Viewing Levels

```bash
# List all levels
python challenge.py list

# View a specific level
python challenge.py show BabyAI-GoToRedBallNoDists-v0 42

# Verify an action sequence
python challenge.py verify BabyAI-GoToRedBallNoDists-v0 42 "right right forward forward forward"
```

## Writing Your Solver

Create a Python module with a `solve(env_name, seed)` function that returns a list of action strings:

You can use either of two interfaces:

**Option A: `solve_with_env(env)` (recommended)**
The suite creates and resets the env for you. You step it directly.

```python
# my_solver.py
from engine.grid import get_grid_info, find_objects, ACTION_MAP, DIR_DELTAS

def solve_with_env(env):
    """Solve a pre-reset env. Return (success: bool, actions: list[str])."""
    obs = env.unwrapped.gen_obs()
    mission = obs['mission']
    info = get_grid_info(env)

    actions = []
    # ... your solving logic, calling env.step(ACTION_MAP[action]) as you go ...

    return success, actions
```

**Option B: `solve(env_name, seed)`**
You create your own env. Actions are verified by replaying on a fresh env.

```python
# my_solver.py
import gymnasium as gym
from engine.grid import get_grid_info, find_objects, ACTION_MAP

def solve(env_name, seed):
    """Return a list of action strings to complete the level."""
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    # ... your logic ...
    env.close()
    return actions
```

Then run the test suite:

```bash
python challenge.py suite my_solver
```

## Difficulty Tiers

| Tier | Levels | What's Involved |
|------|--------|-----------------|
| **Easy** (D1-3) | GoTo variants, Open Door | Navigate to an object or open a door. Single room. |
| **Medium** (D4-6) | Pickup, Unlock, Maze (open) | Pick up objects, use keys, multi-room with open doors. |
| **Hard** (D7-9) | KeyCorridor, BlockedUnlock, Boss | Keys behind doors, blocked paths, compound missions ("X and Y"). |
| **Nightmare** (D10-13) | KeyChain, MegaMaze, Backtrack, Compound | Chained key dependencies, 16-room mazes, 150+ step solutions, put-then-pickup missions. |
| **Impossible** (D15) | Labyrinth | 36-room maze (6x6 rooms, 25x25 grid), 60 closed doors, 200-500 step solutions. |

## Key Concepts

- **Rooms** are separated by walls with doors between them. The grid is divided into rooms.
- **Closed doors** can be toggled open by facing them and using `toggle`.
- **Locked doors** require the matching colored key. Pick up the key, face the door, `toggle`.
- **Inventory** — you can only carry one object at a time. `drop` to make room.
- **Compound missions** — "pick up X and put Y next to Z" requires completing multiple sub-tasks in order.
- **"Go to"** completes when you are adjacent to and facing the target object.
- **"Put X next to Y"** — pick up X, navigate near Y, drop X adjacent to Y.

## Engine API

The `engine/` package provides:

```python
from engine.grid import (
    get_grid_info,    # Returns dict: grid, pos, dir, carrying, w, h
    find_objects,     # Find objects by type/color
    render_grid,      # ASCII rendering
    grid_to_dict,     # JSON-serializable grid state
    ACTION_MAP,       # {'left': 0, 'right': 1, 'forward': 2, ...}
    DIR_DELTAS,       # {0: (1,0), 1: (0,1), 2: (-1,0), 3: (0,-1)}
    DIR_NAMES,        # {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
)

from engine.levels import ALL_LEVELS  # List of all level definitions
```

Use `gym.make(level_id)` to create environments. Use `env.reset(seed=N)` to set up a specific instance. Use `env.step(action_int)` to take actions.

Good luck!
