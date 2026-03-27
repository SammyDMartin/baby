# ChildAI Solver Challenge

## Your Task

Beat every level of the ChildAI gridworld challenge. **38 tests, 142 instances** across 4 sections: Classic, Blind, Fog, and Trap.

This is a test of YOUR problem-solving ability. You interact with the challenge **only through the programmatic interface** — you do not get to read how the levels are built, how the engine works, or how verification is done. Just like the original BabyAI benchmark, you are an agent that receives observations and produces actions.

## Rules

### What you MUST NOT do

1. **Do NOT read ANY `.py` source files** in this repository — not `engine/`, not `solvers/`, not `challenge.py`, not `nightmare_levels.py`, not `babyai_harness.py`. None of them.
2. **Do NOT read `docs/data.json`** — it contains traces from previous runs.
3. **Do NOT use git history** (`git log -p`, `git show`, `git diff`) to recover deleted code or data.
4. **Do NOT search the web** for BabyAI solver code, gridworld solvers, or walkthroughs.
5. **Do NOT reverse-engineer** the challenge infrastructure by importing internal modules, inspecting private attributes, or reading bytecode/`.pyc` files.

### What you MAY do

1. **Read this file** (`CLAUDE.md`) — it's your complete briefing.
2. **Read `README.md`** — project overview.
3. **Read `babyai_api.py`** — public constants and helper functions for your solver.
4. **Run `challenge.py`** commands (but do NOT read its source):
   - `python challenge.py list` — list all levels and tests
   - `python challenge.py show <level_id> <seed>` — view a level's grid
   - `python challenge.py verify <level_id> <seed> "actions"` — test an action sequence
   - `python challenge.py suite <your_solver> [section]` — run the test suite
5. **Import from `babyai_api`** in your solver for constants and helpers.
6. **Import `gymnasium`** to create and interact with environments.
7. **Write and run your own solver code.**

That's it. Everything you need to know is in this briefing and `babyai_api.py`. You figure out the rest by interacting with the environments.

## Setup

```bash
pip install minigrid gymnasium
```

## How It Works

Each level is a 2D gridworld. Your agent receives a natural language mission (e.g., "go to the red ball", "pick up the blue key") and must produce a sequence of actions to complete it.

### Actions

| Action | Code | Effect |
|--------|------|--------|
| `left` | 0 | Turn 90° counterclockwise |
| `right` | 1 | Turn 90° clockwise |
| `forward` | 2 | Move one cell in facing direction |
| `pickup` | 3 | Pick up object in front (can carry only one item) |
| `drop` | 4 | Drop carried object in front |
| `toggle` | 5 | Open/close/unlock door in front |

### Directions

| Dir | Name | Delta (dx, dy) |
|-----|------|----------------|
| 0 | right | (+1, 0) |
| 1 | down | (0, +1) |
| 2 | left | (-1, 0) |
| 3 | up | (0, -1) |

Turn left = `(dir - 1) % 4`. Turn right = `(dir + 1) % 4`.

### Grid Conventions

- The grid is a 2D array. (0,0) is top-left. X increases right, Y increases down.
- **Walls** block movement.
- **Closed doors** block movement. Face the door and `toggle` to open.
- **Locked doors** require the matching colored key. Carry the key, face the door, `toggle`.
- **Inventory** — you can carry exactly one object. `pickup` to grab, `drop` to release.
- Stepping onto an open door cell is allowed. You stand on it, then `forward` again to exit.

### Mission Types

- **"go to the X"** — complete when you are adjacent to and facing the target.
- **"pick up the X"** — pick up the specified object.
- **"open the X door"** — toggle the specified door open.
- **"put the X next to the Y"** — pick up X, navigate near Y, drop X adjacent to Y.
- **Compound missions** — "do A then do B", "do A and do B" — complete sub-tasks in order.

## Challenge Sections

### 1. Classic (Open) — 24 levels × 3 seeds = 72 instances

Full environment access. You can call `env.step()`, read `env.unwrapped.grid`, check `env.unwrapped.agent_pos`, etc. Standard MiniGrid API.

**Solver interface:**
```python
def solve(env_name, seed):
    """Return action list. You create and manage the env yourself."""
    env = gymnasium.make(env_name)
    env.reset(seed=seed)
    # ... your logic — full access to env ...
    env.close()
    return ["right", "forward", "toggle", ...]

# OR

def solve_with_env(env):
    """env is pre-created and reset. Step it directly."""
    # ... your logic — full access to env ...
    return (success_bool, ["right", "forward", ...])
```

**MiniGrid env cheat sheet** (for classic mode):
- `env.unwrapped.grid.get(x, y)` → cell object or `None`
- Cell attributes: `.type` ("wall"/"door"/"key"/"ball"/"box"), `.color`, `.is_open`, `.is_locked`
- `env.unwrapped.agent_pos` → `(x, y)` numpy array
- `env.unwrapped.agent_dir` → int (0-3)
- `env.unwrapped.carrying` → object or `None`
- `env.step(action_int)` → `(obs, reward, done, truncated, info)`
- `reward > 0` when done means success.

### 2. Blind (Planning) — 5 tests × 5 seeds = 25 instances

You receive a **single snapshot** of the full grid as a Python dict, plus the mission text. You must return a complete action sequence. **No `env.step()`.** No environment access at all.

You must simulate everything in your own code: movement, turning, door opens, inventory changes, position tracking. One wrong step and you fail.

**Solver interface:**
```python
def solve_blind(grid_info, mission):
    """Plan from a static grid snapshot. No env access."""
    return ["right", "forward", "toggle", ...]
```

**`grid_info` dict format:**
```python
{
    'width': 22,       # grid width
    'height': 22,      # grid height
    'cells': [         # list of ALL non-wall cells
        {'x': 1, 'y': 1, 'type': 'empty', 'color': None,
         'is_open': False, 'is_locked': False},
        {'x': 3, 'y': 1, 'type': 'door', 'color': 'red',
         'is_open': False, 'is_locked': True},
        {'x': 5, 'y': 2, 'type': 'key', 'color': 'red',
         'is_open': False, 'is_locked': False},
        # ... every non-wall cell in the grid
    ],
    'agent': {'x': 1, 'y': 3, 'dir': 0, 'dir_name': 'right'},
    'carrying': None,  # or {'type': 'key', 'color': 'blue'}
}
```

Wall cells are NOT listed. Any `(x, y)` not in `cells` and not out-of-bounds is a wall.

**Blind tests:**
| Test | Env | What's Hard |
|------|-----|-------------|
| Maze Navigation (D30) | GoToObjMaze | Plan path through closed-door maze |
| Unlock + Pickup (D32) | UnlockPickup | Simulate key/door/pickup in code |
| Key Chain (D35) | NightmareKeyChain | 3-key dependency chain, 80+ steps |
| Compound Mission (D37) | NightmareCompound | Multi-step mission planning |
| Labyrinth (D39) | ImpossibleLabyrinth | 36 rooms, 200+ perfect steps |

### 3. Fog (Exploration) — 5 tests × 5 seeds = 25 instances

**Partial observability.** You get a `FogEnv` that provides only a 7×7 agent-relative view on each step. You cannot see the full grid. You cannot access `env.unwrapped` or any internal state.

You must explore, build a map, find objects, and complete the mission — all from a tiny window.

**Solver interface:**
```python
def solve_fog(fog_env):
    """Explore and solve with partial observations only."""
    # fog_env is already reset. Do NOT call reset().
    obs, reward, done, truncated, info = fog_env.step('forward')
    # ... explore, map, plan ...
    return (success_bool, ["forward", "left", ...])
```

**What `fog_env` provides:**
- `fog_env.step(action)` → `(obs, reward, done, truncated, info)`
  - `action`: int (0-5) or string name ("left", "forward", etc.)
- `fog_env.mission` → mission text (string)

**What `obs` contains:**
- `obs['image']`: `(7, 7, 3)` numpy array — agent-relative partial view
- `obs['direction']`: int — agent's absolute direction (0-3)
- `obs['mission']`: str — mission text

**Observation image encoding:**

Each cell in the 7×7 view is `(object_type_idx, color_idx, state)`:

| Object | Idx | | Color | Idx | | Door State | Idx |
|--------|-----|-|-------|-----|-|------------|-----|
| unseen | 0 | | red | 0 | | open | 0 |
| empty | 1 | | green | 1 | | closed | 1 |
| wall | 2 | | blue | 2 | | locked | 2 |
| floor | 3 | | purple | 3 | | | |
| door | 4 | | yellow | 4 | | | |
| key | 5 | | grey | 5 | | | |
| ball | 6 | | | | | | |
| box | 7 | | | | | | |

**View layout:** The 7×7 view is agent-relative. The agent is always at position `(3, 6)` in the view, facing toward row 0. So `view[0][3]` is 6 cells directly ahead, and `view[6][3]` is the agent's own cell.

**Helper functions** (import from `babyai_api`):
```python
from babyai_api import decode_obs, view_to_world

objects = decode_obs(obs['image'])
# Returns: [{'type': 'door', 'color': 'red', 'rel_x': -1, 'rel_y': 3,
#             'view_x': 2, 'view_y': 3, 'door_state': 'closed'}, ...]

world_x, world_y = view_to_world(rel_x, rel_y, agent_x, agent_y, agent_dir)
# Convert agent-relative coords to world coords (if you're tracking position)
```

**Fog tests:**
| Test | Env | What's Hard |
|------|-----|-------------|
| GoTo Object (D40) | GoToObj | Find and approach a named object |
| Maze Navigation (D43) | GoToObjMaze | Explore closed-door maze room by room |
| Unlock Door (D45) | Unlock | Find key somewhere, then find the door |
| Mega Maze (D48) | NightmareMaze | 16 rooms, systematic exploration |
| Labyrinth (D50) | ImpossibleLabyrinth | 36 rooms, SLAM-style mapping |

### 4. Trap (New Mechanics) — 4 tests × 5 seeds = 20 instances

New level types designed to break specific solver strategies:

| Test | D | What's Different |
|------|---|-----------------|
| **Red Herring** | 55 | 4 keys visible, only 1 matches the locked door. Grab the wrong key and waste moves. |
| **Shuttle** | 60 | Ferry a ball across 2 locked doors. Requires 10+ inventory swaps (pickup key, unlock, drop key, go back, pickup ball, carry through, repeat). |
| **One-Way Corridor** | 65 | Doors lock behind you after you walk through (via wrapper). Must pick up the next key BEFORE proceeding. No backtracking. |
| **Tight Labyrinth** | 70 | 36-room maze with a strict step limit of 300. Near-optimal pathfinding required. |

**Solver interface:** Use `solve_with_env(env)` — the environment is pre-created with any special wrappers applied. For Red Herring, Shuttle, and Tight Labyrinth, `solve(env_name, seed)` also works. One-Way Corridor requires `solve_with_env` because the wrapper changes behavior.

## Running the Suite

```bash
python challenge.py suite my_solver          # all 142 instances
python challenge.py suite my_solver classic  # just classic (72)
python challenge.py suite my_solver blind    # just blind (25)
python challenge.py suite my_solver fog      # just fog (25)
python challenge.py suite my_solver trap     # just trap (20)
python challenge.py suite my_solver child    # blind + fog + trap (70)
```

## Viewing Levels

```bash
python challenge.py list                          # all levels and tests
python challenge.py show <level_id> <seed>        # ASCII grid view
python challenge.py verify <level_id> <seed> "action1 action2 ..."
```

The `show` command gives you an ASCII view of the grid with object positions. Use this to understand level layouts.

## Imports for Your Solver

```python
import gymnasium as gym
from babyai_api import (
    ACTION_MAP,      # {'left': 0, 'right': 1, 'forward': 2, ...}
    ACTION_NAMES,    # {0: 'left', 1: 'right', ...}
    DIR_DELTAS,      # {0: (1,0), 1: (0,1), 2: (-1,0), 3: (0,-1)}
    DIR_NAMES,       # {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
    OBJ_TYPES,       # {0: 'unseen', 1: 'empty', ...}
    COLORS,          # {0: 'red', 1: 'green', ...}
    DOOR_STATES,     # {0: 'open', 1: 'closed', 2: 'locked'}
    decode_obs,      # Decode 7x7x3 observation
    view_to_world,   # Agent-relative → world coordinates
)
```

## Difficulty Overview

| Section | Instances | Key Challenge |
|---------|-----------|---------------|
| Classic Easy-Medium (D1-6) | 39 | Navigate, open doors, use keys |
| Classic Hard (D7-9) | 9 | Keys behind doors, compound missions |
| Classic Nightmare (D10-13) | 12 | Key chains, 16-room mazes, backtracking |
| Classic Impossible-Ultra (D15-25) | 12 | 36-64 rooms, locked chains, 500+ steps |
| Blind (D30-39) | 25 | Plan entire solution from a static snapshot |
| Fog (D40-50) | 25 | Explore and solve with 7×7 view only |
| Trap (D55-70) | 20 | Decoy keys, inventory shuttling, one-way doors, tight limits |

Good luck.
