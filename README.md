# ChildAI Solver Challenge

A harder evolution of the [BabyAI](https://github.com/Farama-Foundation/Minigrid) challenge. Beyond the original 24 classic levels, ChildAI adds **Blind** (pure planning from a grid snapshot), **Fog** (partial observability — 7x7 view only), and **Trap** (new mechanics that defeat naive solvers).

**38 tests, 142 instances.** Classic levels test open-world BFS. ChildAI tests whether you can plan without stepping, explore without seeing, and adapt when the rules change.

**[Web Visualizer →](https://sammydmartin.github.io/baby/)**

## The Challenge

| Section | Tests | Seeds | Instances | Solver Interface |
|---------|-------|-------|-----------|------------------|
| **Classic** (open) | 24 levels | 3 | 72 | `solve(env_name, seed)` or `solve_with_env(env)` |
| **Blind** (planning) | 5 tests | 5 | 25 | `solve_blind(grid_info, mission)` |
| **Fog** (exploration) | 5 tests | 5 | 25 | `solve_fog(fog_env)` |
| **Trap** (new mechanics) | 4 tests | 5 | 20 | `solve_with_env(env)` |

### Classic Levels (D1–D25)

| Tier | What's Involved |
|------|-----------------|
| **Easy** (D1-3) | Navigate to objects, open doors. Single room. |
| **Medium** (D4-6) | Pick up objects, use keys, multi-room navigation. |
| **Hard** (D7-9) | Keys behind doors, compound missions. |
| **Nightmare** (D10-13) | Chained keys, 16-room mazes, 150+ steps. |
| **Impossible** (D15) | 36-room labyrinth, 60 doors, 200-500+ steps. |
| **Ultra** (D20-25) | 16-64 rooms, locked chains, compound missions, 500-1000+ steps. |

### ChildAI Tests (D30–D70)

| Mode | Test | D | What's Hard |
|------|------|---|-------------|
| **Blind** | Maze Navigation | 30 | Plan path through closed-door maze from grid snapshot |
| **Blind** | Unlock + Pickup | 32 | Key + door + pickup. Must simulate inventory in code |
| **Blind** | Key Chain | 35 | 3-key dependency chain. Plan 80+ actions with no env |
| **Blind** | Compound Mission | 37 | Multi-step mission. Plan from grid snapshot |
| **Blind** | Labyrinth | 39 | 36 rooms. Plan 200+ perfect actions from one snapshot |
| **Fog** | GoTo Object | 40 | Find object with 7x7 view. Simple exploration |
| **Fog** | Maze Navigation | 43 | Closed-door maze. Must explore room by room |
| **Fog** | Unlock Door | 45 | Find key somewhere in the grid, then find the door |
| **Fog** | Mega Maze | 48 | 16-room maze. Full exploration + backtracking |
| **Fog** | Labyrinth | 50 | 36 rooms. SLAM-style mapping from partial obs |
| **Trap** | Red Herring | 55 | 4 keys, only 1 is correct. Must reason about colors |
| **Trap** | Shuttle | 60 | Ferry ball across 2 locked doors. 10+ inventory swaps |
| **Trap** | One-Way Corridor | 65 | Doors lock behind you. Forward-only planning |
| **Trap** | Tight Labyrinth | 70 | 36 rooms, step limit 300. Near-optimal path required |

## Rules & Integrity

1. **Do NOT read files in `solvers/`.** Reference solutions live there.
2. **Do NOT search the web** for BabyAI solver implementations.
3. **Do NOT use git history** to recover deleted solver code.
4. **Do NOT read `docs/data.json`.**
5. **Write your own solution** from scratch.

You may freely read: `engine/`, `challenge.py`, and other non-solver files.

## Quick Start

```bash
pip install minigrid gymnasium

python challenge.py list                                    # List all levels + tests
python challenge.py show BabyAI-GoToRedBallNoDists-v0 42    # View a level
python challenge.py verify BabyAI-GoToRedBallNoDists-v0 42 "right right forward"

python challenge.py suite my_solver          # Run all sections
python challenge.py suite my_solver classic  # Just classic
python challenge.py suite my_solver child    # Just blind+fog+trap
python challenge.py suite my_solver blind    # Just blind
python challenge.py suite my_solver fog      # Just fog
python challenge.py suite my_solver trap     # Just trap
```

**Full instructions in [CLAUDE.md](CLAUDE.md)** — grid mechanics, solver interfaces, observation encoding, engine API.

## Project Structure

```
CLAUDE.md              # Challenge instructions (read this first)
challenge.py           # View levels, verify solutions, run test suite
engine/
  grid.py              # Grid state extraction, rendering, constants
  levels.py            # Standard + Nightmare level definitions
  impossible.py        # Impossible Labyrinth definition
  ultra.py             # Ultra-tier level definitions
  child.py             # ChildAI level classes + test specifications
  wrappers.py          # FogEnv, OneWayDoorWrapper, obs decode helpers
docs/                  # Web visualizer (GitHub Pages)
solvers/               # REFERENCE SOLUTIONS — do not read if attempting the challenge
```

## Writing Your Solver

```python
# my_solver.py
def solve(env_name, seed):
    """Classic open mode."""
    return ["right", "forward", "toggle", ...]

def solve_with_env(env):
    """Classic open mode with pre-reset env."""
    return success, actions

def solve_blind(grid_info, mission):
    """Blind mode. Grid dict + mission text. No env.step()."""
    return ["right", "forward", "toggle", ...]

def solve_fog(fog_env):
    """Fog mode. 7x7 partial view only. fog_env.step() available."""
    return success, actions
```

## Credits

- BabyAI platform: Chevalier-Boisvert, Bahdanau, Bengio et al. ([paper](https://arxiv.org/abs/1810.08272))
- Challenge design: Built by Claude (Anthropic) in conversation with Sammy Martin, March 2026
