# BabyAI Solver & Visualizer

Programmatic solver for [BabyAI](https://github.com/Farama-Foundation/Minigrid) gridworld levels with an interactive web visualizer. Includes 17 standard levels + 4 custom nightmare challenges.

**[Live Demo →](https://sammydmartin.github.io/baby/)**

## Results

**95% solve rate** (60/63 level instances across 21 levels, 3 seeds each)

| Category | Levels | Solve Rate |
|----------|--------|------------|
| Standard (Easy) | GoTo, Open Door, Pickup | 100% |
| Standard (Medium) | Maze, Unlock, Put Next To | 100% |
| Standard (Hard) | KeyCorridor, BlockedUnlock, Boss | 83% |
| Nightmare | KeyChain, MegaMaze, Backtrack, Compound | 100% |

## Project Structure

```
engine/                 # Core solver module
  grid.py              # Grid rendering (ASCII + JSON for web)
  pathfinding.py       # BFS pathfinding, flood fill, door finding
  solver.py            # Mission parser + subgoal solvers
  levels.py            # Level definitions (standard + nightmare)
docs/                  # GitHub Pages site
  index.html           # Interactive visualizer
  data.json            # Pre-generated solver results
generate_data.py       # Runs solver, exports JSON for web
```

### Legacy files (original exploration)
```
babyai_harness.py      # Original text rendering harness
solve_babyai.py        # Solver v1
solve_v2.py            # Solver v2 (79% baseline)
show_tasks.py          # Task viewer script
nightmare_levels.py    # Original nightmare level definitions
```

## Nightmare Levels

| Level | Design | Difficulty |
|-------|--------|------------|
| **Key Chain** | 4-room corridor, 3 locked doors, chained key dependencies | Must manage single-item inventory across 3 key pickups |
| **Mega Maze** | 4×4 room grid (16 rooms), all doors closed | 100+ step navigation sequences |
| **Backtrack** | 5-room corridor, key at far end, target at other end behind locked door | Full double traversal (~150 steps) |
| **Compound** | 3-room corridor, put-next-to then pickup | Multi-phase inventory management |

## Setup

```bash
pip install minigrid gymnasium
```

## Running

```bash
# Run the solver test suite
python -c "from engine.solver import solve_level; from engine.levels import ALL_LEVELS
for l in ALL_LEVELS:
    r = solve_level(l['id'], seed=42)
    print(f'{l[\"name\"]:35s} {\"OK\" if r[\"success\"] else \"FAIL\"} ({r[\"num_steps\"]} steps)')"

# Regenerate web data
python generate_data.py

# View the web interface locally
cd docs && python -m http.server 8000
```

## Solver Architecture

The solver uses BFS pathfinding with a hierarchical planning approach:

1. **Mission parsing**: Regex-based parser handles goto, pickup, open, put-next-to, and compound missions
2. **Subgoal solving**: Each mission type has a dedicated solver that plans and executes actions
3. **Door navigation**: Iterative door-opening when targets are behind closed/locked doors
4. **Key chain resolution**: Recursive key dependency resolution (key A unlocks door to key B which unlocks door to target)
5. **Blocker handling**: Detects and moves objects blocking doors before unlocking

## Credits

- BabyAI platform: Maxime Chevalier-Boisvert, Dzmitry Bahdanau, Yoshua Bengio et al. ([paper](https://arxiv.org/abs/1810.08272))
- This project: Built by Claude (Anthropic) in conversation with Sammy Martin, March 2026
