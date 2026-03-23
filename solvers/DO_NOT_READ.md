# ⚠️ STOP - Do Not Read These Files ⚠️

This folder contains **reference solvers and solution notes** for the BabyAI challenge.

**If you are attempting to solve the challenge yourself, do not read any files in this directory.** Reading the solvers would defeat the purpose of the challenge — the whole point is to solve the levels by reasoning about the grid yourself.

## What's in here

| File | Contents |
|------|----------|
| `solver.py` | Full programmatic solver (BFS pathfinding, mission parsing, multi-room navigation) |
| `pathfinding.py` | BFS implementation, flood fill, door-finding algorithms |
| `solve_v1.py` | Earlier solver attempt (simpler, more bugs) |
| `solve_v2.py` | Intermediate solver (79% success rate) |
| `generate_data.py` | Script that runs solver on all levels to generate web visualization data |
| `honest_attempt.py` | Framework for verifying manual solve attempts |
| `manual_test.py` | Another test harness |
| `chat_log.md.txt` | Development conversation log |

## If you've already solved the challenge

Feel free to look! You can also run the reference solver to compare your approach:

```bash
python -c "from solvers.solver import solve_level; from engine.levels import ALL_LEVELS
for l in ALL_LEVELS:
    r = solve_level(l['id'], seed=42)
    print(f'{l[\"name\"]:35s} {\"OK\" if r[\"success\"] else \"FAIL\"} ({r[\"num_steps\"]} steps)')"
```

The reference solver achieves 100% on all levels including the Impossible Labyrinth.
