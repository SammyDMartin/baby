"""
BabyAI Challenge Harness

Use this to:
  1. View any challenge level (renders the grid as text)
  2. Submit and verify your action sequences
  3. Run your solver against the full test suite

Usage:
  python challenge.py show <level_id> [seed]        Show a level
  python challenge.py verify <level_id> <seed> "action1 action2 ..."  Verify actions
  python challenge.py list                           List all levels
  python challenge.py suite <your_solver_module>     Run full test suite
"""
import sys
import gymnasium as gym
from engine.grid import render_grid, get_grid_info, find_objects, ACTION_MAP, DIR_NAMES, DIR_ARROWS
from engine.levels import ALL_LEVELS
from engine.impossible import IMPOSSIBLE_LEVELS


def list_levels():
    """List all challenge levels with difficulty ratings."""
    print("=" * 72)
    print("BabyAI CHALLENGE LEVELS")
    print("=" * 72)
    print()
    for lvl in ALL_LEVELS:
        d = lvl['difficulty']
        if d <= 3:
            tier = "Easy"
        elif d <= 6:
            tier = "Medium"
        elif d <= 9:
            tier = "Hard"
        elif d <= 13:
            tier = "Nightmare"
        elif d <= 15:
            tier = "Impossible"
        else:
            tier = "Ultra"
        print(f"  [{tier:10s} D{d:2d}]  {lvl['id']}")
        print(f"                    {lvl['name']}: {lvl['desc']}")
        print()


def show_level(env_name, seed=42):
    """Render a level for manual inspection."""
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    grid = env.unwrapped.grid
    agent_pos = tuple(env.unwrapped.agent_pos)
    agent_dir = env.unwrapped.agent_dir
    w, h = grid.width, grid.height

    # Find target objects
    mission = obs['mission']

    print(f"{'=' * 72}")
    print(f"Level: {env_name}  (seed={seed})")
    print(f"Mission: {mission}")
    print(f"Agent: ({agent_pos[0]},{agent_pos[1]}) facing {DIR_NAMES[agent_dir]}")

    carrying = env.unwrapped.carrying
    if carrying:
        print(f"Carrying: {carrying.color} {carrying.type}")

    print(f"Grid: {w}x{h}")
    print(f"{'=' * 72}")
    print()

    # Render grid
    header = "    " + "".join(f"{x:>3}" for x in range(w))
    print(header)
    for y in range(h):
        row = f"{y:2d}  "
        for x in range(w):
            if (x, y) == agent_pos:
                row += f" A{DIR_ARROWS[agent_dir]}"
            else:
                cell = grid.get(x, y)
                if cell is None:
                    row += "  ."
                elif cell.type == 'wall':
                    row += " ##"
                elif cell.type == 'door':
                    c = cell.color[0].upper()
                    s = 'O' if cell.is_open else ('L' if cell.is_locked else 'C')
                    row += f" {c}{s}"
                elif cell.type == 'key':
                    row += f" {cell.color[0].upper()}k"
                elif cell.type == 'ball':
                    row += f" {cell.color[0].upper()}o"
                elif cell.type == 'box':
                    row += f" {cell.color[0].upper()}x"
                elif cell.type == 'goal':
                    row += " GL"
                elif cell.type == 'lava':
                    row += " LA"
                else:
                    row += f" {cell.type[:2]}"
        print(row)

    print()
    print("Legend:")
    print("  ## = wall   A>/Av/A</A^ = agent (facing direction)")
    print("  XC = closed door (X=color)   XO = open door   XL = locked door")
    print("  Xk = key   Xo = ball   Xx = box   (X = color: R G B P Y E=grey)")
    print()

    # List key objects
    print("Objects:")
    for y in range(h):
        for x in range(w):
            cell = grid.get(x, y)
            if cell and cell.type not in ('wall', 'floor'):
                extra = ""
                if cell.type == 'door':
                    state = 'open' if cell.is_open else ('locked' if cell.is_locked else 'closed')
                    extra = f" ({state})"
                color = getattr(cell, 'color', '')
                if color:
                    print(f"  ({x:2d},{y:2d}) {color} {cell.type}{extra}")

    env.close()
    print()
    print("Actions: left, right, forward, pickup, drop, toggle, done")
    print("  left/right = turn in place")
    print("  forward = move one cell in facing direction")
    print("  toggle = open/close/unlock the door you're facing")
    print("  pickup = pick up the object you're facing")
    print("  drop = drop carried object in front of you")


def verify_actions(env_name, seed, actions_str):
    """Verify an action sequence against a level."""
    actions = actions_str.strip().split()

    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    mission = obs['mission']

    print(f"Level: {env_name}  (seed={seed})")
    print(f"Mission: {mission}")
    print(f"Actions: {len(actions)}")
    print()

    total_reward = 0
    for i, a in enumerate(actions):
        if a not in ACTION_MAP:
            print(f"  INVALID action '{a}' at step {i+1}")
            env.close()
            return False

        obs, reward, done, truncated, _ = env.step(ACTION_MAP[a])
        total_reward += reward

        if done:
            success = reward > 0
            if success:
                print(f"  SUCCESS at step {i+1}/{len(actions)}")
                print(f"  Reward: {total_reward:.3f}")
            else:
                print(f"  FAILED at step {i+1}/{len(actions)}")
            env.close()
            return success

        if truncated:
            print(f"  TRUNCATED at step {i+1} (too many steps)")
            env.close()
            return False

    pos = tuple(env.unwrapped.agent_pos)
    d = env.unwrapped.agent_dir
    print(f"  INCOMPLETE after {len(actions)} steps")
    print(f"  Agent at ({pos[0]},{pos[1]}) facing {DIR_NAMES[d]}")
    env.close()
    return False


def run_suite(solver_module_name):
    """Run a solver module against all levels.

    The solver module must have a function with ONE of these signatures:

        solve(env_name, seed) -> list of action strings
            Solver creates its own env internally. Actions are verified
            by the solver itself (we trust the result).

        solve_with_env(env) -> (success: bool, actions: list[str])
            Solver receives a pre-reset env. Steps it directly and
            returns whether it succeeded plus the actions taken.

    The suite creates each env, passes it to the solver, and checks success.
    """
    import importlib
    mod = importlib.import_module(solver_module_name)

    has_solve_env = hasattr(mod, 'solve_with_env')
    solve_fn = getattr(mod, 'solve', None)

    seeds_per_level = 3
    total_wins, total_tests = 0, 0

    print(f"{'=' * 72}")
    print(f"RUNNING FULL TEST SUITE ({len(ALL_LEVELS)} levels x {seeds_per_level} seeds)")
    print(f"{'=' * 72}")
    print()

    results = []
    for lvl in ALL_LEVELS:
        wins = 0
        for seed in range(42, 42 + seeds_per_level):
            try:
                if has_solve_env:
                    env = gym.make(lvl['id'])
                    env.reset(seed=seed)
                    success, actions = mod.solve_with_env(env)
                    env.close()
                    ok = success
                elif solve_fn:
                    # Solver manages its own env — verify by replay
                    actions = solve_fn(lvl['id'], seed)
                    ok = verify_actions_quiet(lvl['id'], seed, actions)
                else:
                    raise ValueError("Solver needs solve() or solve_with_env()")

                if ok:
                    wins += 1
            except Exception as e:
                print(f"  ERROR on {lvl['name']} seed={seed}: {e}")
            total_tests += 1

        total_wins += wins
        rate = wins / seeds_per_level
        bar = "#" * int(rate * 20) + "." * (20 - int(rate * 20))
        results.append((lvl['name'], wins, seeds_per_level, rate))
        print(f"  {lvl['name']:35s} [{bar}] {wins}/{seeds_per_level}")

    print()
    print(f"{'=' * 72}")
    print(f"OVERALL: {total_wins}/{total_tests} ({total_wins * 100 // max(total_tests, 1)}%)")
    print(f"{'=' * 72}")
    return total_wins, total_tests


def verify_actions_quiet(env_name, seed, actions):
    """Verify without printing."""
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    for a in actions:
        if a not in ACTION_MAP:
            env.close()
            return False
        obs, reward, done, truncated, _ = env.step(ACTION_MAP[a])
        if done:
            env.close()
            return reward > 0
        if truncated:
            env.close()
            return False
    env.close()
    return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'list':
        list_levels()
    elif cmd == 'show':
        env_name = sys.argv[2] if len(sys.argv) > 2 else 'BabyAI-GoToRedBallNoDists-v0'
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
        show_level(env_name, seed)
    elif cmd == 'verify':
        env_name = sys.argv[2]
        seed = int(sys.argv[3])
        actions_str = sys.argv[4]
        verify_actions(env_name, seed, actions_str)
    elif cmd == 'suite':
        solver_mod = sys.argv[2]
        run_suite(solver_mod)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
