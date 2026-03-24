"""
ChildAI Challenge Harness

Upgraded from BabyAI: adds Blind, Fog, and Trap modes alongside Classic.

Usage:
  python challenge.py list                           List all levels and tests
  python challenge.py show <level_id> [seed]         Show a level grid
  python challenge.py verify <level_id> <seed> "actions"  Verify action sequence
  python challenge.py suite <solver_module> [section] Run test suite

Sections: classic, blind, fog, trap, child (=blind+fog+trap), all (default)
"""
import sys
import importlib
import gymnasium as gym
from engine.grid import render_grid, get_grid_info, find_objects, grid_to_dict
from engine.grid import ACTION_MAP, DIR_NAMES, DIR_ARROWS
from engine.levels import ALL_LEVELS
from engine.child import CHILD_TESTS, BLIND_TESTS, FOG_TESTS, TRAP_TESTS
from engine.wrappers import FogEnv, OneWayDoorWrapper


# ── List / Show / Verify (unchanged from BabyAI) ───────────────────────

def list_levels():
    print("=" * 76)
    print("ChildAI CHALLENGE — LEVELS & TESTS")
    print("=" * 76)

    print("\n── Classic Levels (open mode, 3 seeds each) ──\n")
    for lvl in ALL_LEVELS:
        d = lvl['difficulty']
        if d <= 3: tier = "Easy"
        elif d <= 6: tier = "Medium"
        elif d <= 9: tier = "Hard"
        elif d <= 13: tier = "Nightmare"
        elif d <= 15: tier = "Impossible"
        else: tier = "Ultra"
        print(f"  [{tier:10s} D{d:2d}]  {lvl['id']}")
        print(f"                    {lvl['name']}: {lvl['desc']}")
        print()

    for section_name, tests in [("Blind", BLIND_TESTS), ("Fog", FOG_TESTS), ("Trap", TRAP_TESTS)]:
        print(f"\n── {section_name} Tests (5 seeds each) ──\n")
        for t in tests:
            w = f" [wrapper: {t['wrapper']}]" if t.get('wrapper') else ""
            print(f"  [D{t['difficulty']:2d}]  {t['name']}{w}")
            print(f"          env: {t['id']}  mode: {t['mode']}")
            print(f"          {t['desc']}")
            print()

    classic_n = len(ALL_LEVELS) * 3
    child_n = sum(t['seeds'] for t in CHILD_TESTS)
    print(f"Total: {len(ALL_LEVELS)} classic levels ({classic_n} instances)")
    print(f"     + {len(CHILD_TESTS)} child tests ({child_n} instances)")
    print(f"     = {classic_n + child_n} total instances")


def show_level(env_name, seed=42):
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    grid = env.unwrapped.grid
    agent_pos = tuple(env.unwrapped.agent_pos)
    agent_dir = env.unwrapped.agent_dir
    w, h = grid.width, grid.height
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
    print("  XC = closed door   XO = open door   XL = locked door")
    print("  Xk = key   Xo = ball   Xx = box   (X = R G B P Y E=grey)")
    print()

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
    print("Actions: left, right, forward, pickup, drop, toggle")


def verify_actions(env_name, seed, actions_str):
    actions = actions_str.strip().split()
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    mission = obs['mission']

    print(f"Level: {env_name}  (seed={seed})")
    print(f"Mission: {mission}")
    print(f"Actions: {len(actions)}")
    print()

    for i, a in enumerate(actions):
        if a not in ACTION_MAP:
            print(f"  INVALID action '{a}' at step {i+1}")
            env.close()
            return False
        obs, reward, done, truncated, _ = env.step(ACTION_MAP[a])
        if done:
            success = reward > 0
            print(f"  {'SUCCESS' if success else 'FAILED'} at step {i+1}/{len(actions)}")
            env.close()
            return success
        if truncated:
            print(f"  TRUNCATED at step {i+1}")
            env.close()
            return False

    pos = tuple(env.unwrapped.agent_pos)
    d = env.unwrapped.agent_dir
    print(f"  INCOMPLETE after {len(actions)} steps")
    print(f"  Agent at ({pos[0]},{pos[1]}) facing {DIR_NAMES[d]}")
    env.close()
    return False


# ── Replay verification ────────────────────────────────────────────────

def replay_verify(env, actions):
    """Replay action list on a pre-reset env. Returns True if mission completed."""
    for a in actions:
        if isinstance(a, str):
            a_int = ACTION_MAP.get(a)
            if a_int is None:
                return False
        else:
            a_int = int(a)
        obs, reward, done, truncated, _ = env.step(a_int)
        if done:
            return reward > 0
        if truncated:
            return False
    return False


def verify_quiet(env_name, seed, actions):
    """Verify actions by replay on a fresh env."""
    env = gym.make(env_name)
    env.reset(seed=seed)
    ok = replay_verify(env, actions)
    env.close()
    return ok


# ── Test runners for each mode ─────────────────────────────────────────

def run_classic_test(solver_mod, env_id, seed):
    """Classic/open mode: solver gets full env access."""
    has_solve_env = hasattr(solver_mod, 'solve_with_env')
    has_solve = hasattr(solver_mod, 'solve')

    if has_solve_env:
        env = gym.make(env_id)
        env.reset(seed=seed)
        success, actions = solver_mod.solve_with_env(env)
        env.close()
        return success
    elif has_solve:
        actions = solver_mod.solve(env_id, seed)
        return verify_quiet(env_id, seed, actions)
    return None


def run_blind_test(solver_mod, env_id, seed):
    """Blind mode: solver gets grid snapshot + mission, no env.step()."""
    if not hasattr(solver_mod, 'solve_blind'):
        return None

    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    grid_info = grid_to_dict(env)
    mission = obs['mission']
    env.close()

    actions = solver_mod.solve_blind(grid_info, mission)
    return verify_quiet(env_id, seed, actions)


def run_fog_test(solver_mod, env_id, seed, wrapper=None):
    """Fog mode: solver gets FogEnv (partial obs only)."""
    if not hasattr(solver_mod, 'solve_fog'):
        return None

    env = gym.make(env_id)
    if wrapper == 'oneway':
        env = OneWayDoorWrapper(env)
    fog = FogEnv(env)
    fog.reset(seed=seed)

    success, actions = solver_mod.solve_fog(fog)
    recorded = fog.actions_taken
    fog.close()

    # Verify by replay on fresh env (with same wrapper)
    env2 = gym.make(env_id)
    if wrapper == 'oneway':
        env2 = OneWayDoorWrapper(env2)
    env2.reset(seed=seed)
    ok = replay_verify(env2, recorded)
    env2.close()
    return ok


def run_open_test_with_wrapper(solver_mod, env_id, seed, wrapper):
    """Open mode with a wrapper (e.g. one-way doors)."""
    if not hasattr(solver_mod, 'solve_with_env'):
        return None  # solve() can't handle wrappers

    env = gym.make(env_id)
    if wrapper == 'oneway':
        env = OneWayDoorWrapper(env)
    env.reset(seed=seed)

    success, actions = solver_mod.solve_with_env(env)
    env.close()

    # Verify by replay with wrapper
    env2 = gym.make(env_id)
    if wrapper == 'oneway':
        env2 = OneWayDoorWrapper(env2)
    env2.reset(seed=seed)
    ok = replay_verify(env2, actions)
    env2.close()
    return ok


def run_test(solver_mod, test_def, seed):
    """Run a single test instance. Returns True/False/None (None=skipped)."""
    env_id = test_def['id']
    mode = test_def.get('mode', 'open')
    wrapper = test_def.get('wrapper')

    try:
        if mode == 'blind':
            return run_blind_test(solver_mod, env_id, seed)
        elif mode == 'fog':
            return run_fog_test(solver_mod, env_id, seed, wrapper)
        elif mode == 'open':
            if wrapper:
                return run_open_test_with_wrapper(solver_mod, env_id, seed, wrapper)
            return run_classic_test(solver_mod, env_id, seed)
    except Exception as e:
        print(f"    ERROR: {e}")
        return False
    return None


# ── Suite runner ────────────────────────────────────────────────────────

def run_suite(solver_module_name, section='all'):
    mod = importlib.import_module(solver_module_name)

    sections = {
        'classic': ('Classic (Open)', ALL_LEVELS, 3, 'open'),
        'blind':   ('Blind (Planning)', BLIND_TESTS, None, None),
        'fog':     ('Fog (Exploration)', FOG_TESTS, None, None),
        'trap':    ('Trap (New Mechanics)', TRAP_TESTS, None, None),
    }

    if section == 'all':
        run_sections = ['classic', 'blind', 'fog', 'trap']
    elif section == 'child':
        run_sections = ['blind', 'fog', 'trap']
    elif section in sections:
        run_sections = [section]
    else:
        print(f"Unknown section: {section}")
        print("Options: classic, blind, fog, trap, child, all")
        return 0, 0

    grand_wins, grand_tests, grand_skipped = 0, 0, 0

    for sec_name in run_sections:
        label, tests, fixed_seeds, fixed_mode = sections[sec_name]

        print(f"\n{'=' * 72}")
        print(f"  {label}")
        print(f"{'=' * 72}\n")

        sec_wins, sec_tests, sec_skipped = 0, 0, 0

        for test_def in tests:
            # Classic levels use a different format
            if fixed_seeds is not None:
                # Classic level dict -> test_def format
                td = {
                    'id': test_def['id'],
                    'name': test_def['name'],
                    'mode': fixed_mode,
                    'seeds': fixed_seeds,
                }
                seeds = range(42, 42 + fixed_seeds)
            else:
                td = test_def
                seeds = range(42, 42 + test_def['seeds'])

            wins = 0
            skips = 0
            for seed in seeds:
                result = run_test(mod, td, seed)
                sec_tests += 1
                if result is None:
                    skips += 1
                    sec_skipped += 1
                elif result:
                    wins += 1
                    sec_wins += 1

            total_for_level = len(list(seeds))
            if skips == total_for_level:
                bar = "." * 20
                status = "SKIP"
            else:
                rate = wins / max(total_for_level - skips, 1)
                bar = "#" * int(rate * 20) + "." * (20 - int(rate * 20))
                status = f"{wins}/{total_for_level}"

            name = td.get('name', td['id'])
            wrapper_tag = f" [{td.get('wrapper', '')}]" if td.get('wrapper') else ""
            print(f"  {name:35s}{wrapper_tag:10s} [{bar}] {status}")

        grand_wins += sec_wins
        grand_tests += sec_tests
        grand_skipped += sec_skipped

        tested = sec_tests - sec_skipped
        print(f"\n  Section: {sec_wins}/{tested}" +
              (f" ({sec_skipped} skipped)" if sec_skipped else ""))

    print(f"\n{'=' * 72}")
    tested = grand_tests - grand_skipped
    pct = grand_wins * 100 // max(tested, 1)
    print(f"OVERALL: {grand_wins}/{tested} ({pct}%)" +
          (f"  [{grand_skipped} skipped]" if grand_skipped else ""))
    print(f"{'=' * 72}")
    return grand_wins, grand_tests


# ── CLI ─────────────────────────────────────────────────────────────────

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
        section = sys.argv[3] if len(sys.argv) > 3 else 'all'
        run_suite(solver_mod, section)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
