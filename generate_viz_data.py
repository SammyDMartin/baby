#!/usr/bin/env python3
"""Generate visualization data for the BabyAI solver visualizer.

Runs the solver against all levels and outputs docs/data.json.
Can also embed the data directly into docs/index.html for offline use.

Usage:
    python generate_viz_data.py [solver_module] [--embed]

    solver_module: Python module with solve_with_env() (default: attempts.my_solver)
    --embed: Also inline data into docs/index.html so it works from file://
"""
import sys
import json
import importlib
import numpy as np
import gymnasium as gym


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from engine.grid import grid_to_dict
from engine.levels import ALL_LEVELS
from engine.impossible import IMPOSSIBLE_LEVELS

SEEDS = [42, 43, 44]


def run_suite(solver_module_name):
    mod = importlib.import_module(solver_module_name)
    solve_fn = getattr(mod, 'solve_with_env', None)

    all_level_defs = ALL_LEVELS + IMPOSSIBLE_LEVELS
    levels_data = []
    total_wins = 0
    total_tests = 0

    for ldef in all_level_defs:
        level_id = ldef['id']
        name = ldef.get('name', level_id)
        difficulty = ldef.get('difficulty', 0)
        desc = ldef.get('desc', '')

        runs = []
        wins = 0

        for seed in SEEDS:
            total_tests += 1
            env = gym.make(level_id)
            obs, _ = env.reset(seed=seed)
            mission = obs['mission']

            # Capture initial grid state
            initial_grid = grid_to_dict(env)

            try:
                if solve_fn:
                    success, actions = solve_fn(env)
                else:
                    actions = mod.solve(level_id, seed)
                    # Re-verify
                    env2 = gym.make(level_id)
                    env2.reset(seed=seed)
                    success = False
                    for a in actions:
                        from engine.grid import ACTION_MAP
                        o, r, done, trunc, _ = env2.step(ACTION_MAP[a])
                        if done:
                            success = r > 0
                            break
                    env2.close()
            except Exception as e:
                print(f"  ERROR on {name} seed={seed}: {e}")
                success = False
                actions = []

            if success:
                wins += 1
                total_wins += 1

            runs.append({
                'seed': seed,
                'mission': mission,
                'success': success,
                'num_steps': len(actions),
                'actions': actions,
                'initial_grid': initial_grid,
            })

            env.close()

        rate = wins / len(SEEDS) if SEEDS else 0
        status = f"{'#' * wins}{'.' * (len(SEEDS) - wins)}"
        print(f"  {name:35s} [{status}] {wins}/{len(SEEDS)}")

        levels_data.append({
            'id': level_id,
            'name': name,
            'difficulty': difficulty,
            'desc': desc,
            'wins': wins,
            'total': len(SEEDS),
            'rate': rate,
            'runs': runs,
        })

    data = {
        'levels': levels_data,
        'total_wins': total_wins,
        'total_tests': total_tests,
        'overall_rate': total_wins / total_tests if total_tests else 0,
    }

    return data


def embed_in_html(data):
    """Inject EMBEDDED_DATA into index.html so it works from file://."""
    html_path = 'docs/index.html'
    with open(html_path, 'r') as f:
        html = f.read()

    data_script = f'<script>const EMBEDDED_DATA = {json.dumps(data, cls=NumpyEncoder)};</script>'
    marker = '<!-- EMBEDDED_DATA -->'

    if marker in html:
        # Replace existing embedded data
        import re
        html = re.sub(
            r'<!-- EMBEDDED_DATA -->.*?<!-- /EMBEDDED_DATA -->',
            f'{marker}\n{data_script}\n<!-- /EMBEDDED_DATA -->',
            html,
            flags=re.DOTALL,
        )
        print("  Updated existing embedded data")
    else:
        # Insert before the main <script> tag
        html = html.replace(
            '<script>\nlet DATA = null;',
            f'{marker}\n{data_script}\n<!-- /EMBEDDED_DATA -->\n\n<script>\nlet DATA = null;',
        )
        print("  Injected embedded data block")

    with open(html_path, 'w') as f:
        f.write(html)
    print(f"  Embedded {len(json.dumps(data, cls=NumpyEncoder))} bytes into {html_path}")


def main():
    solver = None
    embed = False

    for arg in sys.argv[1:]:
        if arg == '--embed':
            embed = True
        elif not arg.startswith('-'):
            solver = arg

    if solver is None:
        print("Usage: python generate_viz_data.py <solver_module> [--embed]")
        print("  solver_module: Python module with solve_with_env() or solve()")
        sys.exit(1)

    print(f"Running solver: {solver}")
    print("=" * 72)

    data = run_suite(solver)

    print("=" * 72)
    print(f"OVERALL: {data['total_wins']}/{data['total_tests']} ({data['overall_rate']:.0%})")

    # Always write data.json
    with open('docs/data.json', 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)
    print(f"\nWrote docs/data.json ({len(json.dumps(data, cls=NumpyEncoder))} bytes)")

    if embed:
        embed_in_html(data)

    print("\nTo view: python -m http.server 8080 -d docs")
    print("  Then open http://localhost:8080")


if __name__ == '__main__':
    main()
