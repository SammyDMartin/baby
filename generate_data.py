"""Generate JSON data for the web interface.
Runs the solver on all levels and exports replay data."""
import json
import sys
import numpy as np
from engine.solver import solve_level
from engine.grid import grid_to_dict
from engine.levels import ALL_LEVELS


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate():
    results = []
    total_wins = 0
    total_tests = 0

    for lvl in ALL_LEVELS:
        level_results = []
        for seed in range(42, 45):  # 3 seeds per level
            try:
                r = solve_level(lvl['id'], seed=seed)
                # Only keep initial_grid, final_grid, and actions
                # The web UI will replay from initial state
                level_results.append({
                    'seed': seed,
                    'mission': r['mission'],
                    'success': r['success'],
                    'num_steps': r['num_steps'],
                    'actions': r['actions'],
                    'initial_grid': r['initial_grid'],
                    'final_grid': r['final_grid'],
                })
                if r['success']:
                    total_wins += 1
                total_tests += 1
            except Exception as e:
                level_results.append({
                    'seed': seed,
                    'mission': 'ERROR',
                    'success': False,
                    'num_steps': 0,
                    'actions': [],
                    'error': str(e),
                })
                total_tests += 1

        wins = sum(1 for r in level_results if r.get('success'))
        results.append({
            'id': lvl['id'],
            'name': lvl['name'],
            'difficulty': lvl['difficulty'],
            'desc': lvl['desc'],
            'wins': wins,
            'total': len(level_results),
            'rate': wins / max(len(level_results), 1),
            'runs': level_results,
        })

        status = 'OK' if wins == len(level_results) else 'PARTIAL' if wins > 0 else 'FAIL'
        print(f'{status:7s} {lvl["name"]:35s} {wins}/{len(level_results)}', file=sys.stderr)

    data = {
        'levels': results,
        'total_wins': total_wins,
        'total_tests': total_tests,
        'overall_rate': total_wins / max(total_tests, 1),
    }

    print(f'\nOverall: {total_wins}/{total_tests} ({total_wins*100//max(total_tests,1)}%)', file=sys.stderr)
    return data


if __name__ == '__main__':
    data = generate()
    output_path = 'docs/data.json'
    with open(output_path, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)
    size = len(json.dumps(data, cls=NumpyEncoder))
    print(f'Wrote {output_path} ({size // 1024}KB)', file=sys.stderr)
