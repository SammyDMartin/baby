"""
Legacy nightmare level definitions.

These levels are now defined in engine/levels.py.
This file is kept for backwards compatibility only.
"""
from engine.levels import NIGHTMARE_LEVELS, _NIGHTMARE_CLASSES

# Re-export for any code that imports from here
NIGHTMARE_LEVELS_DICT = {cls_name: cls for cls_name, cls in _NIGHTMARE_CLASSES.items()}

if __name__ == "__main__":
    import gymnasium as gym
    from engine.grid import render_grid

    for env_id in _NIGHTMARE_CLASSES:
        print(f"\n{'='*70}")
        print(f"{env_id}")
        print(f"{'='*70}")
        env = gym.make(env_id)
        obs, _ = env.reset(seed=42)
        print(f"Mission: {obs['mission']}")
        print(render_grid(env))
        env.close()
