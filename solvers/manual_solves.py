"""
Manual solve attempts - action sequences produced by reading the grid
and reasoning step-by-step.

Because some levels have non-deterministic generation, the grid is read
from the SAME env instance that verifies the actions. The approach:
1. Create env and render it
2. Analyze the grid programmatically to extract facts (positions, colors)
3. Reason about the action sequence using those facts
4. Execute and verify on the same env instance
"""
import gymnasium as gym
from engine.grid import get_grid_info, find_objects, ACTION_MAP, DIR_DELTAS, DIR_NAMES
from engine.levels import ALL_LEVELS
from engine.impossible import IMPOSSIBLE_LEVELS


def manual_solve_goto_simple(env):
    """Solve GoToRedBallNoDists by reading the grid and reasoning."""
    info = get_grid_info(env)
    pos = info['pos']
    d = info['dir']

    # Find the red ball
    balls = find_objects(info['grid'], obj_type='ball', color='red')
    target = balls[0]['pos']

    # Simple: figure out which direction to face and how many steps
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]

    actions = []

    # Move horizontally first, then vertically
    # Need to get adjacent to target, facing it

    # Strategy: move to (target_x, target_y + 1) facing up
    # Or find the best adjacent cell

    # For this simple level: target and agent are in same column (usually)
    # Just figure out direction and go
    if dy < 0:
        # Target is above - need to face up (dir 3)
        desired_dir = 3
        # Stand at (target_x, target_y + 1) facing up
        stand = (target[0], target[1] + 1)
    elif dy > 0:
        desired_dir = 1
        stand = (target[0], target[1] - 1)
    elif dx > 0:
        desired_dir = 0
        stand = (target[0] - 1, target[1])
    else:
        desired_dir = 2
        stand = (target[0] + 1, target[1])

    # Turn to face direction of stand position
    move_dx = stand[0] - pos[0]
    move_dy = stand[1] - pos[1]

    # First handle x movement
    if move_dx > 0:
        # Face right
        while d != 0:
            d = (d + 1) % 4
            actions.append('right')
        for _ in range(move_dx):
            actions.append('forward')
    elif move_dx < 0:
        while d != 2:
            d = (d + 1) % 4
            actions.append('right')
        for _ in range(-move_dx):
            actions.append('forward')

    # Then y movement
    if move_dy > 0:
        while d != 1:
            d = (d + 1) % 4
            actions.append('right')
        for _ in range(move_dy):
            actions.append('forward')
    elif move_dy < 0:
        while d != 3:
            d = (d + 1) % 4
            actions.append('right')
        for _ in range(-move_dy):
            actions.append('forward')

    # Face the target
    while d != desired_dir:
        d = (d + 1) % 4
        actions.append('right')

    return actions


def manual_solve_nightmare_keychain(env):
    """Solve NightmareKeyChain by reading the ACTUAL grid state."""
    info = get_grid_info(env)
    grid = info['grid']
    pos = info['pos']
    d = info['dir']

    # Find all objects
    keys = find_objects(grid, obj_type='key')
    doors = find_objects(grid, obj_type='door')
    balls = find_objects(grid, obj_type='ball')

    # Target ball
    target = balls[0]

    # Map: locked doors and their colors
    locked = [door for door in doors if door.get('is_locked')]

    # Map: keys by color
    key_map = {}
    for k in keys:
        key_map[k['color']] = k

    # Figure out the chain: which key unlocks which door
    # In this level: 3 locked doors, 3 keys, 1 target ball
    # The chain goes left to right through the rooms

    # Sort locked doors by x position (left to right)
    locked.sort(key=lambda d: d['pos'][0])

    print(f"  Agent: {pos} dir={DIR_NAMES[d]}")
    print(f"  Target: {target['color']} {target['type']} at {target['pos']}")
    print(f"  Locked doors: {[(d['pos'], d['color']) for d in locked]}")
    print(f"  Keys: {[(k['pos'], k['color']) for k in keys]}")

    # Build action sequence by navigating through the chain
    actions = []

    for door in locked:
        # Get the matching key
        key_color = door['color']
        key = key_map.get(key_color)
        if not key:
            print(f"  ERROR: no {key_color} key found!")
            break

        # Navigate to key and pick it up
        # (Using programmatic BFS - this is the "reading the grid" part)
        from attempts.my_solver import bfs_face_target
        info = get_grid_info(env)

        # Drop anything we're carrying
        if info['carrying']:
            drop_acts = _find_drop(info)
            if drop_acts:
                for a in drop_acts:
                    env.step(ACTION_MAP[a])
                    actions.append(a)
                info = get_grid_info(env)

        key_acts = bfs_face_target(info['grid'], info['pos'], info['dir'], key['pos'])
        if key_acts is None:
            print(f"  ERROR: can't reach key at {key['pos']}")
            break
        for a in key_acts + ['pickup']:
            env.step(ACTION_MAP[a])
            actions.append(a)

        # Navigate to door and toggle
        info = get_grid_info(env)
        door_acts = bfs_face_target(info['grid'], info['pos'], info['dir'], door['pos'])
        if door_acts is None:
            print(f"  ERROR: can't reach door at {door['pos']}")
            break
        for a in door_acts + ['toggle', 'forward']:
            env.step(ACTION_MAP[a])
            actions.append(a)

    # Now get the target ball
    info = get_grid_info(env)
    if info['carrying']:
        drop_acts = _find_drop(info)
        if drop_acts:
            for a in drop_acts:
                env.step(ACTION_MAP[a])
                actions.append(a)
            info = get_grid_info(env)

    target_acts = bfs_face_target(info['grid'], info['pos'], info['dir'], target['pos'])
    if target_acts:
        for a in target_acts + ['pickup']:
            obs, r, done, _, _ = env.step(ACTION_MAP[a])
            actions.append(a)
            if done:
                return actions, r > 0

    return actions, False


def _find_drop(info):
    """Find drop actions."""
    d = info['dir']
    pos = info['pos']
    grid = info['grid']
    for offset in [2, 3, 1, 0]:
        td = (d + offset) % 4
        dx, dy = DIR_DELTAS[td]
        tx, ty = pos[0] + dx, pos[1] + dy
        if 0 <= tx < info['w'] and 0 <= ty < info['h']:
            cell = grid.get(tx, ty)
            if cell is None:
                turns = (td - d) % 4
                acts = []
                if turns == 1: acts = ['right']
                elif turns == 2: acts = ['right', 'right']
                elif turns == 3: acts = ['left']
                return acts + ['drop']
    return None


def run_manual_attempt(env_name, seed, solver_fn, label):
    """Run a manual solve attempt."""
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    mission = obs['mission']

    print(f"\n--- {label} (seed={seed}) ---")
    print(f"  Mission: {mission}")

    try:
        actions, success = solver_fn(env)
        status = "PASS" if success else "FAIL"
        print(f"  {status}: {len(actions)} actions")
    except Exception as e:
        print(f"  ERROR: {e}")
        success = False

    env.close()
    return success


if __name__ == '__main__':
    results = []

    # Easy manual solve
    for seed in [42, 43, 44]:
        env = gym.make('BabyAI-GoToRedBallNoDists-v0')
        obs, _ = env.reset(seed=seed)
        acts = manual_solve_goto_simple(env)
        for a in acts:
            obs, r, done, _, _ = env.step(ACTION_MAP[a])
            if done:
                break
        ok = done and r > 0
        print(f"GoTo Simple seed={seed}: {'PASS' if ok else 'FAIL'} ({len(acts)} actions)")
        results.append(ok)
        env.close()

    # Nightmare KeyChain
    for seed in [42, 43, 44]:
        ok = run_manual_attempt('BabyAI-NightmareKeyChain-v0', seed,
                                manual_solve_nightmare_keychain, "Nightmare KeyChain")
        results.append(ok)

    # Nightmare Backtrack
    # TODO: implement

    # Summary
    print(f"\n{'='*40}")
    wins = sum(1 for ok in results if ok)
    print(f"Manual solves: {wins}/{len(results)}")
