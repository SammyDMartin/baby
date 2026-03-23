"""
BabyAI solver: I understand the mission language, decompose into subgoals,
and use BFS pathfinding to navigate. No use of the built-in bot.
"""
import gymnasium as gym
import minigrid
from minigrid.core.constants import DIR_TO_VEC
from collections import deque
import re
import numpy as np

ACTION_MAP = {'left': 0, 'right': 1, 'forward': 2, 'pickup': 3, 'drop': 4, 'toggle': 5, 'done': 6}
DIR_NAMES = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
DIR_DELTAS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def get_grid_info(env):
    """Extract full grid state."""
    grid = env.unwrapped.grid
    agent_pos = tuple(env.unwrapped.agent_pos)
    agent_dir = env.unwrapped.agent_dir
    carrying = env.unwrapped.carrying
    
    objects = []
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get(x, y)
            if cell and cell.type not in ('wall',):
                obj_info = {
                    'pos': (x, y),
                    'type': cell.type,
                    'color': getattr(cell, 'color', None),
                }
                if cell.type == 'door':
                    obj_info['is_open'] = cell.is_open
                    obj_info['is_locked'] = cell.is_locked
                objects.append(obj_info)
    
    return {
        'grid': grid,
        'agent_pos': agent_pos,
        'agent_dir': agent_dir,
        'carrying': carrying,
        'objects': objects,
        'width': grid.width,
        'height': grid.height,
    }


def is_walkable(grid, x, y):
    """Check if a cell can be walked on."""
    if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
        return False
    cell = grid.get(x, y)
    if cell is None:
        return True
    if cell.type == 'door' and cell.is_open:
        return True
    return False


def bfs_actions(grid, start_pos, start_dir, target_pos, target_face_dir=None):
    """BFS to find action sequence from start to target position.
    Returns list of action names, or None if unreachable.
    If target_face_dir is set, we need to end facing that direction.
    """
    # State: (x, y, dir)
    queue = deque()
    queue.append((start_pos[0], start_pos[1], start_dir, []))
    visited = set()
    visited.add((start_pos[0], start_pos[1], start_dir))
    
    while queue:
        x, y, d, actions = queue.popleft()
        
        # Check if we reached target
        if (x, y) == target_pos:
            if target_face_dir is None or d == target_face_dir:
                return actions
            # If we need specific facing, try turning
        
        # Turn left
        new_d = (d - 1) % 4
        state = (x, y, new_d)
        if state not in visited:
            visited.add(state)
            queue.append((x, y, new_d, actions + ['left']))
        
        # Turn right
        new_d = (d + 1) % 4
        state = (x, y, new_d)
        if state not in visited:
            visited.add(state)
            queue.append((x, y, new_d, actions + ['right']))
        
        # Move forward
        dx, dy = DIR_DELTAS[d]
        nx, ny = x + dx, y + dy
        if is_walkable(grid, nx, ny):
            state = (nx, ny, d)
            if state not in visited:
                visited.add(state)
                queue.append((nx, ny, d, actions + ['forward']))
    
    return None


def bfs_to_adjacent(grid, start_pos, start_dir, target_pos):
    """Find actions to reach a cell adjacent to target, facing the target."""
    tx, ty = target_pos
    best = None
    
    for face_dir, (dx, dy) in DIR_DELTAS.items():
        # Stand at position opposite to face_dir from target
        stand_x = tx - dx
        stand_y = ty - dy
        
        if not (0 <= stand_x < grid.width and 0 <= stand_y < grid.height):
            continue
        
        # Check if we can stand there
        cell = grid.get(stand_x, stand_y)
        if cell is not None and not (cell.type == 'door' and cell.is_open):
            continue
        
        actions = bfs_actions(grid, start_pos, start_dir, (stand_x, stand_y), face_dir)
        if actions is not None:
            if best is None or len(actions) < len(best):
                best = actions
    
    return best


def find_objects(info, obj_type=None, color=None):
    """Find objects matching criteria."""
    results = []
    for obj in info['objects']:
        if obj_type and obj['type'] != obj_type:
            continue
        if color and obj['color'] != color:
            continue
        results.append(obj)
    return results


def parse_mission(mission):
    """Parse a BabyAI mission into structured instructions."""
    mission = mission.strip().lower()
    
    # Patterns
    goto_pattern = r'go to (?:the |a )?(?:(\w+) )?(\w+)'
    open_pattern = r'open (?:the |a )?(?:(\w+) )?door'
    pickup_pattern = r'pick up (?:the |a )?(?:(\w+) )?(\w+)'
    putnext_pattern = r'put (?:the |a )?(?:(\w+) )?(\w+) next to (?:the |a )?(?:(\w+) )?(\w+)'
    open_color_pattern = r'open (?:the |a )?(\w+) door'
    open_loc_pattern = r'open (?:the |a )?door on your (\w+)'
    
    # Put next to
    m = re.match(putnext_pattern, mission)
    if m:
        return {
            'type': 'putnext',
            'obj1_color': m.group(1),
            'obj1_type': m.group(2),
            'obj2_color': m.group(3),
            'obj2_type': m.group(4),
        }
    
    # Pickup
    m = re.match(pickup_pattern, mission)
    if m:
        return {
            'type': 'pickup',
            'color': m.group(1),
            'obj_type': m.group(2),
        }
    
    # Open door with color
    m = re.match(open_color_pattern, mission)
    if m:
        return {
            'type': 'open',
            'color': m.group(1),
        }
    
    # Open door with location
    m = re.match(open_loc_pattern, mission)
    if m:
        return {
            'type': 'open_loc',
            'direction': m.group(1),
        }
    
    # Open door generic
    m = re.match(open_pattern, mission)
    if m:
        return {
            'type': 'open',
            'color': m.group(1),
        }
    
    # Go to
    m = re.match(goto_pattern, mission)
    if m:
        return {
            'type': 'goto',
            'color': m.group(1),
            'obj_type': m.group(2),
        }
    
    return {'type': 'unknown', 'raw': mission}


def get_room_doors(grid, agent_pos):
    """Find doors in the room the agent is in by flood-filling."""
    visited = set()
    doors = []
    queue = deque([agent_pos])
    visited.add(agent_pos)
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x+dx, y+dy
            if (nx, ny) in visited:
                continue
            if nx < 0 or ny < 0 or nx >= grid.width or ny >= grid.height:
                continue
            cell = grid.get(nx, ny)
            if cell is None:
                visited.add((nx, ny))
                queue.append((nx, ny))
            elif cell.type == 'door':
                doors.append({'pos': (nx, ny), 'cell': cell,
                             'direction_from_agent': None})
                # Don't expand through closed doors
                if cell.is_open:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    
    return doors


def relative_direction(agent_pos, agent_dir, target_pos):
    """Get relative direction (left/right/front/behind) of target from agent."""
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
    
    # Compute angle from agent's facing direction
    # Agent facing: DIR_DELTAS[agent_dir]
    fdx, fdy = DIR_DELTAS[agent_dir]
    
    # Cross product to determine left/right
    cross = fdx * dy - fdy * dx
    dot = fdx * dx + fdy * dy
    
    if abs(cross) > abs(dot):
        if cross > 0:
            return 'right'
        else:
            return 'left'
    else:
        if dot > 0:
            return 'front'
        else:
            return 'behind'


def solve_task(env_name, seed=42, verbose=True):
    """Solve a BabyAI task."""
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    mission = obs['mission']
    
    if verbose:
        print(f"=== {env_name} (seed={seed}) ===")
        print(f"Mission: {mission}")
    
    parsed = parse_mission(mission)
    if verbose:
        print(f"Parsed: {parsed}")
    
    info = get_grid_info(env)
    all_actions = []
    max_iterations = 50  # Safety limit for multi-step tasks
    
    if parsed['type'] == 'goto':
        # Find the target object
        targets = find_objects(info, obj_type=parsed['obj_type'], color=parsed.get('color'))
        if not targets:
            print(f"  ERROR: No {parsed.get('color', '')} {parsed['obj_type']} found!")
            env.close()
            return False
        
        target = targets[0]
        actions = bfs_to_adjacent(info['grid'], info['agent_pos'], info['agent_dir'], target['pos'])
        if actions:
            all_actions = actions
        else:
            print(f"  ERROR: No path to {target['pos']}")
            env.close()
            return False
    
    elif parsed['type'] == 'open_loc':
        # Open a door on your left/right/front/behind
        doors = get_room_doors(info['grid'], info['agent_pos'])
        target_dir = parsed['direction']
        
        # Find door in the specified relative direction
        best_door = None
        for door in doors:
            rel = relative_direction(info['agent_pos'], info['agent_dir'], door['pos'])
            if rel == target_dir:
                best_door = door
                break
        
        if best_door is None:
            # Try all doors if exact match fails
            for door in doors:
                best_door = door
                break
        
        if best_door:
            actions = bfs_to_adjacent(info['grid'], info['agent_pos'], info['agent_dir'], best_door['pos'])
            if actions:
                all_actions = actions + ['toggle']
        else:
            print(f"  ERROR: No door found on {target_dir}")
            env.close()
            return False
    
    elif parsed['type'] == 'open':
        # Open a specific colored door - might need to find a key first
        color = parsed.get('color')
        doors = find_objects(info, obj_type='door', color=color)
        
        if not doors:
            # Try without color filter
            doors = find_objects(info, obj_type='door')
        
        target_door = doors[0] if doors else None
        
        if target_door and target_door.get('is_locked'):
            # Need to find a matching key first
            keys = find_objects(info, obj_type='key', color=color)
            
            # Find reachable key
            for key in keys:
                key_actions = bfs_to_adjacent(info['grid'], info['agent_pos'], info['agent_dir'], key['pos'])
                if key_actions is not None:
                    # Go pick up key, then navigate to door
                    all_actions = key_actions + ['pickup']
                    
                    # Simulate picking up key to update state
                    for a in all_actions:
                        obs, _, done, _, _ = env.step(ACTION_MAP[a])
                        if done:
                            break
                    
                    if not done:
                        # Now navigate to door with key
                        info2 = get_grid_info(env)
                        door_actions = bfs_to_adjacent(info2['grid'], info2['agent_pos'], info2['agent_dir'], target_door['pos'])
                        if door_actions:
                            # Execute remaining actions
                            remaining = door_actions + ['toggle']
                            for a in remaining:
                                obs, _, done, _, _ = env.step(ACTION_MAP[a])
                                if done:
                                    break
                            all_actions = all_actions + remaining
                    
                    # We already executed, check result
                    total_reward = 0
                    # Re-check by stepping done if needed
                    if verbose:
                        print(f"  Actions ({len(all_actions)}): {' '.join(all_actions[:20])}{'...' if len(all_actions)>20 else ''}")
                    env.close()
                    
                    # Need to re-run to get proper result
                    return solve_task_execute(env_name, seed, all_actions, verbose)
                    
            if not all_actions:
                print(f"  ERROR: No reachable {color} key found for locked door")
                env.close()
                return False
        
        elif target_door:
            # Door is just closed, toggle it
            actions = bfs_to_adjacent(info['grid'], info['agent_pos'], info['agent_dir'], target_door['pos'])
            if actions:
                all_actions = actions + ['toggle']
            else:
                # Door might be in another room - need to open intermediate doors
                # Do iterative approach: find nearest closed door, open it, repeat
                all_actions = solve_multi_room(env, info, target_door, verbose)
                if all_actions is None:
                    print(f"  ERROR: Cannot reach door")
                    env.close()
                    return False
                return solve_task_execute(env_name, seed, all_actions, verbose)
    
    elif parsed['type'] == 'pickup':
        targets = find_objects(info, obj_type=parsed['obj_type'], color=parsed.get('color'))
        if targets:
            target = targets[0]
            actions = bfs_to_adjacent(info['grid'], info['agent_pos'], info['agent_dir'], target['pos'])
            if actions:
                all_actions = actions + ['pickup']
    
    elif parsed['type'] == 'putnext':
        # Find object to pick up
        obj1s = find_objects(info, obj_type=parsed['obj1_type'], color=parsed.get('obj1_color'))
        obj2s = find_objects(info, obj_type=parsed['obj2_type'], color=parsed.get('obj2_color'))
        
        if obj1s and obj2s:
            obj1 = obj1s[0]
            obj2 = obj2s[0]
            
            # Step 1: Go to obj1 and pick it up
            actions1 = bfs_to_adjacent(info['grid'], info['agent_pos'], info['agent_dir'], obj1['pos'])
            if actions1:
                pickup_actions = actions1 + ['pickup']
                
                # Execute pickup
                for a in pickup_actions:
                    obs, _, done, _, _ = env.step(ACTION_MAP[a])
                
                # Step 2: Find empty cell adjacent to obj2 and go there
                info2 = get_grid_info(env)
                tx, ty = obj2['pos']
                
                best_drop = None
                for face_dir, (dx, dy) in DIR_DELTAS.items():
                    # We stand further back, facing toward drop cell
                    drop_x, drop_y = tx + dx, ty + dy  # Cell adjacent to target
                    stand_x, stand_y = drop_x + dx, drop_y + dy  # Cell we stand on
                    
                    # Check drop cell is empty
                    if not (0 <= drop_x < info2['width'] and 0 <= drop_y < info2['height']):
                        continue
                    drop_cell = info2['grid'].get(drop_x, drop_y)
                    if drop_cell is not None:
                        continue
                    
                    # Check stand cell
                    if not (0 <= stand_x < info2['width'] and 0 <= stand_y < info2['height']):
                        continue
                    stand_cell = info2['grid'].get(stand_x, stand_y)
                    if stand_cell is not None and not (stand_cell.type == 'door' and stand_cell.is_open):
                        continue
                    
                    # Invert face_dir to face toward target
                    toward_dir = (face_dir + 2) % 4
                    
                    acts = bfs_actions(info2['grid'], info2['agent_pos'], info2['agent_dir'], 
                                      (stand_x, stand_y), toward_dir)
                    if acts is not None:
                        if best_drop is None or len(acts) < len(best_drop):
                            best_drop = acts
                
                # Alternative: stand on cell adjacent to obj2, face empty adjacent cell
                # Actually simpler: stand adjacent to obj2, face away, so drop lands adjacent
                # No wait - I need to face an EMPTY cell that is adjacent to obj2.
                # When I drop, the object goes to the cell I'm facing.
                # For "put next to", the dropped object needs to be adjacent to obj2.
                # So I face an empty cell adjacent to obj2, and drop.
                
                # Let me redo this:
                for face_dir, (fdx, fdy) in DIR_DELTAS.items():
                    for stand_dir, (sdx, sdy) in DIR_DELTAS.items():
                        # Drop cell = adjacent to obj2
                        drop_x = tx + fdx * 0  # hmm this isn't right
                        pass
                
                # Simpler approach: for each empty cell adjacent to obj2,
                # find a cell I can stand on that faces that empty cell
                for adj_dir, (adx, ady) in DIR_DELTAS.items():
                    empty_x, empty_y = tx + adx, ty + ady
                    if not (0 <= empty_x < info2['width'] and 0 <= empty_y < info2['height']):
                        continue
                    ecell = info2['grid'].get(empty_x, empty_y)
                    if ecell is not None:
                        continue
                    
                    # Stand one cell back from empty cell, facing it
                    for face_dir, (fdx, fdy) in DIR_DELTAS.items():
                        sx = empty_x - fdx
                        sy = empty_y - fdy
                        if (sx, sy) == (empty_x, empty_y):
                            continue
                        if not (0 <= sx < info2['width'] and 0 <= sy < info2['height']):
                            continue
                        scell = info2['grid'].get(sx, sy)
                        if scell is not None and not (scell.type == 'door' and scell.is_open):
                            continue
                        
                        acts = bfs_actions(info2['grid'], info2['agent_pos'], info2['agent_dir'],
                                          (sx, sy), face_dir)
                        if acts is not None:
                            if best_drop is None or len(acts) < len(best_drop):
                                best_drop = acts
                
                if best_drop is not None:
                    all_actions = pickup_actions + best_drop + ['drop']
                    env.close()
                    return solve_task_execute(env_name, seed, all_actions, verbose)
    
    env.close()
    
    if all_actions:
        return solve_task_execute(env_name, seed, all_actions, verbose)
    else:
        if verbose:
            print("  ERROR: Could not plan actions")
        return False


def solve_multi_room(env, info, target_door, verbose=False):
    """Iteratively open doors to reach a target in a multi-room environment."""
    all_actions = []
    max_iter = 20
    
    for iteration in range(max_iter):
        info = get_grid_info(env)
        
        # Try direct path to target
        direct = bfs_to_adjacent(info['grid'], info['agent_pos'], info['agent_dir'], target_door['pos'])
        if direct is not None:
            all_actions.extend(direct + ['toggle'])
            return all_actions
        
        # Find nearest reachable closed door and open it
        room_doors = get_room_doors(info['grid'], info['agent_pos'])
        
        best_door_actions = None
        best_door = None
        for door in room_doors:
            if door['cell'].is_open:
                continue
            if door['cell'].is_locked:
                continue
            acts = bfs_to_adjacent(info['grid'], info['agent_pos'], info['agent_dir'], door['pos'])
            if acts is not None:
                # Prefer doors that get us closer to target
                if best_door_actions is None or len(acts) < len(best_door_actions):
                    best_door_actions = acts
                    best_door = door
        
        if best_door_actions is None:
            return None
        
        # Open this door
        open_actions = best_door_actions + ['toggle', 'forward']
        for a in open_actions:
            obs, _, done, _, _ = env.step(ACTION_MAP[a])
            if done:
                all_actions.extend(open_actions)
                return all_actions
        all_actions.extend(open_actions)
    
    return None


def solve_task_execute(env_name, seed, actions, verbose=True):
    """Execute a pre-planned action sequence and report results."""
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    
    total_reward = 0
    success = False
    
    for i, action_name in enumerate(actions):
        obs, reward, done, truncated, _ = env.step(ACTION_MAP[action_name])
        total_reward += reward
        if done:
            success = reward > 0
            if verbose:
                print(f"  Actions ({i+1}/{len(actions)}): {' '.join(actions[:30])}{'...' if len(actions)>30 else ''}")
                print(f"  Result: {'SUCCESS' if success else 'FAILED'} (reward={total_reward:.3f})")
            env.close()
            return success
    
    if verbose:
        print(f"  Actions ({len(actions)}): {' '.join(actions[:30])}{'...' if len(actions)>30 else ''}")
        print(f"  Result: {'SUCCESS' if success else 'FAILED - did not complete'} (reward={total_reward:.3f})")
    env.close()
    return success


# ============================================================
# Run the test suite
# ============================================================
if __name__ == "__main__":
    TASKS = [
        # Easy
        ("BabyAI-GoToRedBallNoDists-v0", "L1: GoTo (no distractors)"),
        ("BabyAI-GoToObj-v0", "L2: GoTo object"),
        ("BabyAI-GoToLocal-v0", "L3: GoTo local (distractors)"),
        ("BabyAI-GoToRedBlueBall-v0", "L2b: GoTo red/blue ball"),
        ("BabyAI-OpenDoor-v0", "L4: Open door"),
        ("BabyAI-OpenDoorColor-v0", "L4b: Open colored door"),
        ("BabyAI-PickupLoc-v0", "L5: Pickup object"),
        ("BabyAI-PickupDist-v0", "L5b: Pickup with distractors"),
        ("BabyAI-PutNextLocalS5N3-v0", "L7: Put next to (local)"),
        # Medium
        ("BabyAI-GoToObjMazeOpen-v0", "L6: GoTo through open maze"),
        ("BabyAI-Unlock-v0", "L8: Unlock door"),
        ("BabyAI-UnlockPickup-v0", "L9: Unlock + pickup"),
        # Hard
        ("BabyAI-GoToObjMaze-v0", "L10: GoTo through closed maze"),
        ("BabyAI-KeyCorridor-v0", "L11: Key corridor"),
        ("BabyAI-BossLevelNoUnlock-v0", "L12: Boss (no unlock)"),
        ("BabyAI-BossLevel-v0", "L13: Boss level"),
    ]
    
    results = []
    seeds_per_task = 3
    
    for env_name, label in TASKS:
        print(f"\n{'='*60}")
        print(f"TASK: {label} ({env_name})")
        print(f"{'='*60}")
        
        successes = 0
        for seed in range(42, 42 + seeds_per_task):
            try:
                ok = solve_task(env_name, seed=seed, verbose=True)
                if ok:
                    successes += 1
            except Exception as e:
                print(f"  Seed {seed}: ERROR - {e}")
        
        rate = successes / seeds_per_task
        results.append((label, env_name, rate, successes, seeds_per_task))
        print(f"  >>> {label}: {successes}/{seeds_per_task} = {rate*100:.0f}%")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, env_name, rate, s, t in results:
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  {label:40s} {bar} {s}/{t} ({rate*100:.0f}%)")
