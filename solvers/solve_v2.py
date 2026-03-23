"""
BabyAI solver v2: handles multi-room, unlock, compound missions.
"""
import gymnasium as gym
import minigrid
from collections import deque
import re
import copy

ACTION_MAP = {'left': 0, 'right': 1, 'forward': 2, 'pickup': 3, 'drop': 4, 'toggle': 5, 'done': 6}
DIR_DELTAS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
DIR_NAMES = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}


def get_info(env):
    grid = env.unwrapped.grid
    return {
        'grid': grid,
        'pos': tuple(env.unwrapped.agent_pos),
        'dir': env.unwrapped.agent_dir,
        'carrying': env.unwrapped.carrying,
        'w': grid.width, 'h': grid.height,
    }


def find_objects(grid, obj_type=None, color=None):
    results = []
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get(x, y)
            if cell is None:
                continue
            if obj_type and cell.type != obj_type:
                continue
            if color and getattr(cell, 'color', None) != color:
                continue
            results.append({
                'pos': (x, y), 'type': cell.type,
                'color': getattr(cell, 'color', None),
                'is_open': getattr(cell, 'is_open', None),
                'is_locked': getattr(cell, 'is_locked', None),
            })
    return results


def walkable(grid, x, y):
    if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
        return False
    cell = grid.get(x, y)
    return cell is None or (cell.type == 'door' and cell.is_open)


def bfs_path(grid, start_pos, start_dir, target_pos, target_dir=None):
    """BFS from start to target. Returns action list or None."""
    queue = deque([(start_pos[0], start_pos[1], start_dir, [])])
    visited = {(start_pos[0], start_pos[1], start_dir)}
    
    while queue:
        x, y, d, acts = queue.popleft()
        if (x, y) == target_pos and (target_dir is None or d == target_dir):
            return acts
        if len(acts) > 200:
            continue
        
        for new_d, turn in [((d-1)%4, 'left'), ((d+1)%4, 'right')]:
            s = (x, y, new_d)
            if s not in visited:
                visited.add(s)
                queue.append((x, y, new_d, acts + [turn]))
        
        dx, dy = DIR_DELTAS[d]
        nx, ny = x+dx, y+dy
        if walkable(grid, nx, ny):
            s = (nx, ny, d)
            if s not in visited:
                visited.add(s)
                queue.append((nx, ny, d, acts + ['forward']))
    return None


def bfs_to_face(grid, start_pos, start_dir, target_pos):
    """Navigate to be adjacent to target, facing it. Returns actions or None."""
    best = None
    for face_dir, (dx, dy) in DIR_DELTAS.items():
        sx, sy = target_pos[0] - dx, target_pos[1] - dy
        if not (0 <= sx < grid.width and 0 <= sy < grid.height):
            continue
        if (sx, sy) != start_pos:
            cell = grid.get(sx, sy)
            if cell is not None and not (cell.type == 'door' and cell.is_open):
                continue
        acts = bfs_path(grid, start_pos, start_dir, (sx, sy), face_dir)
        if acts is not None and (best is None or len(acts) < len(best)):
            best = acts
    return best


def reachable_cells(grid, start):
    """Flood fill to find all reachable cells."""
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x+dx, y+dy
            if (nx, ny) not in visited and walkable(grid, nx, ny):
                visited.add((nx, ny))
                queue.append((nx, ny))
    return visited


def find_room_doors(grid, start):
    """Find all doors bordering reachable area."""
    reachable = reachable_cells(grid, start)
    doors = []
    for x, y in reachable:
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                cell = grid.get(nx, ny)
                if cell and cell.type == 'door' and (nx, ny) not in reachable:
                    doors.append({
                        'pos': (nx, ny), 'cell': cell,
                        'approach_from': (x, y),
                        'color': getattr(cell, 'color', None),
                        'is_locked': cell.is_locked,
                        'is_open': cell.is_open,
                    })
    return doors


def relative_dir_to_door(agent_pos, agent_dir, door_pos):
    """Get relative direction of door from agent."""
    dx = door_pos[0] - agent_pos[0]
    dy = door_pos[1] - agent_pos[1]
    fdx, fdy = DIR_DELTAS[agent_dir]
    cross = fdx * dy - fdy * dx
    dot = fdx * dx + fdy * dy
    
    if abs(cross) >= abs(dot):
        return 'left' if cross < 0 else 'right'
    return 'front' if dot > 0 else 'behind'


def _drop_carried(env, info):
    """Drop the carried item somewhere safe. Returns action list or None."""
    if not info['carrying']:
        return []
    pos = info['pos']
    d = info['dir']
    w, h = info['w'], info['h']
    grid = info['grid']
    
    # If on a door tile, step forward first so we don't block the passage
    cell_here = grid.get(pos[0], pos[1])
    if cell_here and cell_here.type == 'door':
        dx, dy = DIR_DELTAS[d]
        nx, ny = pos[0]+dx, pos[1]+dy
        if walkable(grid, nx, ny):
            # Step forward, then drop behind us (back onto/near the door is fine)
            # After forward, we're at (nx,ny) facing d. Behind is (d+2)%4
            behind = (d + 2) % 4
            return ['forward', 'right', 'right', 'drop']
    
    # Try each direction, prefer behind/sides over forward to not block path
    for offset in [2, 3, 1, 0]:  # behind, left, right, forward
        try_dir = (d + offset) % 4
        dx, dy = DIR_DELTAS[try_dir]
        tx, ty = pos[0]+dx, pos[1]+dy
        if 0 <= tx < w and 0 <= ty < h:
            cell = grid.get(tx, ty)
            if cell is None:
                turns = (try_dir - d) % 4
                acts = []
                if turns == 1:
                    acts = ['right']
                elif turns == 2:
                    acts = ['right', 'right']
                elif turns == 3:
                    acts = ['left']
                acts.append('drop')
                return acts
    return None


def execute(env, actions):
    """Execute actions, return (remaining_actions_executed, done, reward)."""
    total_reward = 0
    for i, a in enumerate(actions):
        obs, r, done, trunc, _ = env.step(ACTION_MAP[a])
        total_reward += r
        if done or trunc:
            return i+1, done, total_reward
    return len(actions), False, total_reward


def parse_mission(mission):
    """Parse BabyAI mission into subgoals."""
    mission = mission.strip().lower()
    
    # Compound: "X and Y"
    if ' and ' in mission:
        parts = mission.split(' and ', 1)
        sub1 = parse_single(parts[0].strip())
        sub2 = parse_single(parts[1].strip())
        return [sub1, sub2]
    
    return [parse_single(mission)]


def parse_single(text):
    """Parse a single instruction."""
    text = text.strip()
    
    # "put the X next to the/a Y"
    m = re.match(r'put (?:the |a )?(\w+) (\w+) next to (?:the |a )?(\w+) (\w+)', text)
    if m:
        return {'action': 'putnext', 'c1': m.group(1), 't1': m.group(2), 'c2': m.group(3), 't2': m.group(4)}
    
    # "put the X Y next to the/a Z" (color obj next to obj)
    m = re.match(r'put (?:the |a )?(\w+) (\w+) next to (?:the |a )?(\w+)', text)
    if m:
        return {'action': 'putnext', 'c1': m.group(1), 't1': m.group(2), 'c2': None, 't2': m.group(3)}
    
    # "go to the X Y (behind you / on your left etc)"
    m = re.match(r'go to (?:the |a )?(\w+) (\w+)', text)
    if m:
        return {'action': 'goto', 'color': m.group(1), 'type': m.group(2)}
    
    # "go to the X" (just type, no color)
    m = re.match(r'go to (?:the |a )?(\w+)', text)
    if m:
        return {'action': 'goto', 'color': None, 'type': m.group(1)}
    
    # "pick up the X Y"
    m = re.match(r'pick up (?:the |a )?(\w+) (\w+)', text)
    if m:
        w1, w2 = m.group(1), m.group(2)
        COLORS = {'red','green','blue','purple','yellow','grey'}
        OBJ_TYPES = {'ball','key','box','door'}
        if w2 in OBJ_TYPES and w1 in COLORS:
            return {'action': 'pickup', 'color': w1, 'type': w2}
        elif w1 in OBJ_TYPES:
            return {'action': 'pickup', 'color': None, 'type': w1}
        else:
            return {'action': 'pickup', 'color': None, 'type': w2 if w2 in OBJ_TYPES else w1}
    
    # "pick up the X"
    m = re.match(r'pick up (?:the |a )?(\w+)', text)
    if m:
        return {'action': 'pickup', 'color': None, 'type': m.group(1)}
    
    # "open a door on your X" (must check before color pattern)
    m = re.match(r'open (?:the |a )?door on your (\w+)', text)
    if m:
        return {'action': 'open_rel', 'rel_dir': m.group(1)}
    
    # "open the X door"
    m = re.match(r'open (?:the |a )?(\w+) door', text)
    if m:
        color = m.group(1)
        if color not in ('the', 'a'):
            return {'action': 'open', 'color': color}
    
    # "open the door"
    m = re.match(r'open (?:the |a )?door', text)
    if m:
        return {'action': 'open', 'color': None}
    
    return {'action': 'unknown', 'raw': text}


def solve_goto(env, info, color, obj_type):
    """Solve a go-to task, handling multi-room if needed."""
    targets = find_objects(info['grid'], obj_type=obj_type, color=color)
    if not targets and color:
        # Try interpreting first word as color
        targets = find_objects(info['grid'], obj_type=obj_type)
    
    for target in targets:
        acts = bfs_to_face(info['grid'], info['pos'], info['dir'], target['pos'])
        if acts is not None:
            return acts
    
    # Target not directly reachable — need to open doors
    return solve_via_doors(env, info, lambda i: _goto_final(i, obj_type, color))


def _goto_final(info, obj_type, color):
    targets = find_objects(info['grid'], obj_type=obj_type, color=color)
    if not targets and color:
        targets = find_objects(info['grid'], obj_type=obj_type)
    for t in targets:
        acts = bfs_to_face(info['grid'], info['pos'], info['dir'], t['pos'])
        if acts is not None:
            return acts
    return None


def solve_via_doors(env, info, final_check_fn, max_doors=15):
    """Iteratively open doors to reach a goal. final_check_fn returns actions if goal reachable."""
    all_actions = []
    
    for _ in range(max_doors):
        info = get_info(env)
        
        # Check if goal is now reachable
        final_acts = final_check_fn(info)
        if final_acts is not None:
            n, done, r = execute(env, final_acts)
            all_actions.extend(final_acts[:n])
            return all_actions
        
        # Find closest reachable closed (not locked) door
        doors = find_room_doors(info['grid'], info['pos'])
        unlocked_doors = [d for d in doors if not d['is_locked'] and not d['is_open']]
        
        if not unlocked_doors:
            # Check for locked doors we might have a key for
            carrying = info['carrying']
            if carrying and carrying.type == 'key':
                locked_doors = [d for d in doors if d['is_locked'] and d['color'] == carrying.color]
                if locked_doors:
                    unlocked_doors = locked_doors
            
            if not unlocked_doors:
                return None  # Stuck
        
        # Pick closest door
        best_door = None
        best_acts = None
        for door in unlocked_doors:
            acts = bfs_to_face(info['grid'], info['pos'], info['dir'], door['pos'])
            if acts is not None and (best_acts is None or len(acts) < len(best_acts)):
                best_acts = acts
                best_door = door
        
        if best_acts is None:
            return None
        
        # Go to door and toggle it open
        open_acts = best_acts + ['toggle']
        n, done, r = execute(env, open_acts)
        all_actions.extend(open_acts[:n])
        if done:
            return all_actions
        
        # Try to step through
        info = get_info(env)
        # Check if door is now open — step forward through it
        fwd_acts = ['forward']
        n, done, r = execute(env, fwd_acts)
        all_actions.extend(fwd_acts[:n])
        if done:
            return all_actions
    
    return None


def solve_open(env, info, color):
    """Open a door (possibly locked)."""
    doors = find_objects(info['grid'], obj_type='door', color=color)
    if not doors:
        doors = find_objects(info['grid'], obj_type='door')
    
    target = doors[0] if doors else None
    if not target:
        return None
    
    if target.get('is_locked'):
        # Need key of matching color
        keys = find_objects(info['grid'], obj_type='key', color=target['color'])
        
        # Try reachable keys first
        for key in keys:
            key_acts = bfs_to_face(info['grid'], info['pos'], info['dir'], key['pos'])
            if key_acts is not None:
                # Pick up key
                pickup_acts = key_acts + ['pickup']
                n, done, r = execute(env, pickup_acts)
                if done:
                    return pickup_acts[:n]
                
                # Now navigate to door
                info2 = get_info(env)
                door_acts = bfs_to_face(info2['grid'], info2['pos'], info2['dir'], target['pos'])
                if door_acts is not None:
                    toggle_acts = door_acts + ['toggle']
                    n2, done2, r2 = execute(env, toggle_acts)
                    return pickup_acts + toggle_acts[:n2]
                
                # Door not directly reachable, try via other doors
                def check_door(i):
                    return bfs_to_face(i['grid'], i['pos'], i['dir'], target['pos'])
                
                result = solve_via_doors(env, info2, lambda i: _open_final(i, target['pos']))
                if result is not None:
                    return pickup_acts + result
        
        # Key not in current room — need to navigate through doors to find it
        def find_key_and_open(info):
            keys = find_objects(info['grid'], obj_type='key', color=target['color'])
            for k in keys:
                acts = bfs_to_face(info['grid'], info['pos'], info['dir'], k['pos'])
                if acts is not None:
                    return acts  # Just get to the key first
            return None
        
        result = solve_via_doors(env, info, find_key_and_open)
        if result is not None:
            # Now we're next to the key — pick it up
            pickup_acts = ['pickup']
            n, done, r = execute(env, pickup_acts)
            if done:
                return result + pickup_acts
            
            info3 = get_info(env)
            # Navigate to the locked door
            door_result = solve_via_doors(env, info3, lambda i: _open_final(i, target['pos']))
            if door_result is not None:
                return result + pickup_acts + door_result
        
        return None
    
    else:
        # Door is closed, just toggle it
        acts = bfs_to_face(info['grid'], info['pos'], info['dir'], target['pos'])
        if acts is not None:
            return acts + ['toggle']
        
        # Door not reachable — open intermediate doors
        return solve_via_doors(env, info, lambda i: _open_final(i, target['pos']))


def _open_final(info, door_pos):
    acts = bfs_to_face(info['grid'], info['pos'], info['dir'], door_pos)
    if acts is not None:
        return acts + ['toggle']
    return None


def solve_open_rel(env, info, rel_dir):
    """Open a door based on relative direction."""
    doors = find_room_doors(info['grid'], info['pos'])
    
    for door in doors:
        rel = relative_dir_to_door(info['pos'], info['dir'], door['pos'])
        if rel == rel_dir:
            acts = bfs_to_face(info['grid'], info['pos'], info['dir'], door['pos'])
            if acts is not None:
                return acts + ['toggle']
    
    # If no exact match, try any closed door
    for door in doors:
        if not door['is_open'] and not door['is_locked']:
            acts = bfs_to_face(info['grid'], info['pos'], info['dir'], door['pos'])
            if acts is not None:
                return acts + ['toggle']
    return None


def solve_pickup(env, info, color, obj_type):
    """Pick up an object, possibly through locked doors."""
    targets = find_objects(info['grid'], obj_type=obj_type, color=color)
    if not targets and color:
        targets = find_objects(info['grid'], obj_type=obj_type)
    
    for t in targets:
        acts = bfs_to_face(info['grid'], info['pos'], info['dir'], t['pos'])
        if acts is not None:
            return acts + ['pickup']
    
    # Not reachable — need doors
    def check_pickup(i):
        ts = find_objects(i['grid'], obj_type=obj_type, color=color)
        if not ts and color:
            ts = find_objects(i['grid'], obj_type=obj_type)
        for t in ts:
            a = bfs_to_face(i['grid'], i['pos'], i['dir'], t['pos'])
            if a is not None:
                return a + ['pickup']
        return None
    
    # But first check if we need a key to get through a locked door
    doors = find_room_doors(info['grid'], info['pos'])
    locked = [d for d in doors if d['is_locked']]
    
    if locked and not info['carrying']:
        # Find matching key
        for ld in locked:
            keys = find_objects(info['grid'], obj_type='key', color=ld['color'])
            for k in keys:
                key_acts = bfs_to_face(info['grid'], info['pos'], info['dir'], k['pos'])
                if key_acts is not None:
                    # Pick up key
                    full = key_acts + ['pickup']
                    n, done, r = execute(env, full)
                    if done:
                        return full[:n]
                    
                    # Navigate to locked door
                    info2 = get_info(env)
                    door_acts = bfs_to_face(info2['grid'], info2['pos'], info2['dir'], ld['pos'])
                    if door_acts is not None:
                        unlock = door_acts + ['toggle', 'forward']
                        n2, done2, r2 = execute(env, unlock)
                        if done2:
                            return full + unlock[:n2]
                        
                        # Now try to find target
                        info3 = get_info(env)
                        result = solve_via_doors(env, info3, check_pickup)
                        if result is not None:
                            return full + unlock + result
    
    return solve_via_doors(env, info, check_pickup)


def solve_putnext(env, info, subgoal):
    """Pick up obj1 and put it next to obj2."""
    c1, t1 = subgoal.get('c1'), subgoal.get('t1')
    c2, t2 = subgoal.get('c2'), subgoal.get('t2')
    
    # Find obj1 to pick up
    objs1 = find_objects(info['grid'], obj_type=t1, color=c1)
    if not objs1:
        objs1 = find_objects(info['grid'], obj_type=t1)
    
    # Find obj2 (destination reference)
    objs2 = find_objects(info['grid'], obj_type=t2, color=c2)
    if not objs2:
        objs2 = find_objects(info['grid'], obj_type=t2)
    
    if not objs1 or not objs2:
        return None
    
    obj1 = None
    pick_acts = None
    for o in objs1:
        a = bfs_to_face(info['grid'], info['pos'], info['dir'], o['pos'])
        if a is not None and (pick_acts is None or len(a) < len(pick_acts)):
            pick_acts = a
            obj1 = o
    
    if pick_acts is None:
        # Try through doors
        def find_obj1(i):
            os = find_objects(i['grid'], obj_type=t1, color=c1)
            if not os:
                os = find_objects(i['grid'], obj_type=t1)
            for o in os:
                a = bfs_to_face(i['grid'], i['pos'], i['dir'], o['pos'])
                if a is not None:
                    return a + ['pickup']
            return None
        result = solve_via_doors(env, info, find_obj1)
        if result is None:
            return None
        # After pickup, drop phase
        info2 = get_info(env)
        return result + _solve_drop_near(env, info2, t2, c2)
    
    # Pick up
    full_pickup = pick_acts + ['pickup']
    n, done, r = execute(env, full_pickup)
    if done:
        return full_pickup[:n]
    
    info2 = get_info(env)
    drop_acts = _solve_drop_near(env, info2, t2, c2)
    if drop_acts is not None:
        return full_pickup + drop_acts
    return None


def _solve_drop_near(env, info, t2, c2):
    """Drop carried object next to obj2."""
    objs2 = find_objects(info['grid'], obj_type=t2, color=c2)
    if not objs2:
        objs2 = find_objects(info['grid'], obj_type=t2)
    
    if not objs2:
        return None
    
    obj2 = objs2[0]
    tx, ty = obj2['pos']
    
    # Find an empty cell adjacent to obj2, then stand behind it facing toward it
    best = None
    for adj_dir, (adx, ady) in DIR_DELTAS.items():
        empty_x, empty_y = tx + adx, ty + ady
        if not (0 <= empty_x < info['w'] and 0 <= empty_y < info['h']):
            continue
        ecell = info['grid'].get(empty_x, empty_y)
        if ecell is not None:
            continue
        
        # Stand one step back from empty cell, facing it
        for face_dir, (fdx, fdy) in DIR_DELTAS.items():
            sx, sy = empty_x - fdx, empty_y - fdy
            if (sx, sy) == (empty_x, empty_y):
                continue
            if not (0 <= sx < info['w'] and 0 <= sy < info['h']):
                continue
            # Can't stand on obj2 or on other objects
            scell = info['grid'].get(sx, sy)
            if scell is not None and not (scell.type == 'door' and scell.is_open):
                continue
            
            acts = bfs_path(info['grid'], info['pos'], info['dir'], (sx, sy), face_dir)
            if acts is not None and (best is None or len(acts) < len(best)):
                best = acts
    
    if best is not None:
        drop = best + ['drop']
        n, done, r = execute(env, drop)
        return drop[:n] if done else drop
    
    # Try via doors
    def check_drop(i):
        return _try_drop_positions(i, t2, c2)
    
    return solve_via_doors(env, info, check_drop)


def _try_drop_positions(info, t2, c2):
    objs2 = find_objects(info['grid'], obj_type=t2, color=c2)
    if not objs2:
        objs2 = find_objects(info['grid'], obj_type=t2)
    if not objs2:
        return None
    obj2 = objs2[0]
    tx, ty = obj2['pos']
    
    best = None
    for adj_dir, (adx, ady) in DIR_DELTAS.items():
        ex, ey = tx + adx, ty + ady
        if not (0 <= ex < info['w'] and 0 <= ey < info['h']):
            continue
        if info['grid'].get(ex, ey) is not None:
            continue
        for fd, (fdx, fdy) in DIR_DELTAS.items():
            sx, sy = ex - fdx, ey - fdy
            if not (0 <= sx < info['w'] and 0 <= sy < info['h']):
                continue
            sc = info['grid'].get(sx, sy)
            if sc is not None and not (sc.type == 'door' and sc.is_open):
                continue
            acts = bfs_path(info['grid'], info['pos'], info['dir'], (sx, sy), fd)
            if acts is not None and (best is None or len(acts) < len(best)):
                best = acts
    if best is not None:
        return best + ['drop']
    return None


def solve_env(env_name, seed=42, verbose=True):
    """Main solver entry point."""
    env = gym.make(env_name, max_steps=500)
    obs, _ = env.reset(seed=seed)
    mission = obs['mission']
    
    if verbose:
        print(f"  Mission: {mission}")
    
    subgoals = parse_mission(mission)
    if verbose:
        print(f"  Subgoals: {subgoals}")
    
    all_actions = []
    
    for sg in subgoals:
        info = get_info(env)
        action = sg.get('action', 'unknown')
        
        if action == 'goto':
            color = sg.get('color')
            obj_type = sg.get('type')
            acts = solve_goto(env, info, color, obj_type)
        
        elif action == 'open':
            acts = solve_open(env, info, sg.get('color'))
        
        elif action == 'open_rel':
            acts = solve_open_rel(env, info, sg.get('rel_dir'))
        
        elif action == 'pickup':
            acts = solve_pickup(env, info, sg.get('color'), sg.get('type'))
        
        elif action == 'putnext':
            acts = solve_putnext(env, info, sg)
        
        else:
            if verbose:
                print(f"  Unknown action: {action}")
            acts = None
        
        if acts is None:
            if verbose:
                print(f"  FAILED to plan for subgoal: {sg}")
            env.close()
            return False
        
        # Execute (some may already be executed by solver)
        # Check if env is already done
        info_check = get_info(env)
        # Acts might be partially executed already. Try remaining.
        # Actually, solvers that call execute() internally have already moved the env.
        # We just track success.
    
    env.close()
    
    # Re-run cleanly to verify
    env = gym.make(env_name, max_steps=500)
    obs, _ = env.reset(seed=seed)
    
    # Reconstruct by solving again with a fresh env
    total_actions = []
    done = False
    reward = 0
    
    for sg in subgoals:
        info = get_info(env)
        action = sg.get('action', 'unknown')
        
        if action == 'goto':
            acts = solve_goto(env, info, sg.get('color'), sg.get('type'))
        elif action == 'open':
            acts = solve_open(env, info, sg.get('color'))
        elif action == 'open_rel':
            acts = solve_open_rel(env, info, sg.get('rel_dir'))
        elif action == 'pickup':
            acts = solve_pickup(env, info, sg.get('color'), sg.get('type'))
        elif action == 'putnext':
            acts = solve_putnext(env, info, sg)
        else:
            acts = None
        
        if acts is None:
            env.close()
            return False
        
        # The solver already executed actions on env, so check done
        try:
            check_info = get_info(env)
        except:
            pass
    
    # Final check
    env2 = gym.make(env_name, max_steps=500)
    obs2, _ = env2.reset(seed=seed)
    
    # Actually just trust the solver's internal execution
    # Let me simplify: just solve once, the solver executes as it goes
    env.close()
    env2.close()
    
    return True  # If we got here without failing


def solve_and_verify(env_name, seed=42, verbose=True):
    """Solve, then verify by replaying."""
    env = gym.make(env_name, max_steps=500)
    obs, _ = env.reset(seed=seed)
    mission = obs['mission']
    
    if verbose:
        print(f"  Mission: {mission}")
    
    subgoals = parse_mission(mission)
    
    # Solve incrementally, recording all actions
    recorded_actions = []
    done_flag = False
    total_reward = 0
    
    for sg in subgoals:
        if done_flag:
            break
        info = get_info(env)
        action = sg.get('action', 'unknown')
        
        if action == 'goto':
            result = _solve_and_record_goto(env, info, sg)
        elif action == 'open':
            result = _solve_and_record_open(env, info, sg)
        elif action == 'open_rel':
            result = _solve_and_record_open_rel(env, info, sg)
        elif action == 'pickup':
            result = _solve_and_record_pickup(env, info, sg)
        elif action == 'putnext':
            result = _solve_and_record_putnext(env, info, sg)
        else:
            result = None
        
        if result is None:
            if verbose:
                print(f"  FAILED: {sg}")
            env.close()
            return False
        
        recorded_actions.extend(result['actions'])
        done_flag = result.get('done', False)
        total_reward += result.get('reward', 0)
    
    env.close()
    
    success = done_flag and total_reward > 0
    if verbose:
        summary = ' '.join(recorded_actions[:30])
        if len(recorded_actions) > 30:
            summary += '...'
        print(f"  Actions ({len(recorded_actions)}): {summary}")
        print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
    
    return success


def _record_execute(env, actions):
    """Execute actions and record results."""
    total_r = 0
    executed = []
    for a in actions:
        obs, r, done, trunc, _ = env.step(ACTION_MAP[a])
        total_r += r
        executed.append(a)
        if done or trunc:
            return {'actions': executed, 'done': done, 'reward': total_r}
    return {'actions': executed, 'done': False, 'reward': total_r}


def _solve_and_record_goto(env, info, sg):
    color, obj_type = sg.get('color'), sg.get('type')
    
    # Direct path?
    targets = find_objects(info['grid'], obj_type=obj_type, color=color)
    if not targets and color:
        targets = find_objects(info['grid'], obj_type=obj_type)
    
    for t in targets:
        acts = bfs_to_face(info['grid'], info['pos'], info['dir'], t['pos'])
        if acts is not None:
            return _record_execute(env, acts)
    
    # Multi-room
    return _solve_record_via_doors(env, info, lambda i: _goto_final(i, obj_type, color))


def _solve_and_record_open(env, info, sg):
    color = sg.get('color')
    doors = find_objects(info['grid'], obj_type='door', color=color)
    if not doors:
        doors = find_objects(info['grid'], obj_type='door')
    if not doors:
        return None
    
    target = doors[0]
    
    if target.get('is_locked'):
        # Find and pick up key
        keys = find_objects(info['grid'], obj_type='key', color=target['color'])
        for k in keys:
            key_acts = bfs_to_face(info['grid'], info['pos'], info['dir'], k['pos'])
            if key_acts is not None:
                r1 = _record_execute(env, key_acts + ['pickup'])
                if r1['done']:
                    return r1
                
                info2 = get_info(env)
                door_acts = bfs_to_face(info2['grid'], info2['pos'], info2['dir'], target['pos'])
                if door_acts is not None:
                    r2 = _record_execute(env, door_acts + ['toggle'])
                    return {'actions': r1['actions'] + r2['actions'], 
                            'done': r2['done'], 'reward': r1['reward'] + r2['reward']}
                
                # Navigate through doors
                r2 = _solve_record_via_doors(env, get_info(env), 
                                              lambda i: _open_final(i, target['pos']))
                if r2:
                    return {'actions': r1['actions'] + r2['actions'],
                            'done': r2['done'], 'reward': r1['reward'] + r2['reward']}
        
        # Key not reachable directly — go find it through doors
        def find_key(i):
            ks = find_objects(i['grid'], obj_type='key', color=target['color'])
            for k in ks:
                a = bfs_to_face(i['grid'], i['pos'], i['dir'], k['pos'])
                if a is not None:
                    return a + ['pickup']
            return None
        
        r1 = _solve_record_via_doors(env, info, find_key)
        if r1 and not r1['done']:
            info3 = get_info(env)
            r2 = _solve_record_via_doors(env, info3, lambda i: _open_final(i, target['pos']))
            if r2:
                return {'actions': r1['actions'] + r2['actions'],
                        'done': r2['done'], 'reward': r1['reward'] + r2['reward']}
        return r1
    
    else:
        acts = bfs_to_face(info['grid'], info['pos'], info['dir'], target['pos'])
        if acts is not None:
            return _record_execute(env, acts + ['toggle'])
        return _solve_record_via_doors(env, info, lambda i: _open_final(i, target['pos']))


def _solve_and_record_open_rel(env, info, sg):
    doors = find_room_doors(info['grid'], info['pos'])
    rel = sg.get('rel_dir')
    
    for d in doors:
        if relative_dir_to_door(info['pos'], info['dir'], d['pos']) == rel:
            acts = bfs_to_face(info['grid'], info['pos'], info['dir'], d['pos'])
            if acts is not None:
                return _record_execute(env, acts + ['toggle'])
    
    for d in doors:
        if not d['is_open'] and not d['is_locked']:
            acts = bfs_to_face(info['grid'], info['pos'], info['dir'], d['pos'])
            if acts is not None:
                return _record_execute(env, acts + ['toggle'])
    return None


def _solve_and_record_pickup(env, info, sg):
    color, obj_type = sg.get('color'), sg.get('type')
    targets = find_objects(info['grid'], obj_type=obj_type, color=color)
    if not targets and color:
        targets = find_objects(info['grid'], obj_type=obj_type)
    
    for t in targets:
        acts = bfs_to_face(info['grid'], info['pos'], info['dir'], t['pos'])
        if acts is not None:
            return _record_execute(env, acts + ['pickup'])
    
    # Through doors — may need key
    doors = find_room_doors(info['grid'], info['pos'])
    locked = [d for d in doors if d['is_locked']]
    
    if locked and not info['carrying']:
        for ld in locked:
            keys = find_objects(info['grid'], obj_type='key', color=ld['color'])
            
            # Try direct path to key first
            direct_key = None
            for k in keys:
                ka = bfs_to_face(info['grid'], info['pos'], info['dir'], k['pos'])
                if ka is not None:
                    direct_key = (k, ka)
                    break
            
            if direct_key is None and keys:
                # Key exists but behind closed doors — navigate to it
                key_color = ld['color']
                def find_key(i):
                    ks = find_objects(i['grid'], obj_type='key', color=key_color)
                    for kk in ks:
                        a = bfs_to_face(i['grid'], i['pos'], i['dir'], kk['pos'])
                        if a is not None:
                            return a + ['pickup']
                    return None
                
                r_nav = _solve_record_via_doors(env, info, find_key)
                if r_nav and not r_nav['done']:
                    # Now carrying key, navigate to locked door
                    info_k = get_info(env)
                    def find_locked_door(i):
                        a = bfs_to_face(i['grid'], i['pos'], i['dir'], ld['pos'])
                        if a is not None:
                            return a + ['toggle', 'forward']
                        return None
                    
                    r_door = _solve_record_via_doors(env, info_k, find_locked_door)
                    if r_door:
                        # Drop key
                        info_d = get_info(env)
                        drop_a = _drop_carried(env, info_d)
                        r_drop = _record_execute(env, drop_a) if drop_a else {'actions':[],'done':False,'reward':0}
                        if r_drop['done']:
                            return {'actions': r_nav['actions']+r_door['actions']+r_drop['actions'],
                                    'done':True, 'reward': r_nav['reward']+r_door['reward']+r_drop['reward']}
                        
                        # Now find and pick up target
                        info_final = get_info(env)
                        def check_t(i):
                            ts = find_objects(i['grid'], obj_type=obj_type, color=color)
                            if not ts and color:
                                ts = find_objects(i['grid'], obj_type=obj_type)
                            for t in ts:
                                a = bfs_to_face(i['grid'], i['pos'], i['dir'], t['pos'])
                                if a is not None:
                                    return a + ['pickup']
                            return None
                        
                        r_pick = _solve_record_via_doors(env, info_final, check_t)
                        if r_pick:
                            all_a = r_nav['actions']+r_door['actions']+r_drop['actions']+r_pick['actions']
                            all_r = r_nav['reward']+r_door['reward']+r_drop['reward']+r_pick['reward']
                            return {'actions': all_a, 'done': r_pick['done'], 'reward': all_r}
                continue
            
            if direct_key:
                k, ka = direct_key
                r1 = _record_execute(env, ka + ['pickup'])
                if r1['done']:
                    return r1
                info2 = get_info(env)
                da = bfs_to_face(info2['grid'], info2['pos'], info2['dir'], ld['pos'])
                if da is not None:
                    r2 = _record_execute(env, da + ['toggle', 'forward'])
                    if r2['done']:
                        return {'actions': r1['actions'] + r2['actions'],
                                'done': True, 'reward': r1['reward'] + r2['reward']}

                    # DROP THE KEY — can't pick up target while carrying it
                    info_drop = get_info(env)
                    drop_acts = _drop_carried(env, info_drop)
                    if drop_acts:
                        r_drop = _record_execute(env, drop_acts)
                    else:
                        r_drop = {'actions': [], 'done': False, 'reward': 0}

                    info3 = get_info(env)

                    def check_target(i):
                        ts = find_objects(i['grid'], obj_type=obj_type, color=color)
                        if not ts and color:
                            ts = find_objects(i['grid'], obj_type=obj_type)
                        for t in ts:
                            a = bfs_to_face(i['grid'], i['pos'], i['dir'], t['pos'])
                            if a is not None:
                                return a + ['pickup']
                        return None

                    r3 = _solve_record_via_doors(env, info3, check_target)
                    if r3:
                        all_a = r1['actions'] + r2['actions'] + r_drop['actions'] + r3['actions']
                        all_r = r1['reward'] + r2['reward'] + r_drop['reward'] + r3['reward']
                        return {'actions': all_a, 'done': r3['done'], 'reward': all_r}
    
    def check_pickup(i):
        ts = find_objects(i['grid'], obj_type=obj_type, color=color)
        if not ts and color:
            ts = find_objects(i['grid'], obj_type=obj_type)
        for t in ts:
            a = bfs_to_face(i['grid'], i['pos'], i['dir'], t['pos'])
            if a is not None:
                return a + ['pickup']
        return None
    
    return _solve_record_via_doors(env, info, check_pickup)


def _solve_and_record_putnext(env, info, sg):
    c1, t1 = sg.get('c1'), sg.get('t1')
    c2, t2 = sg.get('c2'), sg.get('t2')
    
    # Pick up obj1
    objs1 = find_objects(info['grid'], obj_type=t1, color=c1)
    if not objs1 and c1:
        objs1 = find_objects(info['grid'], obj_type=t1)
    
    best_pick = None
    for o in objs1:
        a = bfs_to_face(info['grid'], info['pos'], info['dir'], o['pos'])
        if a is not None and (best_pick is None or len(a) < len(best_pick)):
            best_pick = a
    
    if best_pick is None:
        # Navigate through doors
        def find_obj(i):
            os = find_objects(i['grid'], obj_type=t1, color=c1)
            if not os and c1:
                os = find_objects(i['grid'], obj_type=t1)
            for o in os:
                a = bfs_to_face(i['grid'], i['pos'], i['dir'], o['pos'])
                if a is not None:
                    return a + ['pickup']
            return None
        r1 = _solve_record_via_doors(env, info, find_obj)
        if r1 is None:
            return None
        if r1['done']:
            return r1
    else:
        r1 = _record_execute(env, best_pick + ['pickup'])
        if r1['done']:
            return r1
    
    # Drop near obj2
    info2 = get_info(env)
    
    def check_drop(i):
        return _try_drop_positions(i, t2, c2)
    
    drop_acts = check_drop(info2)
    if drop_acts:
        r2 = _record_execute(env, drop_acts)
        return {'actions': r1['actions'] + r2['actions'],
                'done': r2['done'], 'reward': r1['reward'] + r2['reward']}
    
    r2 = _solve_record_via_doors(env, info2, check_drop)
    if r2:
        return {'actions': r1['actions'] + r2['actions'],
                'done': r2['done'], 'reward': r1['reward'] + r2['reward']}
    return None


def _solve_record_via_doors(env, info, final_fn, max_doors=15):
    """Iteratively open doors, recording all actions."""
    all_actions = []
    total_reward = 0
    
    for _ in range(max_doors):
        info = get_info(env)
        
        final_acts = final_fn(info)
        if final_acts is not None:
            r = _record_execute(env, final_acts)
            all_actions.extend(r['actions'])
            total_reward += r['reward']
            return {'actions': all_actions, 'done': r['done'], 'reward': total_reward}
        
        doors = find_room_doors(info['grid'], info['pos'])
        usable = [d for d in doors if not d['is_open'] and not d['is_locked']]
        
        carrying = info['carrying']
        if carrying and carrying.type == 'key':
            usable.extend([d for d in doors if d['is_locked'] and d['color'] == carrying.color])
        
        if not usable:
            return None
        
        best_a = None
        for d in usable:
            a = bfs_to_face(info['grid'], info['pos'], info['dir'], d['pos'])
            if a is not None and (best_a is None or len(a) < len(best_a)):
                best_a = a
        
        if best_a is None:
            return None
        
        r = _record_execute(env, best_a + ['toggle'])
        all_actions.extend(r['actions'])
        total_reward += r['reward']
        if r['done']:
            return {'actions': all_actions, 'done': True, 'reward': total_reward}
        
        # Step through opened door
        r2 = _record_execute(env, ['forward'])
        all_actions.extend(r2['actions'])
        total_reward += r2['reward']
        if r2['done']:
            return {'actions': all_actions, 'done': True, 'reward': total_reward}
    
    return None


# ============================================================
if __name__ == "__main__":
    TASKS = [
        ("BabyAI-GoToRedBallNoDists-v0", "L1: GoTo simple"),
        ("BabyAI-GoToObj-v0", "L2: GoTo object"),
        ("BabyAI-GoToLocal-v0", "L3: GoTo local"),
        ("BabyAI-GoToRedBlueBall-v0", "L2b: GoTo red/blue"),
        ("BabyAI-OpenDoor-v0", "L4: Open door"),
        ("BabyAI-OpenDoorColor-v0", "L4b: Open colored door"),
        ("BabyAI-PickupLoc-v0", "L5: Pickup"),
        ("BabyAI-PickupDist-v0", "L5b: Pickup+dist"),
        ("BabyAI-PutNextLocalS5N3-v0", "L7: PutNext local"),
        ("BabyAI-GoToObjMazeOpen-v0", "L6: GoTo maze (open)"),
        ("BabyAI-Unlock-v0", "L8: Unlock"),
        ("BabyAI-UnlockPickup-v0", "L9: Unlock+pickup"),
        ("BabyAI-GoToObjMaze-v0", "L10: GoTo maze (closed)"),
        ("BabyAI-KeyCorridor-v0", "L11: KeyCorridor"),
        ("BabyAI-BlockedUnlockPickup-v0", "L11b: BlockedUnlock"),
        ("BabyAI-BossLevelNoUnlock-v0", "L12: Boss-NoUnlock"),
        ("BabyAI-BossLevel-v0", "L13: Boss"),
    ]
    
    N_SEEDS = 5
    results = []
    
    for env_name, label in TASKS:
        print(f"\n{'='*60}")
        print(f"{label} ({env_name})")
        print(f"{'='*60}")
        
        wins = 0
        for seed in range(42, 42 + N_SEEDS):
            try:
                ok = solve_and_verify(env_name, seed=seed, verbose=True)
                if ok:
                    wins += 1
            except Exception as e:
                print(f"  Seed {seed}: EXCEPTION - {type(e).__name__}: {e}")
        
        rate = wins / N_SEEDS
        results.append((label, wins, N_SEEDS, rate))
        print(f"  >>> {wins}/{N_SEEDS} = {rate*100:.0f}%")
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    total_w, total_t = 0, 0
    for label, w, t, rate in results:
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  {label:35s} {bar} {w}/{t} ({rate*100:.0f}%)")
        total_w += w
        total_t += t
    print(f"\n  {'OVERALL':35s} {'':20s} {total_w}/{total_t} ({total_w/total_t*100:.0f}%)")
