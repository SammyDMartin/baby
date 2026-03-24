"""BabyAI solver - BFS pathfinding with iterative locked door handling."""
import gymnasium as gym
from collections import deque
from engine.grid import get_grid_info, find_objects, ACTION_MAP, DIR_DELTAS, DIR_NAMES


def solve_with_env(env):
    """Solve a pre-reset env. Return (success, actions)."""
    obs = env.unwrapped.gen_obs()
    mission = obs['mission']
    all_actions = []
    success = execute_mission(env, mission, all_actions)
    return success, all_actions


# ── Mission parsing ──────────────────────────────────────────────────────

def execute_mission(env, mission, all_actions):
    if ', then ' in mission:
        parts = mission.split(', then ', 1)
        if not execute_mission(env, parts[0].strip(), all_actions):
            return False
        return execute_mission(env, parts[1].strip(), all_actions)
    if ' and ' in mission:
        # Check for compound "X and Y" - but "go to" missions can have "and" in object names
        # BabyAI "and" missions: "do X and do Y" - both start with a verb
        parts = mission.split(' and ', 1)
        verb1_ok = any(parts[0].strip().startswith(v) for v in ('go to', 'pick up', 'put ', 'open'))
        verb2_ok = any(parts[1].strip().startswith(v) for v in ('go to', 'pick up', 'put ', 'open'))
        if verb1_ok and verb2_ok:
            if not execute_mission(env, parts[0].strip(), all_actions):
                return False
            return execute_mission(env, parts[1].strip(), all_actions)
    if mission.startswith('go to'):
        return do_goto(env, mission, all_actions)
    if mission.startswith('pick up'):
        return do_pickup(env, mission, all_actions)
    if mission.startswith('put'):
        return do_putnext(env, mission, all_actions)
    if mission.startswith('open'):
        return do_open(env, mission, all_actions)
    return False


def parse_object_desc(text):
    words = text.strip().split()
    words = [w for w in words if w not in ('the', 'a', 'an')]
    obj_type, color = None, None
    colors = {'red', 'green', 'blue', 'purple', 'yellow', 'grey'}
    types = {'ball', 'key', 'box', 'door'}
    for w in words:
        if w in colors:
            color = w
        if w in types:
            obj_type = w
    return color, obj_type


# ── State helpers ────────────────────────────────────────────────────────

def get_state(env):
    info = get_grid_info(env)
    return info['grid'], info['pos'], info['dir'], info['carrying']


def step_action(env, action, all_actions):
    obs, reward, done, truncated, info = env.step(ACTION_MAP[action])
    all_actions.append(action)
    return obs, reward, done, truncated


def direction_to(from_pos, to_pos):
    dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
    for d, (ddx, ddy) in DIR_DELTAS.items():
        if ddx == dx and ddy == dy:
            return d
    return None


def turn_actions(current_dir, target_dir):
    if current_dir == target_dir:
        return []
    diff = (target_dir - current_dir) % 4
    if diff == 1: return ['right']
    if diff == 2: return ['right', 'right']
    if diff == 3: return ['left']
    return []


def is_adjacent(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1


def face_target(env, direction, target_pos, all_actions):
    """Turn to face target_pos. Returns (new_dir, reward, done)."""
    grid, pos, direction, _ = get_state(env)
    target_dir = direction_to(pos, target_pos)
    if target_dir is None:
        return direction, 0, False
    for t in turn_actions(direction, target_dir):
        obs, reward, done, truncated = step_action(env, t, all_actions)
        if done:
            return target_dir, reward, True
    return target_dir, 0, False


# ── BFS pathfinding ──────────────────────────────────────────────────────

def bfs_reachable(grid, start, carrying=None):
    """BFS to find all reachable cells. Closed doors are passable (will toggle).
    Locked doors passable only if carrying matching key.
    Returns set of reachable positions."""
    visited = {start}
    queue = deque([start])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                continue
            cell = grid.get(nx, ny)
            if cell is None or cell.type == 'goal':
                visited.add((nx, ny))
                queue.append((nx, ny))
            elif cell.type == 'door':
                if cell.is_open or (not cell.is_locked):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                elif cell.is_locked and carrying and carrying.type == 'key' and carrying.color == cell.color:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
            # walls and objects block
    return visited


def bfs_path(grid, start, goal, carrying=None):
    """BFS shortest path from start to goal."""
    if start == goal:
        return [start]
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        (cx, cy), path = queue.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                continue
            cell = grid.get(nx, ny)
            passable = False
            if cell is None or cell.type == 'goal':
                passable = True
            elif cell.type == 'door':
                if cell.is_open or not cell.is_locked:
                    passable = True
                elif carrying and carrying.type == 'key' and carrying.color == cell.color:
                    passable = True
            if passable:
                visited.add((nx, ny))
                new_path = path + [(nx, ny)]
                if (nx, ny) == goal:
                    return new_path
                queue.append(((nx, ny), new_path))
    return None


def bfs_path_to_adjacent(grid, start, goal, carrying=None):
    """BFS to find shortest path to a cell adjacent to goal."""
    if is_adjacent(start, goal):
        return [start]
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        (cx, cy), path = queue.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited or (nx, ny) == goal:
                continue
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                continue
            cell = grid.get(nx, ny)
            passable = False
            if cell is None or cell.type == 'goal':
                passable = True
            elif cell.type == 'door':
                if cell.is_open or not cell.is_locked:
                    passable = True
                elif carrying and carrying.type == 'key' and carrying.color == cell.color:
                    passable = True
            if passable:
                visited.add((nx, ny))
                new_path = path + [(nx, ny)]
                if is_adjacent((nx, ny), goal):
                    return new_path
                queue.append(((nx, ny), new_path))
    return None


def find_frontier_locked_doors(grid, start, carrying=None):
    """Find locked doors adjacent to reachable cells (the 'frontier')."""
    reachable = bfs_reachable(grid, start, carrying)
    frontier = []
    seen = set()
    for rx, ry in reachable:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = rx + dx, ry + dy
            if (nx, ny) in seen or (nx, ny) in reachable:
                continue
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                continue
            cell = grid.get(nx, ny)
            if cell and cell.type == 'door' and cell.is_locked:
                seen.add((nx, ny))
                frontier.append({'pos': (nx, ny), 'color': cell.color})
    return frontier


# ── Path following ───────────────────────────────────────────────────────

def follow_path(env, path, all_actions):
    """Follow a BFS path, toggling closed doors. Returns (success, reward, done)."""
    for i in range(1, len(path)):
        grid, pos, direction, carrying = get_state(env)
        next_pos = path[i]
        target_dir = direction_to(pos, next_pos)
        if target_dir is None:
            return False, 0, False

        for t in turn_actions(direction, target_dir):
            obs, reward, done, truncated = step_action(env, t, all_actions)
            if done:
                return True, reward, True

        cell = grid.get(next_pos[0], next_pos[1])
        if cell and cell.type == 'door' and not cell.is_open:
            obs, reward, done, truncated = step_action(env, 'toggle', all_actions)
            if done:
                return True, reward, True
            # Check if door opened
            grid2, _, _, _ = get_state(env)
            door_cell = grid2.get(next_pos[0], next_pos[1])
            if door_cell and door_cell.type == 'door' and not door_cell.is_open:
                return False, 0, False  # toggle failed (wrong key?)
            obs, reward, done, truncated = step_action(env, 'forward', all_actions)
            if done:
                return True, reward, True
        else:
            obs, reward, done, truncated = step_action(env, 'forward', all_actions)
            if done:
                return True, reward, True

        _, new_pos, _, _ = get_state(env)
        if new_pos != next_pos:
            return False, 0, False
    return True, 0, False


# ── Smart navigation (handles locked doors iteratively) ──────────────────

def navigate_adjacent(env, target_pos, all_actions):
    """Navigate to be adjacent to and facing target_pos.
    Iteratively unlocks blocking doors. Returns (success, reward, done)."""
    for attempt in range(50):
        grid, pos, direction, carrying = get_state(env)

        if is_adjacent(pos, target_pos):
            new_dir, reward, done = face_target(env, direction, target_pos, all_actions)
            if done:
                return True, reward, True
            return True, 0, False

        path = bfs_path_to_adjacent(grid, pos, target_pos, carrying)
        if path is not None:
            success, reward, done = follow_path(env, path, all_actions)
            if done:
                return True, reward, True
            grid, pos, direction, carrying = get_state(env)
            if is_adjacent(pos, target_pos):
                new_dir, reward, done = face_target(env, direction, target_pos, all_actions)
                if done:
                    return True, reward, True
                return True, 0, False
            if success:
                continue  # position mismatch, retry
            continue  # path follow failed, retry (door might not have opened)

        # Path blocked by locked doors. Try to unlock one.
        unlocked = try_unlock_one_door(env, all_actions)
        if not unlocked:
            return False, 0, False
        # After unlocking, retry
    return False, 0, False


def navigate_to(env, target_pos, all_actions):
    """Navigate agent to target_pos exactly. Returns (success, reward, done)."""
    for attempt in range(50):
        grid, pos, direction, carrying = get_state(env)
        if pos == target_pos:
            return True, 0, False

        path = bfs_path(grid, pos, target_pos, carrying)
        if path is not None:
            success, reward, done = follow_path(env, path, all_actions)
            if done:
                return True, reward, True
            _, new_pos, _, _ = get_state(env)
            if new_pos == target_pos:
                return True, 0, False
            continue

        unlocked = try_unlock_one_door(env, all_actions)
        if not unlocked:
            return False, 0, False
    return False, 0, False


def try_move_blocking_object(env, all_actions):
    """Try to pick up and move an object that blocks navigation.
    Used when path is blocked by objects, not doors."""
    grid, pos, direction, carrying = get_state(env)
    if carrying:
        return False

    reachable_before = bfs_reachable(grid, pos, carrying)

    # Find objects on the border of reachable area
    blocking_objects = []
    seen = set()
    for rx, ry in list(reachable_before):
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = rx + dx, ry + dy
            if (nx, ny) in reachable_before or (nx, ny) in seen:
                continue
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                continue
            seen.add((nx, ny))
            cell = grid.get(nx, ny)
            if cell and cell.type in ('ball', 'key', 'box'):
                # Check if removing this object would expand reachable area
                # (the cell behind it might have walkable space)
                blocking_objects.append({'pos': (nx, ny), 'type': cell.type, 'color': cell.color})

    if not blocking_objects:
        return False

    # Prioritize non-key objects (avoid moving keys we might need)
    blocking_objects.sort(key=lambda o: 0 if o['type'] != 'key' else 1)

    for obj in blocking_objects:
        ox, oy = obj['pos']
        # Navigate adjacent to the object
        path = bfs_path_to_adjacent(grid, pos, (ox, oy), carrying)
        if path is None:
            continue
        success, reward, done = follow_path(env, path, all_actions)
        if done:
            return True
        grid, pos, direction, carrying = get_state(env)
        if not is_adjacent(pos, (ox, oy)):
            continue
        face_target(env, direction, (ox, oy), all_actions)
        obs, reward, done, truncated = step_action(env, 'pickup', all_actions)
        if done:
            return True
        _, _, _, c = get_state(env)
        if not c:
            continue  # pickup failed

        # Check if this actually expanded our reachable area
        grid, pos, direction, carrying = get_state(env)
        reachable_after = bfs_reachable(grid, pos, carrying)
        if len(reachable_after) > len(reachable_before):
            # Good, we expanded. Drop the object somewhere that doesn't block.
            # Turn away from the picked-up position first so we don't drop it back.
            grid, pos, direction, carrying = get_state(env)
            # Find an empty cell NOT at the original position to drop into
            dropped = False
            for d in range(4):
                ddx, ddy = DIR_DELTAS[d]
                fx, fy = pos[0] + ddx, pos[1] + ddy
                if (fx, fy) == (ox, oy):
                    continue  # don't drop back to original position
                if 0 <= fx < grid.width and 0 <= fy < grid.height:
                    cell = grid.get(fx, fy)
                    if cell is None:
                        for t in turn_actions(direction, d):
                            step_action(env, t, all_actions)
                        step_action(env, 'drop', all_actions)
                        _, _, _, c = get_state(env)
                        if not c:
                            dropped = True
                            break
            if not dropped:
                # Move somewhere else first, then drop
                for d in range(4):
                    ddx, ddy = DIR_DELTAS[d]
                    fx, fy = pos[0] + ddx, pos[1] + ddy
                    if 0 <= fx < grid.width and 0 <= fy < grid.height:
                        cell = grid.get(fx, fy)
                        if cell is None and (fx, fy) != (ox, oy):
                            for t in turn_actions(direction, d):
                                step_action(env, t, all_actions)
                            step_action(env, 'forward', all_actions)
                            try_drop(env, all_actions)
                            dropped = True
                            break
                if not dropped:
                    try_drop(env, all_actions)
            return True
        else:
            # Didn't expand. Drop it back and try next.
            try_drop(env, all_actions)
            grid, pos, direction, carrying = get_state(env)
            continue

    return False


def try_unlock_one_door(env, all_actions):
    """Try to unlock one frontier locked door. Returns True if succeeded."""
    grid, pos, direction, carrying = get_state(env)

    frontier = find_frontier_locked_doors(grid, pos, carrying)
    if not frontier:
        # Maybe blocked by objects, not doors
        return try_move_blocking_object(env, all_actions)

    for door_info in frontier:
        door_pos = door_info['pos']
        door_color = door_info['color']

        # Check if we already have the matching key
        if carrying and carrying.type == 'key' and carrying.color == door_color:
            # Navigate to door and unlock
            path = bfs_path_to_adjacent(grid, pos, door_pos, carrying)
            if path is not None:
                success, reward, done = follow_path(env, path, all_actions)
                if done:
                    return True
                grid, pos, direction, carrying = get_state(env)
                if is_adjacent(pos, door_pos):
                    face_target(env, direction, door_pos, all_actions)
                    obs, reward, done, truncated = step_action(env, 'toggle', all_actions)
                    if done:
                        return True
                    # Verify door opened
                    grid, _, _, _ = get_state(env)
                    dcell = grid.get(door_pos[0], door_pos[1])
                    if dcell and dcell.type == 'door' and dcell.is_open:
                        return True
            continue

        # Need to find and pick up the matching key
        keys = find_objects(grid, obj_type='key', color=door_color)
        reachable = bfs_reachable(grid, pos, carrying)

        accessible_keys = []
        for k in keys:
            kp = k['pos']
            # Key is accessible if some cell adjacent to it is reachable
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if (kp[0] + dx, kp[1] + dy) in reachable:
                    accessible_keys.append(k)
                    break

        if not accessible_keys:
            continue  # can't reach this key

        # Drop current item if carrying
        if carrying:
            if not try_drop(env, all_actions):
                continue
            grid, pos, direction, carrying = get_state(env)

        # Pick up the key
        key_pos = accessible_keys[0]['pos']
        path = bfs_path_to_adjacent(grid, pos, key_pos, carrying)
        if path is None:
            continue
        success, reward, done = follow_path(env, path, all_actions)
        if done:
            return True

        grid, pos, direction, carrying = get_state(env)
        if not is_adjacent(pos, key_pos):
            continue
        face_target(env, direction, key_pos, all_actions)
        obs, reward, done, truncated = step_action(env, 'pickup', all_actions)
        if done:
            return True

        # Verify we picked up the right key
        grid, pos, direction, carrying = get_state(env)
        if not (carrying and carrying.type == 'key' and carrying.color == door_color):
            continue  # pickup failed or got wrong item

        # Now navigate to door and unlock
        path = bfs_path_to_adjacent(grid, pos, door_pos, carrying)
        if path is None:
            # Maybe we can reach it now with the key
            continue
        success, reward, done = follow_path(env, path, all_actions)
        if done:
            return True

        grid, pos, direction, carrying = get_state(env)
        if is_adjacent(pos, door_pos):
            face_target(env, direction, door_pos, all_actions)
            obs, reward, done, truncated = step_action(env, 'toggle', all_actions)
            if done:
                return True
            grid, _, _, _ = get_state(env)
            dcell = grid.get(door_pos[0], door_pos[1])
            if dcell and dcell.type == 'door' and dcell.is_open:
                return True

    # If all doors failed, maybe objects are blocking access to doors
    return try_move_blocking_object(env, all_actions)


# ── Drop helper ──────────────────────────────────────────────────────────

def try_drop(env, all_actions):
    """Drop carried item. Returns True if dropped."""
    grid, pos, direction, carrying = get_state(env)
    if not carrying:
        return True
    dirs_to_try = [direction] + [d for d in range(4) if d != direction]
    for d in dirs_to_try:
        ddx, ddy = DIR_DELTAS[d]
        fx, fy = pos[0] + ddx, pos[1] + ddy
        if 0 <= fx < grid.width and 0 <= fy < grid.height:
            cell = grid.get(fx, fy)
            if cell is None:
                for t in turn_actions(direction, d):
                    obs, reward, done, truncated = step_action(env, t, all_actions)
                    if done:
                        return True
                direction = d
                obs, reward, done, truncated = step_action(env, 'drop', all_actions)
                _, _, _, c = get_state(env)
                if not c:
                    return True
    # No empty adjacent cell. Try moving somewhere with an empty adjacent cell.
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = pos[0] + dx, pos[1] + dy
        if not (0 <= nx < grid.width and 0 <= ny < grid.height):
            continue
        cell = grid.get(nx, ny)
        if cell is None:
            # Move there
            target_dir = direction_to(pos, (nx, ny))
            for t in turn_actions(direction, target_dir):
                step_action(env, t, all_actions)
            step_action(env, 'forward', all_actions)
            grid, pos, direction, carrying = get_state(env)
            if carrying:
                return try_drop(env, all_actions)  # retry from new position
            return True
    return False


# ── Mission handlers ─────────────────────────────────────────────────────

def do_goto(env, mission, all_actions):
    text = mission.replace('go to ', '', 1)
    color, obj_type = parse_object_desc(text)
    grid, pos, _, _ = get_state(env)
    targets = find_objects(grid, obj_type=obj_type, color=color)
    if not targets:
        return False
    targets.sort(key=lambda t: abs(t['pos'][0] - pos[0]) + abs(t['pos'][1] - pos[1]))
    for target in targets:
        success, reward, done = navigate_adjacent(env, target['pos'], all_actions)
        if done and reward > 0:
            return True
        if success:
            return True
    return False


def do_pickup(env, mission, all_actions):
    text = mission.replace('pick up ', '', 1)
    color, obj_type = parse_object_desc(text)
    grid, pos, _, carrying = get_state(env)

    if carrying:
        try_drop(env, all_actions)

    grid, pos, _, carrying = get_state(env)
    targets = find_objects(grid, obj_type=obj_type, color=color)
    if not targets:
        return False
    targets.sort(key=lambda t: abs(t['pos'][0] - pos[0]) + abs(t['pos'][1] - pos[1]))
    for target in targets:
        success, reward, done = navigate_adjacent(env, target['pos'], all_actions)
        if done:
            return reward > 0
        if success:
            # Make sure not carrying anything
            grid, pos, _, carrying = get_state(env)
            if carrying:
                try_drop(env, all_actions)
                # Re-navigate
                success, reward, done = navigate_adjacent(env, target['pos'], all_actions)
                if done:
                    return reward > 0
                if not success:
                    continue
            obs, reward, done, truncated = step_action(env, 'pickup', all_actions)
            if done:
                return reward > 0
            _, _, _, c = get_state(env)
            if c and (color is None or c.color == color) and (obj_type is None or c.type == obj_type):
                return True
    return False


def do_open(env, mission, all_actions):
    text = mission.replace('open ', '', 1)
    color, obj_type = parse_object_desc(text)
    grid, pos, _, _ = get_state(env)
    doors = find_objects(grid, obj_type='door', color=color)
    doors = [d for d in doors if not d.get('is_open', False)]
    if not doors:
        return False
    doors.sort(key=lambda d: abs(d['pos'][0] - pos[0]) + abs(d['pos'][1] - pos[1]))
    for door in doors:
        door_pos = door['pos']
        if door.get('is_locked', False):
            # Need matching key
            grid, pos, _, carrying = get_state(env)
            if not (carrying and carrying.type == 'key' and carrying.color == door['color']):
                if carrying:
                    try_drop(env, all_actions)
                grid, pos, _, carrying = get_state(env)
                keys = find_objects(grid, obj_type='key', color=door['color'])
                if not keys:
                    continue
                keys.sort(key=lambda k: abs(k['pos'][0] - pos[0]) + abs(k['pos'][1] - pos[1]))
                success, reward, done = navigate_adjacent(env, keys[0]['pos'], all_actions)
                if done:
                    return reward > 0
                if not success:
                    continue
                obs, reward, done, truncated = step_action(env, 'pickup', all_actions)
                if done:
                    return reward > 0

            success, reward, done = navigate_adjacent(env, door_pos, all_actions)
            if done:
                return reward > 0
            if success:
                obs, reward, done, truncated = step_action(env, 'toggle', all_actions)
                if done:
                    return reward > 0
                return True
        else:
            success, reward, done = navigate_adjacent(env, door_pos, all_actions)
            if done:
                return reward > 0
            if success:
                obs, reward, done, truncated = step_action(env, 'toggle', all_actions)
                if done:
                    return reward > 0
                return True
    return False


def do_putnext(env, mission, all_actions):
    m = mission.replace('put ', '', 1)
    parts = m.split(' next to ')
    if len(parts) != 2:
        return False
    color_a, type_a = parse_object_desc(parts[0])
    color_b, type_b = parse_object_desc(parts[1])

    grid, pos, _, carrying = get_state(env)

    already_have = (carrying and
                    (color_a is None or carrying.color == color_a) and
                    (type_a is None or carrying.type == type_a))
    if not already_have:
        if carrying:
            try_drop(env, all_actions)
        grid, pos, _, _ = get_state(env)
        targets_a = find_objects(grid, obj_type=type_a, color=color_a)
        if not targets_a:
            return False
        targets_a.sort(key=lambda t: abs(t['pos'][0] - pos[0]) + abs(t['pos'][1] - pos[1]))
        picked = False
        for ta in targets_a:
            success, reward, done = navigate_adjacent(env, ta['pos'], all_actions)
            if done:
                return reward > 0
            if success:
                obs, reward, done, truncated = step_action(env, 'pickup', all_actions)
                if done:
                    return reward > 0
                _, _, _, c = get_state(env)
                if c:
                    picked = True
                    break
        if not picked:
            return False

    grid, pos, _, carrying = get_state(env)
    targets_b = find_objects(grid, obj_type=type_b, color=color_b)
    if not targets_b:
        return False
    targets_b.sort(key=lambda t: abs(t['pos'][0] - pos[0]) + abs(t['pos'][1] - pos[1]))
    target_b_pos = targets_b[0]['pos']

    return drop_next_to(env, target_b_pos, all_actions)


def drop_next_to(env, target_pos, all_actions):
    """Drop carried item in an empty cell adjacent to target_pos.
    Strategy: find empty cells adjacent to target, then find a cell to stand
    on (adjacent to that empty cell, not the empty cell itself or target),
    navigate there, face the empty cell, drop."""
    grid, pos, direction, carrying = get_state(env)
    if not carrying:
        return False
    tx, ty = target_pos

    # Find all empty cells adjacent to target (potential drop locations)
    drop_candidates = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        cx, cy = tx + dx, ty + dy
        if 0 <= cx < grid.width and 0 <= cy < grid.height:
            cell = grid.get(cx, cy)
            if cell is None:
                drop_candidates.append((cx, cy))

    if not drop_candidates:
        return False

    # For each drop cell, find a stand position (adjacent to drop cell, not target or drop cell)
    best = None
    best_dist = float('inf')
    for drop_x, drop_y in drop_candidates:
        for dx2, dy2 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            sx, sy = drop_x + dx2, drop_y + dy2
            if (sx, sy) == target_pos or (sx, sy) == (drop_x, drop_y):
                continue
            if not (0 <= sx < grid.width and 0 <= sy < grid.height):
                continue
            s_cell = grid.get(sx, sy)
            if s_cell is not None and (sx, sy) != pos:
                continue  # not empty (unless it's where we are)
            # Can we reach this stand position?
            path = bfs_path(grid, pos, (sx, sy), carrying)
            if path is not None:
                dist = len(path)
                if dist < best_dist:
                    best_dist = dist
                    best = (sx, sy, drop_x, drop_y)

    # Also check: can we stand ON a drop candidate cell and drop into another?
    # i.e., stand at drop_cell_1, face drop_cell_2 (which is adjacent to target AND to us)
    # This isn't quite right. Let me also try: if we're adjacent to target, we can drop
    # into a cell that's adjacent to target.
    for drop_x, drop_y in drop_candidates:
        # Can we be at pos adjacent to target AND face (drop_x, drop_y)?
        # We need: pos adjacent to (drop_x, drop_y) AND pos on a reachable cell
        # But the above loop already handles this case (stand adjacent to drop cell)
        pass

    if best is None:
        # Last resort: just navigate adjacent to target and try any direction
        success, reward, done = navigate_adjacent(env, target_pos, all_actions)
        if done:
            return reward > 0
        if not success:
            return False
        grid, pos, direction, carrying = get_state(env)
        for d in range(4):
            ddx, ddy = DIR_DELTAS[d]
            fx, fy = pos[0] + ddx, pos[1] + ddy
            if 0 <= fx < grid.width and 0 <= fy < grid.height:
                cell = grid.get(fx, fy)
                if cell is None:
                    for t in turn_actions(direction, d):
                        obs, reward, done, truncated = step_action(env, t, all_actions)
                        if done:
                            return reward > 0
                    obs, reward, done, truncated = step_action(env, 'drop', all_actions)
                    if done:
                        return reward > 0
                    _, _, _, c = get_state(env)
                    if not c:
                        return True
        return False

    stand_x, stand_y, drop_x, drop_y = best

    # Navigate to stand position
    if pos != (stand_x, stand_y):
        path = bfs_path(grid, pos, (stand_x, stand_y), carrying)
        if path is None:
            return False
        success, reward, done = follow_path(env, path, all_actions)
        if done:
            return reward > 0
        if not success:
            return False

    # Face the drop cell
    grid, pos, direction, carrying = get_state(env)
    drop_dir = direction_to(pos, (drop_x, drop_y))
    if drop_dir is None:
        return False
    for t in turn_actions(direction, drop_dir):
        obs, reward, done, truncated = step_action(env, t, all_actions)
        if done:
            return reward > 0

    # Drop
    obs, reward, done, truncated = step_action(env, 'drop', all_actions)
    if done:
        return reward > 0
    _, _, _, c = get_state(env)
    if not c:
        return True
    return False
