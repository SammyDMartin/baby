"""
My BabyAI solver - written from scratch without reading solvers/.

Approach:
  1. Parse the mission text into structured subgoals
  2. For each subgoal, use BFS to find action sequences
  3. Handle multi-room navigation by iteratively opening doors
  4. Handle locked doors by finding keys
  5. Handle compound missions by solving subgoals sequentially

All BFS/pathfinding code is written from first principles here.
"""
import re
import gymnasium as gym
from collections import deque
from engine.grid import get_grid_info, find_objects, ACTION_MAP, DIR_DELTAS, DIR_NAMES


# ─── Pathfinding ─────────────────────────────────────────────────────────────

def can_walk(grid, x, y):
    """Can the agent walk into cell (x,y)?"""
    if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
        return False
    cell = grid.get(x, y)
    if cell is None:
        return True  # empty
    if cell.type == 'door' and cell.is_open:
        return True
    return False


def bfs_to(grid, start_pos, start_dir, target_pos, face_dir=None):
    """BFS to find action sequence from start to target.
    State = (x, y, dir). Returns list of action strings or None."""
    sx, sy = start_pos
    queue = deque([(sx, sy, start_dir, [])])
    visited = {(sx, sy, start_dir)}

    while queue:
        x, y, d, actions = queue.popleft()
        if len(actions) > 400:
            continue

        if (x, y) == target_pos and (face_dir is None or d == face_dir):
            return actions

        # Turn left
        nd = (d - 1) % 4
        if (x, y, nd) not in visited:
            visited.add((x, y, nd))
            queue.append((x, y, nd, actions + ['left']))

        # Turn right
        nd = (d + 1) % 4
        if (x, y, nd) not in visited:
            visited.add((x, y, nd))
            queue.append((x, y, nd, actions + ['right']))

        # Forward
        dx, dy = DIR_DELTAS[d]
        nx, ny = x + dx, y + dy
        if can_walk(grid, nx, ny) and (nx, ny, d) not in visited:
            visited.add((nx, ny, d))
            queue.append((nx, ny, d, actions + ['forward']))

    return None


def bfs_face_target(grid, start_pos, start_dir, target_pos):
    """Find shortest action sequence to stand adjacent to target, facing it."""
    best = None
    for face_d, (dx, dy) in DIR_DELTAS.items():
        # Stand at target - delta, facing face_d
        sx = target_pos[0] - dx
        sy = target_pos[1] - dy
        if sx < 0 or sy < 0 or sx >= grid.width or sy >= grid.height:
            continue
        # Must be able to stand there (or already there)
        if (sx, sy) != start_pos:
            cell = grid.get(sx, sy)
            if cell is not None and not (cell.type == 'door' and cell.is_open):
                continue
        acts = bfs_to(grid, start_pos, start_dir, (sx, sy), face_d)
        if acts is not None and (best is None or len(acts) < len(best)):
            best = acts
    return best


def flood_reachable(grid, start):
    """Flood fill to find all walkable cells from start."""
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited and can_walk(grid, nx, ny):
                visited.add((nx, ny))
                queue.append((nx, ny))
    return visited


def find_nearby_doors(grid, start):
    """Find closed/locked doors bordering the reachable area.
    Also looks through movable objects (they might be hiding doors)."""
    reachable = flood_reachable(grid, start)

    # Extend check through movable objects adjacent to reachable cells
    extra = set()
    for x, y in reachable:
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in reachable and 0 <= nx < grid.width and 0 <= ny < grid.height:
                c = grid.get(nx, ny)
                if c and c.type in ('ball', 'key', 'box'):
                    extra.add((nx, ny))

    check = reachable | extra
    doors = []
    seen = set()
    for x, y in check:
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in seen or (nx, ny) in reachable:
                continue
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                c = grid.get(nx, ny)
                if c and c.type == 'door':
                    seen.add((nx, ny))
                    doors.append({
                        'pos': (nx, ny),
                        'color': getattr(c, 'color', None),
                        'is_locked': c.is_locked,
                        'is_open': c.is_open,
                    })
    return doors


# ─── Mission parsing ─────────────────────────────────────────────────────────

COLORS = {'red', 'green', 'blue', 'purple', 'yellow', 'grey'}
TYPES = {'ball', 'key', 'box', 'door'}


def parse_mission(mission):
    """Parse mission into list of subgoals."""
    m = mission.strip().lower()

    # Sequential: "X, then Y" / "X then Y"
    for sep in [', then ', ' then ']:
        if sep in m:
            parts = m.split(sep, 1)
            return [_parse_one(parts[0].strip()), _parse_one(parts[1].strip())]

    # Compound: "X and Y" - try each split point
    if ' and ' in m:
        idx = 0
        while True:
            pos = m.find(' and ', idx)
            if pos == -1:
                break
            left = _parse_one(m[:pos].strip())
            right = _parse_one(m[pos + 5:].strip())
            if left['action'] != 'unknown' and right['action'] != 'unknown':
                return [left, right]
            idx = pos + 5

    return [_parse_one(m)]


def _parse_one(text):
    """Parse a single instruction."""
    # put X next to Y
    m = re.match(r'put (?:the |a )?(\w+) (\w+) next to (?:the |a )?(\w+) (\w+)', text)
    if m:
        return {'action': 'putnext', 'c1': m.group(1), 't1': m.group(2),
                'c2': m.group(3), 't2': m.group(4)}

    # go to (with optional "on your X" / "behind you")
    m = re.match(r'go to (?:the |a )?(\w+) (\w+)', text)
    if m:
        return {'action': 'goto', 'color': m.group(1), 'type': m.group(2)}
    m = re.match(r'go to (?:the |a )?(\w+)', text)
    if m:
        return {'action': 'goto', 'color': None, 'type': m.group(1)}

    # pick up with relative direction
    m = re.match(r'pick up (?:the |a )?(\w+) on your (\w+)', text)
    if m:
        return {'action': 'pickup', 'color': None, 'type': m.group(1), 'rel': m.group(2)}
    m = re.match(r'pick up (?:the |a )?(\w+) behind you', text)
    if m:
        return {'action': 'pickup', 'color': None, 'type': m.group(1), 'rel': 'behind'}

    # pick up color type
    m = re.match(r'pick up (?:the |a )?(\w+) (\w+)', text)
    if m:
        w1, w2 = m.group(1), m.group(2)
        if w2 in TYPES and w1 in COLORS:
            return {'action': 'pickup', 'color': w1, 'type': w2}
        elif w1 in TYPES:
            return {'action': 'pickup', 'color': None, 'type': w1}
        else:
            return {'action': 'pickup', 'color': None, 'type': w2 if w2 in TYPES else w1}

    m = re.match(r'pick up (?:the |a )?(\w+)', text)
    if m:
        return {'action': 'pickup', 'color': None, 'type': m.group(1)}

    # open door
    m = re.match(r'open (?:the |a )?door on your (\w+)', text)
    if m:
        return {'action': 'open_rel', 'rel': m.group(1)}
    m = re.match(r'open (?:the |a )?(\w+) door', text)
    if m and m.group(1) not in ('the', 'a'):
        return {'action': 'open', 'color': m.group(1)}
    m = re.match(r'open (?:the |a )?door', text)
    if m:
        return {'action': 'open', 'color': None}

    return {'action': 'unknown', 'raw': text}


# ─── Action execution ────────────────────────────────────────────────────────

def step_actions(env, actions):
    """Execute action list on env. Return (success, reward, num_executed)."""
    total_r = 0
    for i, a in enumerate(actions):
        obs, r, done, trunc, _ = env.step(ACTION_MAP[a])
        total_r += r
        if done:
            return (r > 0, total_r, i + 1)
        if trunc:
            return (False, total_r, i + 1)
    return (False, total_r, len(actions))


def drop_item(env, info):
    """Drop carried item. Returns action list or None."""
    if not info['carrying']:
        return []
    d = info['dir']
    pos = info['pos']
    grid = info['grid']
    w, h = info['w'], info['h']
    # Try behind, sides, then forward
    for offset in [2, 3, 1, 0]:
        td = (d + offset) % 4
        dx, dy = DIR_DELTAS[td]
        tx, ty = pos[0] + dx, pos[1] + dy
        if 0 <= tx < w and 0 <= ty < h:
            cell = grid.get(tx, ty)
            if cell is None:
                turns = (td - d) % 4
                acts = []
                if turns == 1:
                    acts = ['right']
                elif turns == 2:
                    acts = ['right', 'right']
                elif turns == 3:
                    acts = ['left']
                return acts + ['drop']
    return None


# ─── Door navigation engine ─────────────────────────────────────────────────

def navigate_opening_doors(env, goal_fn, max_iter=80):
    """Keep opening doors until goal_fn(info) returns an action list.
    Handles closed doors and locked doors (if carrying matching key).
    Returns (all_actions, success) or ([], False)."""
    all_actions = []

    for _ in range(max_iter):
        info = get_grid_info(env)
        result = goal_fn(info)
        if result is not None:
            ok, r, n = step_actions(env, result)
            all_actions.extend(result[:n])
            return all_actions, ok or (r > 0)

        doors = find_nearby_doors(info['grid'], info['pos'])
        openable = [d for d in doors if not d['is_open'] and not d['is_locked']]

        # If carrying a key, also consider locked doors of matching color
        if info['carrying'] and info['carrying'].type == 'key':
            key_color = info['carrying'].color
            openable.extend([d for d in doors if d['is_locked'] and d['color'] == key_color])

        if not openable:
            return all_actions, False

        # Pick door closest to... well, just pick shortest path
        best_acts = None
        for door in openable:
            acts = bfs_face_target(info['grid'], info['pos'], info['dir'], door['pos'])
            if acts is not None and (best_acts is None or len(acts) < len(best_acts)):
                best_acts = acts

        if best_acts is None:
            return all_actions, False

        # Go to door, toggle, step through
        ok, r, n = step_actions(env, best_acts + ['toggle'])
        all_actions.extend((best_acts + ['toggle'])[:n])
        if ok:
            return all_actions, True

        ok2, r2, n2 = step_actions(env, ['forward'])
        all_actions.extend(['forward'][:n2])
        if ok2:
            return all_actions, True

    return all_actions, False


# ─── Blocker handling ────────────────────────────────────────────────────────

def move_blocker_near_door(env, info, door_pos):
    """If an object is blocking access to a door, pick it up and drop elsewhere.
    Returns (actions, success)."""
    grid = info['grid']
    all_acts = []

    # Check cells adjacent to door for movable objects
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        bx, by = door_pos[0] + dx, door_pos[1] + dy
        if 0 <= bx < info['w'] and 0 <= by < info['h']:
            cell = grid.get(bx, by)
            if cell and cell.type in ('ball', 'key', 'box'):
                # This might be blocking. Try to pick up and move it.
                # First drop what we're carrying
                if info['carrying']:
                    drop = drop_item(env, info)
                    if drop:
                        ok, r, n = step_actions(env, drop)
                        all_acts.extend(drop[:n])
                        info = get_grid_info(env)

                acts = bfs_face_target(info['grid'], info['pos'], info['dir'], (bx, by))
                if acts is not None:
                    ok, r, n = step_actions(env, acts + ['pickup'])
                    all_acts.extend((acts + ['pickup'])[:n])
                    if ok:
                        return all_acts, True
                    # Drop it somewhere
                    info2 = get_grid_info(env)
                    drop = drop_item(env, info2)
                    if drop:
                        ok, r, n = step_actions(env, drop)
                        all_acts.extend(drop[:n])
                    return all_acts, False  # Moved blocker, caller should retry

    return all_acts, False


# ─── Key management ──────────────────────────────────────────────────────────

def ensure_have_key(env, key_color, depth=0):
    """Make sure we're carrying the right key. Handles chained dependencies."""
    if depth > 6:
        return [], False

    info = get_grid_info(env)

    # Already have it?
    if info['carrying'] and info['carrying'].type == 'key' and info['carrying'].color == key_color:
        return [], True

    all_actions = []

    # Drop whatever we're carrying
    if info['carrying']:
        drop = drop_item(env, info)
        if drop:
            ok, r, n = step_actions(env, drop)
            all_actions.extend(drop[:n])
            if ok:
                return all_actions, True
            info = get_grid_info(env)

    # Try to find and pick up the key directly
    def find_key(i):
        keys = find_objects(i['grid'], obj_type='key', color=key_color)
        for k in keys:
            acts = bfs_face_target(i['grid'], i['pos'], i['dir'], k['pos'])
            if acts is not None:
                return acts + ['pickup']
        return None

    # Check if key is reachable now
    result = find_key(info)
    if result is not None:
        ok, r, n = step_actions(env, result)
        all_actions.extend(result[:n])
        return all_actions, True

    # Key might be behind doors - try opening them
    # But first check if we're blocked by a LOCKED door that needs a DIFFERENT key
    doors = find_nearby_doors(info['grid'], info['pos'])
    locked = [d for d in doors if d['is_locked']]

    for ld in locked:
        # Recursively get the key for this locked door
        sub_acts, got_it = ensure_have_key(env, ld['color'], depth + 1)
        all_actions.extend(sub_acts)
        if not got_it:
            continue

        # Unlock this door and step through
        info2 = get_grid_info(env)
        face_acts = bfs_face_target(info2['grid'], info2['pos'], info2['dir'], ld['pos'])
        if face_acts is not None:
            ok, r, n = step_actions(env, face_acts + ['toggle', 'forward'])
            all_actions.extend((face_acts + ['toggle', 'forward'])[:n])
            if ok:
                return all_actions, True

            # Drop the door key
            info3 = get_grid_info(env)
            if info3['carrying']:
                drop = drop_item(env, info3)
                if drop:
                    ok, r, n = step_actions(env, drop)
                    all_actions.extend(drop[:n])

            # Now try finding our target key through open doors
            nav_acts, found = navigate_opening_doors(env, find_key)
            all_actions.extend(nav_acts)
            if found:
                return all_actions, True
            i_check = get_grid_info(env)
            if i_check['carrying'] and i_check['carrying'].type == 'key' and i_check['carrying'].color == key_color:
                return all_actions, True

    # No locked doors blocking - just navigate through closed doors
    nav_acts, found = navigate_opening_doors(env, find_key)
    all_actions.extend(nav_acts)
    if found:
        return all_actions, True
    # Check if we got the key even though mission didn't complete
    info_check = get_grid_info(env)
    if info_check['carrying'] and info_check['carrying'].type == 'key' and info_check['carrying'].color == key_color:
        return all_actions, True
    return all_actions, False


# ─── Relative direction helper ───────────────────────────────────────────────

def relative_direction(agent_pos, agent_dir, target_pos):
    """Score how much a target is in a given relative direction from agent."""
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
    fdx, fdy = DIR_DELTAS[agent_dir]
    cross = fdx * dy - fdy * dx
    dot = fdx * dx + fdy * dy
    return cross, dot  # cross<0 = left, cross>0 = right, dot>0 = front


def sort_by_relative(targets, agent_pos, agent_dir, rel_dir):
    """Sort targets by how well they match a relative direction."""
    if not rel_dir:
        return targets
    scored = []
    for t in targets:
        cross, dot = relative_direction(agent_pos, agent_dir, t['pos'])
        if rel_dir == 'left':
            scored.append((cross, t))  # most negative = most left
        elif rel_dir == 'right':
            scored.append((-cross, t))
        elif rel_dir == 'front':
            scored.append((-dot, t))
        elif rel_dir == 'behind':
            scored.append((dot, t))
        else:
            scored.append((0, t))
    scored.sort(key=lambda x: x[0])
    return [s[1] for s in scored]


# ─── Subgoal solvers ────────────────────────────────────────────────────────

def solve_goto(env, info, sg):
    color, otype = sg.get('color'), sg.get('type')
    targets = find_objects(info['grid'], obj_type=otype, color=color)
    if not targets and color:
        targets = find_objects(info['grid'], obj_type=otype)

    # Try direct
    for t in targets:
        acts = bfs_face_target(info['grid'], info['pos'], info['dir'], t['pos'])
        if acts is not None:
            ok, r, n = step_actions(env, acts)
            return acts[:n], ok

    # Through doors
    def check(i):
        ts = find_objects(i['grid'], obj_type=otype, color=color)
        if not ts and color:
            ts = find_objects(i['grid'], obj_type=otype)
        for t in ts:
            a = bfs_face_target(i['grid'], i['pos'], i['dir'], t['pos'])
            if a is not None:
                return a
        return None

    return navigate_opening_doors(env, check)


def solve_open(env, info, sg):
    color = sg.get('color')
    doors = find_objects(info['grid'], obj_type='door', color=color)
    if not doors:
        doors = find_objects(info['grid'], obj_type='door')
    if not doors:
        return [], False

    target = doors[0]
    all_acts = []

    if target.get('is_locked'):
        # Get key first
        key_acts, got = ensure_have_key(env, target['color'])
        all_acts.extend(key_acts)
        if not got:
            return all_acts, False
        info = get_grid_info(env)

    # Navigate to door and toggle
    def check(i):
        a = bfs_face_target(i['grid'], i['pos'], i['dir'], target['pos'])
        return a + ['toggle'] if a is not None else None

    nav_acts, ok = navigate_opening_doors(env, check)
    all_acts.extend(nav_acts)
    return all_acts, ok


def solve_open_rel(env, info, sg):
    rel = sg.get('rel')
    doors = find_nearby_doors(info['grid'], info['pos'])
    # Sort by relative direction
    scored = []
    for d in doors:
        cross, dot = relative_direction(info['pos'], info['dir'], d['pos'])
        if rel == 'left':
            scored.append((cross, d))
        elif rel == 'right':
            scored.append((-cross, d))
        elif rel == 'front':
            scored.append((-dot, d))
        else:
            scored.append((dot, d))
    scored.sort(key=lambda x: x[0])

    for _, door in scored:
        acts = bfs_face_target(info['grid'], info['pos'], info['dir'], door['pos'])
        if acts is not None:
            ok, r, n = step_actions(env, acts + ['toggle'])
            return (acts + ['toggle'])[:n], ok

    return [], False


def solve_pickup(env, info, sg):
    color, otype = sg.get('color'), sg.get('type')
    rel = sg.get('rel')
    all_acts = []

    # Drop if carrying
    if info['carrying']:
        drop = drop_item(env, info)
        if drop:
            ok, r, n = step_actions(env, drop)
            all_acts.extend(drop[:n])
            if ok:
                return all_acts, True
            info = get_grid_info(env)

    # Iteratively: try direct, open doors, handle locked doors
    for attempt in range(15):
        info = get_grid_info(env)

        if info['carrying']:
            drop = drop_item(env, info)
            if drop:
                ok, r, n = step_actions(env, drop)
                all_acts.extend(drop[:n])
                if ok:
                    return all_acts, True
                info = get_grid_info(env)

        targets = find_objects(info['grid'], obj_type=otype, color=color)
        if not targets and color:
            targets = find_objects(info['grid'], obj_type=otype)
        targets = sort_by_relative(targets, info['pos'], info['dir'], rel)

        for t in targets:
            acts = bfs_face_target(info['grid'], info['pos'], info['dir'], t['pos'])
            if acts is not None:
                ok, r, n = step_actions(env, acts + ['pickup'])
                all_acts.extend((acts + ['pickup'])[:n])
                return all_acts, ok

        # Not directly reachable - try doors
        doors = find_nearby_doors(info['grid'], info['pos'])
        closed = [d for d in doors if not d['is_open'] and not d['is_locked']]
        locked = [d for d in doors if d['is_locked']]

        if closed:
            best_a = None
            for d in closed:
                a = bfs_face_target(info['grid'], info['pos'], info['dir'], d['pos'])
                if a is not None and (best_a is None or len(a) < len(best_a)):
                    best_a = a
            if best_a:
                ok, r, n = step_actions(env, best_a + ['toggle', 'forward'])
                all_acts.extend((best_a + ['toggle', 'forward'])[:n])
                if ok:
                    return all_acts, True
                continue

        if locked:
            for ld in locked:
                key_acts, got = ensure_have_key(env, ld['color'])
                all_acts.extend(key_acts)
                if got:
                    info2 = get_grid_info(env)
                    face = bfs_face_target(info2['grid'], info2['pos'], info2['dir'], ld['pos'])
                    if face:
                        ok, r, n = step_actions(env, face + ['toggle', 'forward'])
                        all_acts.extend((face + ['toggle', 'forward'])[:n])
                        if ok:
                            return all_acts, True
                        # Drop key
                        info3 = get_grid_info(env)
                        if info3['carrying']:
                            drop = drop_item(env, info3)
                            if drop:
                                step_actions(env, drop)
                                all_acts.extend(drop)
                    else:
                        # Can't reach door even with key - maybe blocked
                        # Drop key, move blocker, re-get key
                        info_b = get_grid_info(env)
                        drop = drop_item(env, info_b)
                        if drop:
                            ok, r, n = step_actions(env, drop)
                            all_acts.extend(drop[:n])
                        blk_acts, _ = move_blocker_near_door(env, get_grid_info(env), ld['pos'])
                        all_acts.extend(blk_acts)
                        # Re-get key and try again
                        key_acts2, got2 = ensure_have_key(env, ld['color'])
                        all_acts.extend(key_acts2)
                        if got2:
                            info4 = get_grid_info(env)
                            face2 = bfs_face_target(info4['grid'], info4['pos'], info4['dir'], ld['pos'])
                            if face2:
                                ok, r, n = step_actions(env, face2 + ['toggle', 'forward'])
                                all_acts.extend((face2 + ['toggle', 'forward'])[:n])
                                if ok:
                                    return all_acts, True
                                info5 = get_grid_info(env)
                                if info5['carrying']:
                                    drop = drop_item(env, info5)
                                    if drop:
                                        step_actions(env, drop)
                                        all_acts.extend(drop)
                    break
            continue

        break

    return all_acts, False


def solve_putnext(env, info, sg):
    c1, t1 = sg.get('c1'), sg.get('t1')
    c2, t2 = sg.get('c2'), sg.get('t2')
    all_acts = []

    # Drop if carrying
    if info['carrying']:
        drop = drop_item(env, info)
        if drop:
            ok, r, n = step_actions(env, drop)
            all_acts.extend(drop[:n])
            if ok:
                return all_acts, True
            info = get_grid_info(env)

    # Find and pick up obj1
    def find_obj1(i):
        objs = find_objects(i['grid'], obj_type=t1, color=c1)
        if not objs and c1:
            objs = find_objects(i['grid'], obj_type=t1)
        for o in objs:
            a = bfs_face_target(i['grid'], i['pos'], i['dir'], o['pos'])
            if a is not None:
                return a + ['pickup']
        return None

    r1 = find_obj1(info)
    if r1:
        ok, r, n = step_actions(env, r1)
        all_acts.extend(r1[:n])
        if ok:
            return all_acts, True
    else:
        nav, ok = navigate_opening_doors(env, find_obj1)
        all_acts.extend(nav)
        if ok:
            return all_acts, True
        if not nav:
            return all_acts, False

    # Now drop near obj2
    info2 = get_grid_info(env)

    def find_drop_spot(i):
        objs2 = find_objects(i['grid'], obj_type=t2, color=c2)
        if not objs2 and c2:
            objs2 = find_objects(i['grid'], obj_type=t2)
        if not objs2:
            return None
        obj2 = objs2[0]
        tx, ty = obj2['pos']

        best = None
        for _, (adx, ady) in DIR_DELTAS.items():
            ex, ey = tx + adx, ty + ady
            if not (0 <= ex < i['w'] and 0 <= ey < i['h']):
                continue
            if i['grid'].get(ex, ey) is not None:
                continue
            for fd, (fdx, fdy) in DIR_DELTAS.items():
                sx, sy = ex - fdx, ey - fdy
                if not (0 <= sx < i['w'] and 0 <= sy < i['h']):
                    continue
                sc = i['grid'].get(sx, sy)
                if sc is not None and not (sc.type == 'door' and sc.is_open):
                    continue
                acts = bfs_to(i['grid'], i['pos'], i['dir'], (sx, sy), fd)
                if acts is not None and (best is None or len(acts) < len(best)):
                    best = acts
        return best + ['drop'] if best else None

    drop_acts = find_drop_spot(info2)
    if drop_acts:
        ok, r, n = step_actions(env, drop_acts)
        all_acts.extend(drop_acts[:n])
        return all_acts, ok

    nav, ok = navigate_opening_doors(env, find_drop_spot)
    all_acts.extend(nav)
    return all_acts, ok


# ─── Main solver entry point ────────────────────────────────────────────────

def solve_with_env(env):
    """Solve a pre-reset env. Returns (success, actions)."""
    obs = env.unwrapped.gen_obs()
    mission = obs['mission']
    subgoals = parse_mission(mission)

    all_actions = []
    success = False

    for sg in subgoals:
        info = get_grid_info(env)
        action = sg.get('action')

        if action == 'goto':
            acts, ok = solve_goto(env, info, sg)
        elif action == 'open':
            acts, ok = solve_open(env, info, sg)
        elif action == 'open_rel':
            acts, ok = solve_open_rel(env, info, sg)
        elif action == 'pickup':
            acts, ok = solve_pickup(env, info, sg)
        elif action == 'putnext':
            acts, ok = solve_putnext(env, info, sg)
        else:
            acts, ok = [], False

        all_actions.extend(acts)
        if ok:
            success = True
            break
        if not acts:
            break

    return success, all_actions


# Also provide the simpler interface
def solve(env_name, seed):
    """Solve and return action list."""
    env = gym.make(env_name)
    env.reset(seed=seed)
    success, actions = solve_with_env(env)
    env.close()
    return actions
