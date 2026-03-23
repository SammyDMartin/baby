"""BabyAI solver: parses missions, plans and executes action sequences.
Handles standard levels, multi-room navigation, chained key dependencies,
blocked doors, and compound missions."""
import re
import gymnasium as gym
from engine.grid import get_grid_info, find_objects, grid_to_dict, ACTION_MAP, DIR_DELTAS
from solvers.pathfinding import (
    bfs_path, bfs_to_face, find_room_doors, relative_dir_to_door,
    reachable_cells, walkable,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _execute(env, actions):
    """Execute actions on env, return result dict with replay data."""
    total_r = 0
    executed = []
    steps_data = []
    for a in actions:
        grid_before = grid_to_dict(env)
        obs, r, done, trunc, _ = env.step(ACTION_MAP[a])
        total_r += r
        executed.append(a)
        steps_data.append({'action': a, 'grid': grid_before})
        if done or trunc:
            return {'actions': executed, 'done': done, 'reward': total_r, 'steps': steps_data}
    return {'actions': executed, 'done': False, 'reward': total_r, 'steps': steps_data}


def _merge(*results):
    """Merge multiple result dicts sequentially."""
    merged = {'actions': [], 'done': False, 'reward': 0, 'steps': []}
    for r in results:
        if r is None:
            continue
        merged['actions'].extend(r['actions'])
        merged['steps'].extend(r['steps'])
        merged['reward'] += r['reward']
        merged['done'] = r['done']
    return merged


def _drop_carried(env, info):
    """Drop carried item in an empty adjacent cell."""
    if not info['carrying']:
        return []
    pos, d = info['pos'], info['dir']
    w, h = info['w'], info['h']
    grid = info['grid']

    for offset in [2, 3, 1, 0]:  # behind, left, right, forward
        try_dir = (d + offset) % 4
        dx, dy = DIR_DELTAS[try_dir]
        tx, ty = pos[0] + dx, pos[1] + dy
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


# ── mission parsing ──────────────────────────────────────────────────────────

COLORS = {'red', 'green', 'blue', 'purple', 'yellow', 'grey'}
OBJ_TYPES = {'ball', 'key', 'box', 'door'}


def parse_mission(mission):
    """Parse BabyAI mission into list of subgoals."""
    mission = mission.strip().lower()

    # "X, then Y" or "X then Y"
    for sep in [', then ', ' then ']:
        if sep in mission:
            parts = mission.split(sep, 1)
            return [_parse_single(parts[0].strip()), _parse_single(parts[1].strip())]

    # "X and Y" - split on "and" only if both sides parse as valid actions
    if ' and ' in mission:
        # Try splitting at each occurrence of " and "
        idx = 0
        while True:
            pos = mission.find(' and ', idx)
            if pos == -1:
                break
            left = mission[:pos].strip()
            right = mission[pos + 5:].strip()
            p1 = _parse_single(left)
            p2 = _parse_single(right)
            if p1['action'] != 'unknown' and p2['action'] != 'unknown':
                return [p1, p2]
            idx = pos + 5

    return [_parse_single(mission)]


def _parse_single(text):
    text = text.strip()

    m = re.match(r'put (?:the |a )?(\w+) (\w+) next to (?:the |a )?(\w+) (\w+)', text)
    if m:
        return {'action': 'putnext', 'c1': m.group(1), 't1': m.group(2), 'c2': m.group(3), 't2': m.group(4)}

    m = re.match(r'go to (?:the |a )?(\w+) (\w+)', text)
    if m:
        return {'action': 'goto', 'color': m.group(1), 'type': m.group(2)}
    m = re.match(r'go to (?:the |a )?(\w+)', text)
    if m:
        return {'action': 'goto', 'color': None, 'type': m.group(1)}

    # "pick up the X on your left/right" or "pick up the X behind you"
    m = re.match(r'pick up (?:the |a )?(\w+) on your (\w+)', text)
    if m:
        return {'action': 'pickup', 'color': None, 'type': m.group(1), 'rel_dir': m.group(2)}
    m = re.match(r'pick up (?:the |a )?(\w+) behind you', text)
    if m:
        return {'action': 'pickup', 'color': None, 'type': m.group(1), 'rel_dir': 'behind'}

    # "pick up the X Y on your left" (color + type + relative)
    m = re.match(r'pick up (?:the |a )?(\w+) (\w+) on your (\w+)', text)
    if m:
        w1, w2, rel = m.group(1), m.group(2), m.group(3)
        if w2 in OBJ_TYPES and w1 in COLORS:
            return {'action': 'pickup', 'color': w1, 'type': w2, 'rel_dir': rel}

    # "pick up the X Y"
    m = re.match(r'pick up (?:the |a )?(\w+) (\w+)', text)
    if m:
        w1, w2 = m.group(1), m.group(2)
        if w2 in OBJ_TYPES and w1 in COLORS:
            return {'action': 'pickup', 'color': w1, 'type': w2}
        elif w1 in OBJ_TYPES:
            return {'action': 'pickup', 'color': None, 'type': w1}
        else:
            return {'action': 'pickup', 'color': None, 'type': w2 if w2 in OBJ_TYPES else w1}

    m = re.match(r'pick up (?:the |a )?(\w+)', text)
    if m:
        return {'action': 'pickup', 'color': None, 'type': m.group(1)}

    m = re.match(r'open (?:the |a )?door on your (\w+)', text)
    if m:
        return {'action': 'open_rel', 'rel_dir': m.group(1)}

    m = re.match(r'open (?:the |a )?(\w+) door', text)
    if m:
        color = m.group(1)
        if color not in ('the', 'a'):
            return {'action': 'open', 'color': color}

    m = re.match(r'open (?:the |a )?door', text)
    if m:
        return {'action': 'open', 'color': None}

    return {'action': 'unknown', 'raw': text}


# ── general navigation engine ────────────────────────────────────────────────

def _navigate_via_doors(env, info, final_fn, max_iters=80):
    """Iteratively open/unlock doors until final_fn(info) returns actions.
    Handles: closed doors, locked doors (if carrying key), and blocked doors."""
    all_results = []

    for _ in range(max_iters):
        info = get_grid_info(env)
        final_acts = final_fn(info)
        if final_acts is not None:
            r = _execute(env, final_acts)
            all_results.append(r)
            return _merge(*all_results)

        doors = find_room_doors(info['grid'], info['pos'])
        usable = [d for d in doors if not d['is_open'] and not d['is_locked']]

        carrying = info['carrying']
        if carrying and carrying.type == 'key':
            usable.extend([d for d in doors if d['is_locked'] and d['color'] == carrying.color])

        if not usable:
            return None

        best_a, best_d = None, None
        for d in usable:
            a = bfs_to_face(info['grid'], info['pos'], info['dir'], d['pos'])
            if a is not None and (best_a is None or len(a) < len(best_a)):
                best_a = a
                best_d = d

        if best_a is None:
            return None

        r = _execute(env, best_a + ['toggle'])
        all_results.append(r)
        if r['done']:
            return _merge(*all_results)

        r2 = _execute(env, ['forward'])
        all_results.append(r2)
        if r2['done']:
            return _merge(*all_results)

    return None


def _ensure_reach_target(env, info, target_pos, interact_actions=None):
    """Navigate to face target_pos, possibly through multiple doors/locked doors.
    If we encounter locked doors, try to find the key first.
    interact_actions: actions to perform once facing the target (e.g. ['toggle'], ['pickup'])"""
    if interact_actions is None:
        interact_actions = []

    # Direct path?
    acts = bfs_to_face(info['grid'], info['pos'], info['dir'], target_pos)
    if acts is not None:
        return _execute(env, acts + interact_actions)

    # Navigate via doors
    def check(i):
        a = bfs_to_face(i['grid'], i['pos'], i['dir'], target_pos)
        if a is not None:
            return a + interact_actions
        return None

    return _navigate_via_doors(env, info, check)


def _ensure_carrying_key(env, info, key_color, depth=0):
    """Ensure we're carrying a key of given color.
    Handles: key in current room, key behind closed doors, key behind locked doors."""
    if depth > 5:
        return None

    carrying = info['carrying']
    if carrying and carrying.type == 'key' and carrying.color == key_color:
        return {'actions': [], 'done': False, 'reward': 0, 'steps': []}

    # Drop what we're carrying if it's not the right key
    all_results = []
    if carrying:
        drop = _drop_carried(env, info)
        if drop:
            r = _execute(env, drop)
            all_results.append(r)
            if r['done']:
                return _merge(*all_results)
            info = get_grid_info(env)

    # Find the key
    keys = find_objects(info['grid'], obj_type='key', color=key_color)
    if not keys:
        return None

    for k in keys:
        # Try direct pickup
        acts = bfs_to_face(info['grid'], info['pos'], info['dir'], k['pos'])
        if acts is not None:
            r = _execute(env, acts + ['pickup'])
            all_results.append(r)
            return _merge(*all_results)

    # Key behind doors - try navigating
    def find_key(i, _kc=key_color):
        ks = find_objects(i['grid'], obj_type='key', color=_kc)
        for kk in ks:
            a = bfs_to_face(i['grid'], i['pos'], i['dir'], kk['pos'])
            if a is not None:
                return a + ['pickup']
        return None

    # Check if blocked by locked door that we need another key for
    doors = find_room_doors(info['grid'], info['pos'])
    locked = [d for d in doors if d['is_locked']]

    for ld in locked:
        # Check if key we want is behind this locked door
        # Try getting the key for this door first (recursive)
        r_key = _ensure_carrying_key(env, get_grid_info(env), ld['color'], depth + 1)
        if r_key is not None and not r_key['done']:
            all_results.append(r_key)
            # Now unlock this door
            info2 = get_grid_info(env)
            r_unlock = _ensure_reach_target(env, info2, ld['pos'], ['toggle', 'forward'])
            if r_unlock is not None:
                all_results.append(r_unlock)
                if r_unlock['done']:
                    return _merge(*all_results)
                # Now try to find our target key
                info3 = get_grid_info(env)
                # Drop the door key first
                if info3['carrying']:
                    drop = _drop_carried(env, info3)
                    if drop:
                        r_drop = _execute(env, drop)
                        all_results.append(r_drop)
                        if r_drop['done']:
                            return _merge(*all_results)

                r_find = _navigate_via_doors(env, get_grid_info(env), find_key)
                if r_find is not None:
                    all_results.append(r_find)
                    return _merge(*all_results)

    # Try through unlocked doors
    r = _navigate_via_doors(env, info, find_key)
    if r is not None:
        all_results.append(r)
        return _merge(*all_results)

    return None


def _unlock_and_enter(env, info, door_info):
    """Unlock a locked door and step through it. Handles blocked doors."""
    key_color = door_info['color']
    door_pos = door_info['pos']
    all_results = []

    # Get the key
    r_key = _ensure_carrying_key(env, info, key_color)
    if r_key is None:
        return None
    all_results.append(r_key)
    if r_key['done']:
        return _merge(*all_results)

    # Navigate to door and unlock
    info2 = get_grid_info(env)
    r_unlock = _ensure_reach_target(env, info2, door_pos, ['toggle', 'forward'])
    if r_unlock is not None:
        all_results.append(r_unlock)
        return _merge(*all_results)

    # Door might be blocked by an object. Drop key, clear blocker, pick key back up.
    info2 = get_grid_info(env)
    drop = _drop_carried(env, info2)
    if not drop:
        return None
    r_drop = _execute(env, drop)
    all_results.append(r_drop)
    if r_drop['done']:
        return _merge(*all_results)

    # Move blocker
    info3 = get_grid_info(env)
    r_blocker = _move_blocker(env, info3, door_pos)
    if r_blocker is None:
        return None
    all_results.append(r_blocker)
    if r_blocker['done']:
        return _merge(*all_results)

    # Pick key back up
    info4 = get_grid_info(env)
    r_rekey = _ensure_carrying_key(env, info4, key_color)
    if r_rekey is None:
        return None
    all_results.append(r_rekey)
    if r_rekey['done']:
        return _merge(*all_results)

    # Now try unlock again
    info5 = get_grid_info(env)
    r_unlock2 = _ensure_reach_target(env, info5, door_pos, ['toggle', 'forward'])
    if r_unlock2 is None:
        return None
    all_results.append(r_unlock2)
    return _merge(*all_results)


def _move_blocker(env, info, door_pos):
    """If there's an object blocking a door, move it out of the way."""
    # Check cells adjacent to door for movable objects
    grid = info['grid']
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        bx, by = door_pos[0] + dx, door_pos[1] + dy
        if 0 <= bx < info['w'] and 0 <= by < info['h']:
            cell = grid.get(bx, by)
            if cell and cell.type in ('ball', 'key', 'box'):
                # This could be blocking - try to pick it up and move it
                acts = bfs_to_face(info['grid'], info['pos'], info['dir'], (bx, by))
                if acts is not None:
                    all_results = []
                    # Drop anything we're carrying first
                    if info['carrying']:
                        drop = _drop_carried(env, info)
                        if drop:
                            r = _execute(env, drop)
                            all_results.append(r)
                            info = get_grid_info(env)
                            acts = bfs_to_face(info['grid'], info['pos'], info['dir'], (bx, by))
                            if acts is None:
                                continue

                    r = _execute(env, acts + ['pickup'])
                    all_results.append(r)
                    if r['done']:
                        return _merge(*all_results)
                    # Drop it somewhere else
                    info2 = get_grid_info(env)
                    drop = _drop_carried(env, info2)
                    if drop:
                        r2 = _execute(env, drop)
                        all_results.append(r2)
                    return _merge(*all_results)
    return None


# ── subgoal solvers ──────────────────────────────────────────────────────────

def _solve_goto(env, info, sg):
    color, obj_type = sg.get('color'), sg.get('type')
    targets = find_objects(info['grid'], obj_type=obj_type, color=color)
    if not targets and color:
        targets = find_objects(info['grid'], obj_type=obj_type)

    for t in targets:
        acts = bfs_to_face(info['grid'], info['pos'], info['dir'], t['pos'])
        if acts is not None:
            return _execute(env, acts)

    def check(i):
        ts = find_objects(i['grid'], obj_type=obj_type, color=color)
        if not ts and color:
            ts = find_objects(i['grid'], obj_type=obj_type)
        for t in ts:
            a = bfs_to_face(i['grid'], i['pos'], i['dir'], t['pos'])
            if a is not None:
                return a
        return None

    return _navigate_via_doors(env, info, check)


def _solve_open(env, info, sg):
    color = sg.get('color')
    doors = find_objects(info['grid'], obj_type='door', color=color)
    if not doors:
        doors = find_objects(info['grid'], obj_type='door')
    if not doors:
        return None
    target = doors[0]

    if target.get('is_locked'):
        all_results = []
        r = _ensure_carrying_key(env, info, target['color'])
        if r is None:
            return None
        all_results.append(r)
        if r['done']:
            return _merge(*all_results)
        info2 = get_grid_info(env)
        r2 = _ensure_reach_target(env, info2, target['pos'], ['toggle'])
        if r2:
            all_results.append(r2)
        return _merge(*all_results)

    acts = bfs_to_face(info['grid'], info['pos'], info['dir'], target['pos'])
    if acts is not None:
        return _execute(env, acts + ['toggle'])

    def check(i):
        a = bfs_to_face(i['grid'], i['pos'], i['dir'], target['pos'])
        return a + ['toggle'] if a is not None else None

    return _navigate_via_doors(env, info, check)


def _solve_open_rel(env, info, sg):
    doors = find_room_doors(info['grid'], info['pos'])
    rel = sg.get('rel_dir')

    for d in doors:
        if relative_dir_to_door(info['pos'], info['dir'], d['pos']) == rel:
            acts = bfs_to_face(info['grid'], info['pos'], info['dir'], d['pos'])
            if acts is not None:
                return _execute(env, acts + ['toggle'])

    for d in doors:
        if not d['is_open'] and not d['is_locked']:
            acts = bfs_to_face(info['grid'], info['pos'], info['dir'], d['pos'])
            if acts is not None:
                return _execute(env, acts + ['toggle'])
    return None


def _filter_by_rel_dir(targets, agent_pos, agent_dir, rel_dir):
    """Filter objects by relative direction from agent.
    Uses soft matching: picks objects with the strongest component in the given direction."""
    if not rel_dir:
        return targets

    # Compute relative vector for each target and score by direction
    fdx, fdy = DIR_DELTAS[agent_dir]
    scored = []
    for t in targets:
        dx = t['pos'][0] - agent_pos[0]
        dy = t['pos'][1] - agent_pos[1]
        cross = fdx * dy - fdy * dx  # positive = right, negative = left
        dot = fdx * dx + fdy * dy    # positive = front, negative = behind

        if rel_dir == 'left':
            scored.append((cross, t))  # most negative cross = most left
        elif rel_dir == 'right':
            scored.append((-cross, t))  # most positive cross = most right
        elif rel_dir == 'front':
            scored.append((-dot, t))
        elif rel_dir == 'behind':
            scored.append((dot, t))
        else:
            scored.append((0, t))

    if not scored:
        return targets

    # Sort: lowest score = best match for the direction
    scored.sort(key=lambda x: x[0])
    return [s[1] for s in scored]


def _solve_pickup(env, info, sg):
    """Pick up an object, handling: locked doors, chained keys, blockers."""
    color, obj_type = sg.get('color'), sg.get('type')
    rel_dir = sg.get('rel_dir')
    all_results = []

    # Drop anything we're carrying
    if info['carrying']:
        drop = _drop_carried(env, info)
        if drop:
            r = _execute(env, drop)
            all_results.append(r)
            if r['done']:
                return _merge(*all_results)
            info = get_grid_info(env)

    # Try direct pickup
    targets = find_objects(info['grid'], obj_type=obj_type, color=color)
    if not targets and color:
        targets = find_objects(info['grid'], obj_type=obj_type)
    targets = _filter_by_rel_dir(targets, info['pos'], info['dir'], rel_dir)

    for t in targets:
        acts = bfs_to_face(info['grid'], info['pos'], info['dir'], t['pos'])
        if acts is not None:
            r = _execute(env, acts + ['pickup'])
            all_results.append(r)
            return _merge(*all_results)

    # Need to go through doors
    # Check for locked doors and handle them with key chains
    for attempt in range(10):
        info = get_grid_info(env)

        # Drop anything we're carrying (from previous key usage)
        if info['carrying']:
            drop = _drop_carried(env, info)
            if drop:
                r = _execute(env, drop)
                all_results.append(r)
                if r['done']:
                    return _merge(*all_results)
                info = get_grid_info(env)

        # Check if target is now reachable
        targets = find_objects(info['grid'], obj_type=obj_type, color=color)
        if not targets and color:
            targets = find_objects(info['grid'], obj_type=obj_type)
        for t in targets:
            acts = bfs_to_face(info['grid'], info['pos'], info['dir'], t['pos'])
            if acts is not None:
                r = _execute(env, acts + ['pickup'])
                all_results.append(r)
                return _merge(*all_results)

        # Find doors we can open
        doors = find_room_doors(info['grid'], info['pos'])
        unlocked_closed = [d for d in doors if not d['is_open'] and not d['is_locked']]
        locked = [d for d in doors if d['is_locked']]

        if unlocked_closed:
            # Open nearest unlocked closed door
            best_a, best_d = None, None
            for d in unlocked_closed:
                a = bfs_to_face(info['grid'], info['pos'], info['dir'], d['pos'])
                if a is not None and (best_a is None or len(a) < len(best_a)):
                    best_a = a
                    best_d = d

            if best_a is not None:
                # Check if door is blocked
                r = _execute(env, best_a + ['toggle'])
                all_results.append(r)
                if r['done']:
                    return _merge(*all_results)

                # Check if toggle worked (door opened)
                info_after = get_grid_info(env)
                cell_at_door = info_after['grid'].get(best_d['pos'][0], best_d['pos'][1])
                if cell_at_door and cell_at_door.type == 'door' and not cell_at_door.is_open:
                    # Door didn't open - might be blocked
                    r_blocker = _move_blocker(env, get_grid_info(env), best_d['pos'])
                    if r_blocker:
                        all_results.append(r_blocker)
                        # Try again
                        info = get_grid_info(env)
                        acts = bfs_to_face(info['grid'], info['pos'], info['dir'], best_d['pos'])
                        if acts:
                            r = _execute(env, acts + ['toggle'])
                            all_results.append(r)
                            if r['done']:
                                return _merge(*all_results)

                r2 = _execute(env, ['forward'])
                all_results.append(r2)
                if r2['done']:
                    return _merge(*all_results)
                continue

        if locked:
            # Try to unlock a locked door
            for ld in locked:
                r_unlock = _unlock_and_enter(env, get_grid_info(env), ld)
                if r_unlock is not None:
                    all_results.append(r_unlock)
                    if r_unlock['done']:
                        return _merge(*all_results)
                    break
            else:
                return None  # Can't unlock any door
            continue

        # No doors to open
        # Try general navigation with what we have
        def check_pickup(i, _ot=obj_type, _c=color):
            ts = find_objects(i['grid'], obj_type=_ot, color=_c)
            if not ts and _c:
                ts = find_objects(i['grid'], obj_type=_ot)
            for t in ts:
                a = bfs_to_face(i['grid'], i['pos'], i['dir'], t['pos'])
                if a is not None:
                    return a + ['pickup']
            return None

        r = _navigate_via_doors(env, info, check_pickup)
        if r:
            all_results.append(r)
        return _merge(*all_results) if all_results else None

    return _merge(*all_results) if all_results else None


def _solve_putnext(env, info, sg):
    c1, t1 = sg.get('c1'), sg.get('t1')
    c2, t2 = sg.get('c2'), sg.get('t2')
    all_results = []

    # Drop anything we're carrying first
    if info['carrying']:
        drop = _drop_carried(env, info)
        if drop:
            r = _execute(env, drop)
            all_results.append(r)
            if r['done']:
                return _merge(*all_results)
            info = get_grid_info(env)

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
        def find_obj(i, _t=t1, _c=c1):
            os = find_objects(i['grid'], obj_type=_t, color=_c)
            if not os and _c:
                os = find_objects(i['grid'], obj_type=_t)
            for o in os:
                a = bfs_to_face(i['grid'], i['pos'], i['dir'], o['pos'])
                if a is not None:
                    return a + ['pickup']
            return None
        r1 = _navigate_via_doors(env, info, find_obj)
        if r1 is None:
            return None
        all_results.append(r1)
        if r1['done']:
            return _merge(*all_results)
    else:
        r1 = _execute(env, best_pick + ['pickup'])
        all_results.append(r1)
        if r1['done']:
            return _merge(*all_results)

    # Drop near obj2
    info2 = get_grid_info(env)

    def check_drop(i, _t2=t2, _c2=c2):
        return _find_drop_position(i, _t2, _c2)

    drop_acts = check_drop(info2)
    if drop_acts:
        r2 = _execute(env, drop_acts)
        all_results.append(r2)
        return _merge(*all_results)

    r2 = _navigate_via_doors(env, info2, check_drop)
    if r2:
        all_results.append(r2)
    return _merge(*all_results) if all_results else None


def _find_drop_position(info, t2, c2):
    """Find actions to drop carried obj next to obj2."""
    objs2 = find_objects(info['grid'], obj_type=t2, color=c2)
    if not objs2 and c2:
        objs2 = find_objects(info['grid'], obj_type=t2)
    if not objs2:
        return None
    obj2 = objs2[0]
    tx, ty = obj2['pos']

    best = None
    for _, (adx, ady) in DIR_DELTAS.items():
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


# ── main entry point ─────────────────────────────────────────────────────────

def solve_level(env_name, seed=42, max_steps=None):
    """Solve a BabyAI level. Returns result dict with full replay data."""
    kwargs = {}
    if max_steps is not None:
        kwargs['max_steps'] = max_steps
    env = gym.make(env_name, **kwargs)
    obs, _ = env.reset(seed=seed)
    mission = obs['mission']
    initial_grid = grid_to_dict(env)

    subgoals = parse_mission(mission)
    all_actions = []
    all_steps = []
    total_reward = 0
    success = False

    for sg in subgoals:
        info = get_grid_info(env)
        action = sg.get('action', 'unknown')

        if action == 'goto':
            result = _solve_goto(env, info, sg)
        elif action == 'open':
            result = _solve_open(env, info, sg)
        elif action == 'open_rel':
            result = _solve_open_rel(env, info, sg)
        elif action == 'pickup':
            result = _solve_pickup(env, info, sg)
        elif action == 'putnext':
            result = _solve_putnext(env, info, sg)
        else:
            result = None

        if result is None:
            break

        all_actions.extend(result['actions'])
        all_steps.extend(result['steps'])
        total_reward += result['reward']
        if result['done']:
            success = total_reward > 0
            break

    final_grid = grid_to_dict(env)
    env.close()

    return {
        'env': env_name,
        'seed': seed,
        'mission': mission,
        'subgoals': subgoals,
        'actions': all_actions,
        'num_steps': len(all_actions),
        'success': success,
        'reward': total_reward,
        'initial_grid': initial_grid,
        'final_grid': final_grid,
        'steps': all_steps,
    }
