"""BabyAI solver: parses missions, plans and executes action sequences."""
import re
import gymnasium as gym
from engine.grid import get_grid_info, find_objects, grid_to_dict, ACTION_MAP, DIR_DELTAS
from engine.pathfinding import bfs_path, bfs_to_face, find_room_doors, relative_dir_to_door


# ── helpers ──────────────────────────────────────────────────────────────────

def _execute(env, actions):
    """Execute actions on env, return {actions, done, reward, steps_data}."""
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


def _drop_carried(env, info):
    """Drop carried item. Returns action list or None."""
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

    # "X and Y" (but not "pick up X and Y" which is ambiguous)
    if ' and ' in mission and not mission.startswith('pick up'):
        # Check it's truly compound (both sides have verbs)
        parts = mission.split(' and ', 1)
        p1 = _parse_single(parts[0].strip())
        p2 = _parse_single(parts[1].strip())
        if p1['action'] != 'unknown' and p2['action'] != 'unknown':
            return [p1, p2]

    return [_parse_single(mission)]


def _parse_single(text):
    text = text.strip()

    # "put the X Y next to the Z W"
    m = re.match(r'put (?:the |a )?(\w+) (\w+) next to (?:the |a )?(\w+) (\w+)', text)
    if m:
        return {'action': 'putnext', 'c1': m.group(1), 't1': m.group(2), 'c2': m.group(3), 't2': m.group(4)}

    # "go to the X Y"
    m = re.match(r'go to (?:the |a )?(\w+) (\w+)', text)
    if m:
        return {'action': 'goto', 'color': m.group(1), 'type': m.group(2)}

    m = re.match(r'go to (?:the |a )?(\w+)', text)
    if m:
        return {'action': 'goto', 'color': None, 'type': m.group(1)}

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

    # "open a door on your X"
    m = re.match(r'open (?:the |a )?door on your (\w+)', text)
    if m:
        return {'action': 'open_rel', 'rel_dir': m.group(1)}

    # "open the X door"
    m = re.match(r'open (?:the |a )?(\w+) door', text)
    if m:
        color = m.group(1)
        if color not in ('the', 'a'):
            return {'action': 'open', 'color': color}

    m = re.match(r'open (?:the |a )?door', text)
    if m:
        return {'action': 'open', 'color': None}

    return {'action': 'unknown', 'raw': text}


# ── door navigation ──────────────────────────────────────────────────────────

def _navigate_via_doors(env, info, final_fn, max_doors=20):
    """Iteratively open doors until final_fn(info) returns actions."""
    all_actions = []
    all_steps = []
    total_reward = 0

    for _ in range(max_doors):
        info = get_grid_info(env)
        final_acts = final_fn(info)
        if final_acts is not None:
            r = _execute(env, final_acts)
            all_actions.extend(r['actions'])
            all_steps.extend(r['steps'])
            total_reward += r['reward']
            return {'actions': all_actions, 'done': r['done'], 'reward': total_reward, 'steps': all_steps}

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

        r = _execute(env, best_a + ['toggle'])
        all_actions.extend(r['actions'])
        all_steps.extend(r['steps'])
        total_reward += r['reward']
        if r['done']:
            return {'actions': all_actions, 'done': True, 'reward': total_reward, 'steps': all_steps}

        r2 = _execute(env, ['forward'])
        all_actions.extend(r2['actions'])
        all_steps.extend(r2['steps'])
        total_reward += r2['reward']
        if r2['done']:
            return {'actions': all_actions, 'done': True, 'reward': total_reward, 'steps': all_steps}

    return None


def _merge(r1, r2):
    """Merge two result dicts."""
    if r2 is None:
        return r1
    return {
        'actions': r1['actions'] + r2['actions'],
        'done': r2['done'],
        'reward': r1['reward'] + r2['reward'],
        'steps': r1['steps'] + r2['steps'],
    }


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
        return _solve_unlock_door(env, info, target)

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


def _solve_unlock_door(env, info, target):
    """Unlock a locked door: find key, pick up, navigate to door, toggle."""
    key_color = target['color']
    door_pos = target['pos']

    # Try direct key pickup
    keys = find_objects(info['grid'], obj_type='key', color=key_color)
    for k in keys:
        ka = bfs_to_face(info['grid'], info['pos'], info['dir'], k['pos'])
        if ka is not None:
            r1 = _execute(env, ka + ['pickup'])
            if r1['done']:
                return r1
            info2 = get_grid_info(env)
            da = bfs_to_face(info2['grid'], info2['pos'], info2['dir'], door_pos)
            if da is not None:
                r2 = _execute(env, da + ['toggle'])
                return _merge(r1, r2)
            # Door behind other doors
            def check_door(i):
                a = bfs_to_face(i['grid'], i['pos'], i['dir'], door_pos)
                return a + ['toggle'] if a is not None else None
            r2 = _navigate_via_doors(env, get_grid_info(env), check_door)
            if r2:
                return _merge(r1, r2)

    # Key behind doors
    def find_key(i):
        ks = find_objects(i['grid'], obj_type='key', color=key_color)
        for kk in ks:
            a = bfs_to_face(i['grid'], i['pos'], i['dir'], kk['pos'])
            if a is not None:
                return a + ['pickup']
        return None

    r1 = _navigate_via_doors(env, info, find_key)
    if r1 and not r1['done']:
        def check_door(i):
            a = bfs_to_face(i['grid'], i['pos'], i['dir'], door_pos)
            return a + ['toggle'] if a is not None else None
        r2 = _navigate_via_doors(env, get_grid_info(env), check_door)
        if r2:
            return _merge(r1, r2)

    return None


def _solve_pickup(env, info, sg):
    color, obj_type = sg.get('color'), sg.get('type')

    # If carrying something, drop it first
    if info['carrying']:
        drop = _drop_carried(env, info)
        if drop:
            r_drop = _execute(env, drop)
            if r_drop['done']:
                return r_drop
            info = get_grid_info(env)

    targets = find_objects(info['grid'], obj_type=obj_type, color=color)
    if not targets and color:
        targets = find_objects(info['grid'], obj_type=obj_type)

    for t in targets:
        acts = bfs_to_face(info['grid'], info['pos'], info['dir'], t['pos'])
        if acts is not None:
            return _execute(env, acts + ['pickup'])

    # Need to go through doors — possibly locked
    doors = find_room_doors(info['grid'], info['pos'])
    locked = [d for d in doors if d['is_locked']]

    if locked and not info['carrying']:
        for ld in locked:
            keys = find_objects(info['grid'], obj_type='key', color=ld['color'])

            # Try direct key path
            for k in keys:
                ka = bfs_to_face(info['grid'], info['pos'], info['dir'], k['pos'])
                if ka is not None:
                    r1 = _execute(env, ka + ['pickup'])
                    if r1['done']:
                        return r1
                    info2 = get_grid_info(env)
                    da = bfs_to_face(info2['grid'], info2['pos'], info2['dir'], ld['pos'])
                    if da is not None:
                        r2 = _execute(env, da + ['toggle', 'forward'])
                        r_so_far = _merge(r1, r2)
                        if r_so_far['done']:
                            return r_so_far

                        # Drop key before picking up target
                        info3 = get_grid_info(env)
                        drop_a = _drop_carried(env, info3)
                        if drop_a:
                            r_drop = _execute(env, drop_a)
                            r_so_far = _merge(r_so_far, r_drop)
                            if r_so_far['done']:
                                return r_so_far

                        # Find target
                        info4 = get_grid_info(env)

                        def check_target(i, _ot=obj_type, _c=color):
                            ts = find_objects(i['grid'], obj_type=_ot, color=_c)
                            if not ts and _c:
                                ts = find_objects(i['grid'], obj_type=_ot)
                            for t in ts:
                                a = bfs_to_face(i['grid'], i['pos'], i['dir'], t['pos'])
                                if a is not None:
                                    return a + ['pickup']
                            return None

                        r3 = _navigate_via_doors(env, info4, check_target)
                        if r3:
                            return _merge(r_so_far, r3)

            # Key behind closed doors
            if keys:
                key_color = ld['color']
                door_pos = ld['pos']

                def find_key(i, _kc=key_color):
                    ks = find_objects(i['grid'], obj_type='key', color=_kc)
                    for kk in ks:
                        a = bfs_to_face(i['grid'], i['pos'], i['dir'], kk['pos'])
                        if a is not None:
                            return a + ['pickup']
                    return None

                r_nav = _navigate_via_doors(env, info, find_key)
                if r_nav and not r_nav['done']:
                    info_k = get_grid_info(env)

                    def find_locked(i, _dp=door_pos):
                        a = bfs_to_face(i['grid'], i['pos'], i['dir'], _dp)
                        return a + ['toggle', 'forward'] if a is not None else None

                    r_door = _navigate_via_doors(env, info_k, find_locked)
                    if r_door:
                        r_so_far = _merge(r_nav, r_door)
                        info_d = get_grid_info(env)
                        drop_a = _drop_carried(env, info_d)
                        if drop_a:
                            r_drop = _execute(env, drop_a)
                            r_so_far = _merge(r_so_far, r_drop)

                        def check_t(i, _ot=obj_type, _c=color):
                            ts = find_objects(i['grid'], obj_type=_ot, color=_c)
                            if not ts and _c:
                                ts = find_objects(i['grid'], obj_type=_ot)
                            for t in ts:
                                a = bfs_to_face(i['grid'], i['pos'], i['dir'], t['pos'])
                                if a is not None:
                                    return a + ['pickup']
                            return None

                        r_pick = _navigate_via_doors(env, get_grid_info(env), check_t)
                        if r_pick:
                            return _merge(r_so_far, r_pick)

    def check_pickup(i, _ot=obj_type, _c=color):
        ts = find_objects(i['grid'], obj_type=_ot, color=_c)
        if not ts and _c:
            ts = find_objects(i['grid'], obj_type=_ot)
        for t in ts:
            a = bfs_to_face(i['grid'], i['pos'], i['dir'], t['pos'])
            if a is not None:
                return a + ['pickup']
        return None

    return _navigate_via_doors(env, info, check_pickup)


def _solve_putnext(env, info, sg):
    c1, t1 = sg.get('c1'), sg.get('t1')
    c2, t2 = sg.get('c2'), sg.get('t2')

    # Drop anything we're carrying first
    if info['carrying']:
        drop = _drop_carried(env, info)
        if drop:
            r_drop = _execute(env, drop)
            if r_drop['done']:
                return r_drop
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
        if r1['done']:
            return r1
    else:
        r1 = _execute(env, best_pick + ['pickup'])
        if r1['done']:
            return r1

    # Drop near obj2
    info2 = get_grid_info(env)

    def check_drop(i, _t2=t2, _c2=c2):
        return _find_drop_position(i, _t2, _c2)

    drop_acts = check_drop(info2)
    if drop_acts:
        r2 = _execute(env, drop_acts)
        return _merge(r1, r2)

    r2 = _navigate_via_doors(env, info2, check_drop)
    if r2:
        return _merge(r1, r2)
    return None


def _find_drop_position(info, t2, c2):
    """Find actions to drop carried obj next to obj2."""
    from engine.pathfinding import bfs_path
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

def solve_level(env_name, seed=42, max_steps=500):
    """Solve a BabyAI level. Returns a result dict with full replay data."""
    env = gym.make(env_name, max_steps=max_steps)
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
