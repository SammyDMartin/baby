"""BFS pathfinding for BabyAI gridworlds."""
from collections import deque
from engine.grid import DIR_DELTAS, ACTION_MAP


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
        if len(acts) > 300:
            continue

        for new_d, turn in [((d-1) % 4, 'left'), ((d+1) % 4, 'right')]:
            s = (x, y, new_d)
            if s not in visited:
                visited.add(s)
                queue.append((x, y, new_d, acts + [turn]))

        dx, dy = DIR_DELTAS[d]
        nx, ny = x + dx, y + dy
        if walkable(grid, nx, ny):
            s = (nx, ny, d)
            if s not in visited:
                visited.add(s)
                queue.append((nx, ny, d, acts + ['forward']))
    return None


def bfs_to_face(grid, start_pos, start_dir, target_pos):
    """Navigate to be adjacent to target, facing it."""
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
    """Flood fill from start."""
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited and walkable(grid, nx, ny):
                visited.add((nx, ny))
                queue.append((nx, ny))
    return visited


def find_room_doors(grid, start):
    """Find all doors bordering reachable area from start."""
    reachable = reachable_cells(grid, start)
    doors = []
    seen = set()
    for x, y in reachable:
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in seen:
                continue
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                cell = grid.get(nx, ny)
                if cell and cell.type == 'door' and (nx, ny) not in reachable:
                    seen.add((nx, ny))
                    doors.append({
                        'pos': (nx, ny), 'cell': cell,
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
