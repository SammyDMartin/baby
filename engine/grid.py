"""Grid rendering and state extraction for BabyAI environments."""
import gymnasium as gym

ACTION_MAP = {'left': 0, 'right': 1, 'forward': 2, 'pickup': 3, 'drop': 4, 'toggle': 5, 'done': 6}
DIR_DELTAS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
DIR_NAMES = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
DIR_ARROWS = {0: '>', 1: 'v', 2: '<', 3: '^'}

COLOR_MAP = {
    'red': '#e74c3c', 'green': '#2ecc71', 'blue': '#3498db',
    'purple': '#9b59b6', 'yellow': '#f1c40f', 'grey': '#95a5a6',
}


def get_grid_info(env):
    """Extract full grid state from environment."""
    grid = env.unwrapped.grid
    return {
        'grid': grid,
        'pos': tuple(env.unwrapped.agent_pos),
        'dir': env.unwrapped.agent_dir,
        'carrying': env.unwrapped.carrying,
        'w': grid.width, 'h': grid.height,
    }


def find_objects(grid, obj_type=None, color=None):
    """Find all objects matching criteria."""
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


def render_grid(env):
    """Render grid as ASCII text."""
    grid = env.unwrapped.grid
    agent_pos = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir
    w, h = grid.width, grid.height

    lines = []
    lines.append(f"Grid: {w}x{h} | Agent: ({agent_pos[0]},{agent_pos[1]}) facing {DIR_NAMES[agent_dir]}")

    carrying = env.unwrapped.carrying
    if carrying:
        lines.append(f"Carrying: {carrying.color} {carrying.type}")

    header = "   " + "".join(f"{x:>4}" for x in range(w))
    lines.append(header)

    for y in range(h):
        row = f"{y:2d} "
        for x in range(w):
            if (x, y) == tuple(agent_pos):
                row += f"  A{DIR_ARROWS[agent_dir]}"
            else:
                cell = grid.get(x, y)
                if cell is None:
                    row += "   ."
                elif cell.type == 'wall':
                    row += "  ##"
                elif cell.type == 'door':
                    c = cell.color[0].upper()
                    s = 'O' if cell.is_open else ('L' if cell.is_locked else 'C')
                    row += f" {c}D{s}"
                elif cell.type == 'key':
                    row += f" {cell.color[0].upper()}Ky"
                elif cell.type == 'ball':
                    row += f" {cell.color[0].upper()}Bl"
                elif cell.type == 'box':
                    row += f" {cell.color[0].upper()}Bx"
                elif cell.type == 'goal':
                    row += "  GL"
                elif cell.type == 'lava':
                    row += "  LA"
                else:
                    row += f"  {cell.type[:2]}"
        lines.append(row)
    return "\n".join(lines)


def grid_to_dict(env):
    """Convert grid state to a JSON-serializable dictionary for web rendering."""
    grid = env.unwrapped.grid
    agent_pos = tuple(env.unwrapped.agent_pos)
    agent_dir = env.unwrapped.agent_dir
    w, h = grid.width, grid.height

    cells = []
    for y in range(h):
        for x in range(w):
            cell = grid.get(x, y)
            if cell is None:
                cells.append({'x': x, 'y': y, 'type': 'empty'})
            else:
                entry = {
                    'x': x, 'y': y, 'type': cell.type,
                    'color': getattr(cell, 'color', None),
                }
                if cell.type == 'door':
                    entry['is_open'] = cell.is_open
                    entry['is_locked'] = cell.is_locked
                cells.append(entry)

    carrying = env.unwrapped.carrying
    carry_info = None
    if carrying:
        carry_info = {'type': carrying.type, 'color': carrying.color}

    return {
        'width': w, 'height': h,
        'cells': cells,
        'agent': {'x': agent_pos[0], 'y': agent_pos[1], 'dir': agent_dir, 'dir_name': DIR_NAMES[agent_dir]},
        'carrying': carry_info,
    }
