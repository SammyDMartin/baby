"""ChildAI Public API — constants and helpers for solver authors.

This is the ONLY source file you may read (besides your own solver).
Import from here for constants, observation decoding, and coordinate helpers.

Do NOT read engine/, solvers/, challenge.py, or any other .py file.
"""
import numpy as np


# ── Actions ─────────────────────────────────────────────────────────────

ACTION_MAP = {
    'left': 0,      # turn left (rotate 90° counterclockwise)
    'right': 1,     # turn right (rotate 90° clockwise)
    'forward': 2,   # move one cell in facing direction
    'pickup': 3,    # pick up object in front (can only carry one)
    'drop': 4,      # drop carried object in front
    'toggle': 5,    # open/close/unlock door in front
}
ACTION_NAMES = {v: k for k, v in ACTION_MAP.items()}


# ── Directions ──────────────────────────────────────────────────────────
# Direction 0=right(+x), 1=down(+y), 2=left(-x), 3=up(-y)
# Turn left = (dir - 1) % 4,  turn right = (dir + 1) % 4

DIR_DELTAS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
DIR_NAMES = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}


# ── Observation Encoding (for fog mode 7x7x3 arrays) ───────────────────

OBJ_TYPES = {
    0: 'unseen', 1: 'empty', 2: 'wall', 3: 'floor', 4: 'door',
    5: 'key', 6: 'ball', 7: 'box', 8: 'goal', 9: 'lava',
}
OBJ_TO_IDX = {v: k for k, v in OBJ_TYPES.items()}

COLORS = {
    0: 'red', 1: 'green', 2: 'blue', 3: 'purple', 4: 'yellow', 5: 'grey',
}
COLOR_TO_IDX = {v: k for k, v in COLORS.items()}

DOOR_STATES = {0: 'open', 1: 'closed', 2: 'locked'}


# ── Fog Observation Helpers ─────────────────────────────────────────────

def decode_obs(obs_image):
    """Decode a MiniGrid partial observation into readable objects.

    The 7x7x3 obs_image is agent-relative:
      - Agent is at view position (3, 6), facing toward row 0
      - Each cell is (object_type_idx, color_idx, state)

    Args:
        obs_image: (H, W, 3) numpy array from obs['image']

    Returns:
        List of dicts for non-trivial visible cells:
        [{'type', 'color', 'view_x', 'view_y', 'rel_x', 'rel_y', ...}, ...]

        rel_x: left/right offset (-3 to +3, negative = agent's left)
        rel_y: forward distance (0 = agent's row, 6 = farthest visible)
    """
    h, w = obs_image.shape[:2]
    agent_vx = w // 2
    agent_vy = h - 1
    objects = []
    for vy in range(h):
        for vx in range(w):
            obj_idx = int(obs_image[vy, vx, 0])
            color_idx = int(obs_image[vy, vx, 1])
            state = int(obs_image[vy, vx, 2])
            if obj_idx in (0, 1, 2):  # unseen, empty, wall
                continue
            entry = {
                'type': OBJ_TYPES.get(obj_idx, 'unknown'),
                'color': COLORS.get(color_idx, 'unknown'),
                'view_x': vx, 'view_y': vy,
                'rel_x': vx - agent_vx,
                'rel_y': agent_vy - vy,
            }
            if obj_idx == 4:  # door
                entry['door_state'] = DOOR_STATES.get(state, 'unknown')
            objects.append(entry)
    return objects


def view_to_world(rel_x, rel_y, agent_x, agent_y, agent_dir):
    """Convert agent-relative coordinates to world coordinates.

    In the agent's frame:
      rel_x: right is positive, left is negative
      rel_y: forward is positive

    Args:
        rel_x, rel_y: agent-relative offset (from decode_obs)
        agent_x, agent_y: agent's world position (if you're tracking it)
        agent_dir: agent's absolute direction (0=right, 1=down, 2=left, 3=up)

    Returns:
        (world_x, world_y) integer tuple
    """
    # Forward and right vectors for each direction
    #   dir 0 (right):  forward=(+1, 0), right=( 0,+1)
    #   dir 1 (down):   forward=( 0,+1), right=(-1, 0)
    #   dir 2 (left):   forward=(-1, 0), right=( 0,-1)
    #   dir 3 (up):     forward=( 0,-1), right=(+1, 0)
    fwd = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    rgt = [(0, 1), (-1, 0), (0, -1), (1, 0)]

    fx, fy = fwd[agent_dir]
    rx, ry = rgt[agent_dir]

    world_x = agent_x + rel_y * fx + rel_x * rx
    world_y = agent_y + rel_y * fy + rel_x * ry
    return (world_x, world_y)
