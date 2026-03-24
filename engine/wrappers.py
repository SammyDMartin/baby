"""Environment wrappers for ChildAI challenge modes.

FogEnv: Enforces partial observability (7x7 agent-relative view).
OneWayDoorWrapper: Doors close and lock behind the agent.
Observation decoding helpers for working with partial obs.
"""
import gymnasium as gym
import numpy as np
from engine.grid import ACTION_MAP, DIR_DELTAS

# ── Observation encoding constants (from MiniGrid) ──────────────────────

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

INT_TO_ACTION = {v: k for k, v in ACTION_MAP.items()}


# ── FogEnv ──────────────────────────────────────────────────────────────

class FogEnv:
    """Restricted environment enforcing partial observability.

    The solver gets:
      - step(action) -> obs, reward, done, truncated, info
      - obs['image']:     7x7x3 partial view (agent-relative)
      - obs['direction']: agent's absolute direction (0=right,1=down,2=left,3=up)
      - obs['mission']:   mission text

    The solver does NOT get:
      - Full grid (env.unwrapped.grid)
      - Agent world position (env.unwrapped.agent_pos)
      - Carrying state (env.unwrapped.carrying)

    The view is agent-relative: agent is always at
    (view_size//2, view_size-1) facing toward row 0.
    For default view_size=7, agent is at (3, 6).

    Each cell in obs['image'] is (object_type_idx, color_idx, state).
    Use decode_obs() to get a readable list of visible objects.
    """

    def __init__(self, env):
        self._env = env
        self._mission = None
        self._actions = []  # record for verification
        self._done = False

    def step(self, action):
        """Take an action. Accepts int or action name string."""
        if self._done:
            raise RuntimeError("Episode is done. Cannot step further.")
        if isinstance(action, str):
            action_int = ACTION_MAP[action]
            action_name = action
        else:
            action_int = int(action)
            action_name = INT_TO_ACTION.get(action_int, str(action_int))

        self._actions.append(action_name)
        obs, reward, done, truncated, info = self._env.step(action_int)
        self._done = done or truncated
        return self._filter(obs), reward, done, truncated, {}

    def reset(self, **kwargs):
        """Reset the environment. Returns (obs, info)."""
        self._actions = []
        self._done = False
        obs, info = self._env.reset(**kwargs)
        self._mission = obs.get('mission', '')
        return self._filter(obs), {}

    def _filter(self, obs):
        return {
            'image': np.array(obs['image']),
            'direction': int(obs.get('direction', self._env.unwrapped.agent_dir)),
            'mission': self._mission or obs.get('mission', ''),
        }

    @property
    def mission(self):
        return self._mission

    @property
    def actions_taken(self):
        """Actions taken so far (string names), for verification."""
        return list(self._actions)

    def close(self):
        self._env.close()

    def __getattr__(self, name):
        blocked = {
            'unwrapped', 'grid', 'agent_pos', 'agent_dir', 'carrying',
            'gen_obs', 'step_count',
        }
        if name in blocked:
            raise AttributeError(
                f"FogEnv: access to '{name}' is blocked. "
                f"Use step()/reset() and obs['image']/obs['direction']/obs['mission'] only."
            )
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


# ── OneWayDoorWrapper ───────────────────────────────────────────────────

class OneWayDoorWrapper(gym.Wrapper):
    """Doors close and lock behind the agent after passing through.

    When the agent moves off a door cell, that door closes and locks.
    The agent cannot return the way it came. Forces forward-only planning.

    In MiniGrid, open doors are passable (agent stands on the door cell).
    This wrapper detects when the agent leaves a door cell and locks it.
    """

    def __init__(self, env):
        super().__init__(env)
        self._on_door = False
        self._door_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        pos = tuple(self.unwrapped.agent_pos)
        cell = self.unwrapped.grid.get(*pos)
        self._on_door = bool(cell and cell.type == 'door' and cell.is_open)
        self._door_pos = pos if self._on_door else None
        return obs, info

    def step(self, action):
        old_pos = tuple(self.unwrapped.agent_pos)
        was_on_door = self._on_door
        door_pos = self._door_pos

        obs, reward, done, truncated, info = self.env.step(action)

        new_pos = tuple(self.unwrapped.agent_pos)

        # If agent was on an open door and moved off, lock it behind them
        if was_on_door and door_pos and old_pos != new_pos:
            grid = self.unwrapped.grid
            door_cell = grid.get(*door_pos)
            if door_cell and door_cell.type == 'door' and door_cell.is_open:
                door_cell.is_open = False
                door_cell.is_locked = True

        # Track if agent is now on a door cell
        cell = self.unwrapped.grid.get(*new_pos)
        if cell and cell.type == 'door' and cell.is_open:
            self._on_door = True
            self._door_pos = new_pos
        else:
            self._on_door = False
            self._door_pos = None

        return obs, reward, done, truncated, info


# ── Observation decoding helpers ────────────────────────────────────────

def decode_obs(obs_image):
    """Decode a MiniGrid partial observation into readable objects.

    Args:
        obs_image: (H, W, 3) numpy array. Each cell = (obj_type, color, state).

    Returns:
        List of visible objects: [{'type', 'color', 'view_x', 'view_y',
        'rel_x', 'rel_y', 'state'}, ...]

        Coordinates:
          view_x, view_y: position in the 7x7 view grid
          rel_x: left/right offset from agent (-3 to +3, negative=left)
          rel_y: distance in front of agent (0=same row, 6=farthest visible)
          Agent is at view (3, 6), i.e. rel (0, 0).
    """
    h, w = obs_image.shape[:2]
    agent_vx = w // 2
    agent_vy = h - 1
    objects = []
    for vy in range(h):
        for vx in range(w):
            obj_idx, color_idx, state = int(obs_image[vy, vx, 0]), int(obs_image[vy, vx, 1]), int(obs_image[vy, vx, 2])
            if obj_idx in (0, 1, 2):  # unseen, empty, wall - skip
                continue
            entry = {
                'type': OBJ_TYPES.get(obj_idx, 'unknown'),
                'color': COLORS.get(color_idx, 'unknown'),
                'view_x': vx, 'view_y': vy,
                'rel_x': vx - agent_vx,
                'rel_y': agent_vy - vy,  # positive = in front
            }
            if obj_idx == 4:  # door
                entry['door_state'] = DOOR_STATES.get(state, 'unknown')
            objects.append(entry)
    return objects


def view_to_world(rel_x, rel_y, agent_x, agent_y, agent_dir):
    """Convert agent-relative coordinates to world coordinates.

    Args:
        rel_x: left/right offset (negative=left of agent)
        rel_y: forward distance (positive=in front)
        agent_x, agent_y: agent's world position
        agent_dir: agent's absolute direction (0=right,1=down,2=left,3=up)

    Returns:
        (world_x, world_y)
    """
    # Agent's forward and right vectors based on direction
    # dir 0 (right):  forward=(1,0),  right=(0,1)
    # dir 1 (down):   forward=(0,1),  right=(-1,0)
    # dir 2 (left):   forward=(-1,0), right=(0,-1)
    # dir 3 (up):     forward=(0,-1), right=(1,0)
    forward_vectors = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
    right_vectors = {0: (0, 1), 1: (-1, 0), 2: (0, -1), 3: (1, 0)}

    fx, fy = forward_vectors[agent_dir]
    rx, ry = right_vectors[agent_dir]

    world_x = agent_x + rel_y * fx + rel_x * rx
    world_y = agent_y + rel_y * fy + rel_x * ry
    return (world_x, world_y)
