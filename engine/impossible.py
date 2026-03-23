"""Impossible Labyrinth level definition."""
import gymnasium as gym
from gymnasium import register
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc
import random


class ImpossibleLabyrinth(RoomGridLevel):
    """6x6 room grid. Every door closed. Target in far corner.
    Distractor objects everywhere. 200-300 step sequences."""

    def __init__(self, **kwargs):
        super().__init__(
            room_size=5,
            num_rows=6,
            num_cols=6,
            max_steps=1000,
            **kwargs
        )

    def gen_mission(self):
        self.place_agent(0, 0)

        # Place doors between ALL adjacent rooms - all closed, not locked
        for i in range(6):
            for j in range(6):
                if i < 5:
                    try:
                        self.add_door(i, j, door_idx=0, locked=False)
                    except Exception:
                        pass
                if j < 5:
                    try:
                        self.add_door(i, j, door_idx=1, locked=False)
                    except Exception:
                        pass

        # Target in far corner
        target, _ = self.add_object(5, 5, 'ball', 'red')

        # Fixed distractor pattern - deterministic to avoid generation randomness
        dist_patterns = [
            ('ball', 'blue'), ('box', 'green'), ('key', 'purple'),
            ('ball', 'yellow'), ('box', 'grey'), ('key', 'blue'),
            ('ball', 'green'), ('box', 'purple'), ('key', 'yellow'),
        ]
        idx = 0
        for i in range(6):
            for j in range(6):
                if (i, j) == (5, 5):
                    continue
                try:
                    t, c = dist_patterns[idx % len(dist_patterns)]
                    self.add_object(i, j, t, c)
                    idx += 1
                except Exception:
                    idx += 1

        self.check_objs_reachable()
        self.instrs = GoToInstr(ObjDesc(target.type, target.color))


# Register
try:
    register(id='BabyAI-ImpossibleLabyrinth-v0',
             entry_point=f'{ImpossibleLabyrinth.__module__}:{ImpossibleLabyrinth.__name__}')
except Exception:
    pass

IMPOSSIBLE_LEVELS = [
    {"id": "BabyAI-ImpossibleLabyrinth-v0", "name": "Impossible: Labyrinth", "difficulty": 15,
     "desc": "6x6 rooms (36 rooms), all doors closed, 50+ distractors. 200-300+ step navigation."},
]
