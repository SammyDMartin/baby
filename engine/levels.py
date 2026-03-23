"""Level definitions for standard BabyAI and custom nightmare levels."""
import gymnasium as gym
from gymnasium import register
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import (
    GoToInstr, PickupInstr, PutNextInstr, OpenInstr,
    ObjDesc, BeforeInstr, AndInstr,
)
import random

# ── Standard BabyAI levels ───────────────────────────────────────────────────

STANDARD_LEVELS = [
    {"id": "BabyAI-GoToRedBallNoDists-v0", "name": "GoTo Simple", "difficulty": 1,
     "desc": "Navigate to a red ball in an empty room."},
    {"id": "BabyAI-GoToObj-v0", "name": "GoTo Object", "difficulty": 1,
     "desc": "Navigate to a named object."},
    {"id": "BabyAI-GoToLocal-v0", "name": "GoTo Local", "difficulty": 2,
     "desc": "Navigate to object with distractors."},
    {"id": "BabyAI-GoToRedBlueBall-v0", "name": "GoTo Red/Blue", "difficulty": 2,
     "desc": "Navigate to red or blue ball among distractors."},
    {"id": "BabyAI-OpenDoor-v0", "name": "Open Door", "difficulty": 2,
     "desc": "Open a door based on description."},
    {"id": "BabyAI-OpenDoorColor-v0", "name": "Open Colored Door", "difficulty": 2,
     "desc": "Open a specific colored door."},
    {"id": "BabyAI-PickupLoc-v0", "name": "Pickup", "difficulty": 3,
     "desc": "Pick up an object by description."},
    {"id": "BabyAI-PickupDist-v0", "name": "Pickup + Distractors", "difficulty": 3,
     "desc": "Pick up an object among many distractors."},
    {"id": "BabyAI-PutNextLocalS5N3-v0", "name": "Put Next To", "difficulty": 4,
     "desc": "Pick up object and place it next to another."},
    {"id": "BabyAI-GoToObjMazeOpen-v0", "name": "GoTo Maze (Open)", "difficulty": 4,
     "desc": "Navigate through open multi-room maze."},
    {"id": "BabyAI-Unlock-v0", "name": "Unlock Door", "difficulty": 5,
     "desc": "Find key and unlock a door."},
    {"id": "BabyAI-UnlockPickup-v0", "name": "Unlock + Pickup", "difficulty": 6,
     "desc": "Unlock door, drop key, pick up object behind it."},
    {"id": "BabyAI-GoToObjMaze-v0", "name": "GoTo Maze (Closed)", "difficulty": 6,
     "desc": "Navigate through maze with closed doors."},
    {"id": "BabyAI-KeyCorridor-v0", "name": "Key Corridor", "difficulty": 7,
     "desc": "Find key behind doors in a corridor to unlock target room."},
    {"id": "BabyAI-BlockedUnlockPickup-v0", "name": "Blocked Unlock", "difficulty": 7,
     "desc": "Move blocking object, unlock door, pick up target."},
    {"id": "BabyAI-BossLevelNoUnlock-v0", "name": "Boss (No Unlock)", "difficulty": 8,
     "desc": "Compound missions across multiple rooms."},
    {"id": "BabyAI-BossLevel-v0", "name": "Boss Level", "difficulty": 9,
     "desc": "Full compound missions with unlocking across rooms."},
]


# ── Nightmare levels ─────────────────────────────────────────────────────────

class NightmareKeyChain(RoomGridLevel):
    """4-room corridor. 3 locked doors with chained keys.
    Must get key1, unlock door1, get key2, unlock door2, etc.
    Only carry one item at a time - must drop keys after use."""
    def __init__(self, **kwargs):
        super().__init__(room_size=6, num_rows=1, num_cols=4, max_steps=300, **kwargs)

    def gen_mission(self):
        colors = ['red', 'blue', 'green', 'yellow']
        random.shuffle(colors)
        self.place_agent(0, 0)
        for i in range(3):
            self.add_door(i, 0, door_idx=0, color=colors[i], locked=True)
            self.add_object(i, 0, 'key', colors[i])
        target, _ = self.add_object(3, 0, 'ball', colors[3])
        self.check_objs_reachable()
        self.instrs = PickupInstr(ObjDesc(target.type, target.color))


class NightmareMaze(RoomGridLevel):
    """4x4 room grid (16 rooms). All doors closed. Target in far corner.
    Agent must open 6+ doors to reach target. 100+ step sequences."""
    def __init__(self, **kwargs):
        super().__init__(room_size=6, num_rows=4, num_cols=4, max_steps=600, **kwargs)

    def gen_mission(self):
        self.place_agent(0, 0)
        for i in range(4):
            for j in range(4):
                if i < 3:
                    try:
                        self.add_door(i, j, door_idx=0, locked=False)
                    except Exception:
                        pass
                if j < 3:
                    try:
                        self.add_door(i, j, door_idx=1, locked=False)
                    except Exception:
                        pass
        target, _ = self.add_object(3, 3, 'ball', 'red')
        dist_colors = ['blue', 'green', 'purple', 'yellow', 'grey']
        for i in range(4):
            for j in range(4):
                if (i, j) != (3, 3) and random.random() < 0.4:
                    try:
                        self.add_object(i, j, random.choice(['ball', 'key', 'box']),
                                       random.choice(dist_colors))
                    except Exception:
                        pass
        self.check_objs_reachable()
        self.instrs = GoToInstr(ObjDesc(target.type, target.color))


class NightmareBacktrack(RoomGridLevel):
    """1x5 corridor. Target behind locked door at one end,
    key at the other end. Must traverse entire corridor twice."""
    def __init__(self, **kwargs):
        super().__init__(room_size=6, num_rows=1, num_cols=5, max_steps=400, **kwargs)

    def gen_mission(self):
        self.place_agent(2, 0)
        for i in range(4):
            if i == 0:
                self.add_door(i, 0, door_idx=0, color='red', locked=True)
            else:
                self.add_door(i, 0, door_idx=0, locked=False)
        self.add_object(4, 0, 'key', 'red')
        target, _ = self.add_object(0, 0, 'ball', 'blue')
        for i in range(1, 5):
            try:
                self.add_object(i, 0, random.choice(['box', 'ball']),
                               random.choice(['green', 'yellow', 'purple', 'grey']))
            except Exception:
                pass
        self.check_objs_reachable()
        self.instrs = PickupInstr(ObjDesc(target.type, target.color))


class NightmareCompound(RoomGridLevel):
    """3x3 rooms. Put blue ball next to green box, then pick up red key.
    Objects spread across rooms behind closed doors. Inventory management."""
    def __init__(self, **kwargs):
        super().__init__(room_size=7, num_rows=3, num_cols=3, max_steps=500, **kwargs)

    def gen_mission(self):
        self.place_agent(1, 1)
        self.open_all_doors()
        obj_a, _ = self.add_object(0, 0, 'key', 'red')
        obj_b, _ = self.add_object(2, 0, 'ball', 'blue')
        obj_c, _ = self.add_object(0, 2, 'box', 'green')
        for _ in range(5):
            try:
                r, c = random.randint(0, 2), random.randint(0, 2)
                self.add_object(r, c, random.choice(['ball', 'box']),
                               random.choice(['purple', 'grey', 'yellow']))
            except Exception:
                pass
        self.check_objs_reachable()
        sub1 = PutNextInstr(ObjDesc(obj_b.type, obj_b.color), ObjDesc(obj_c.type, obj_c.color))
        sub2 = PickupInstr(ObjDesc(obj_a.type, obj_a.color))
        self.instrs = BeforeInstr(sub1, sub2)


# Register nightmare levels
_NIGHTMARE_CLASSES = {
    'BabyAI-NightmareKeyChain-v0': NightmareKeyChain,
    'BabyAI-NightmareMaze-v0': NightmareMaze,
    'BabyAI-NightmareBacktrack-v0': NightmareBacktrack,
    'BabyAI-NightmareCompound-v0': NightmareCompound,
}

for env_id, cls in _NIGHTMARE_CLASSES.items():
    try:
        register(id=env_id, entry_point=f'{cls.__module__}:{cls.__name__}')
    except Exception:
        pass

NIGHTMARE_LEVELS = [
    {"id": "BabyAI-NightmareKeyChain-v0", "name": "Nightmare: Key Chain", "difficulty": 10,
     "desc": "4-room corridor with 3 locked doors. Chained key dependencies, single-item inventory."},
    {"id": "BabyAI-NightmareMaze-v0", "name": "Nightmare: Mega Maze", "difficulty": 11,
     "desc": "4x4 room grid (16 rooms), all doors closed. 100+ step navigation."},
    {"id": "BabyAI-NightmareBacktrack-v0", "name": "Nightmare: Backtrack", "difficulty": 12,
     "desc": "5-room corridor. Key at far end, target behind locked door at other end. Full double traversal."},
    {"id": "BabyAI-NightmareCompound-v0", "name": "Nightmare: Compound", "difficulty": 13,
     "desc": "3x3 rooms. Put-next-to then pickup. Multi-step inventory management across rooms."},
]

ALL_LEVELS = STANDARD_LEVELS + NIGHTMARE_LEVELS
