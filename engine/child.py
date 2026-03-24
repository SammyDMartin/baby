"""ChildAI level definitions and test specifications.

New level types that defeat simple BFS-on-env solvers:
  - RedHerring: decoy keys that waste moves
  - Shuttle: ferry an object across multiple locked doors
  - OneWayCorridor: doors lock behind you (via wrapper)
  - TightLabyrinth: 36-room maze with strict step limit

Plus test specifications for Blind and Fog modes on existing levels.
"""
import gymnasium as gym
from gymnasium import register
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import (
    GoToInstr, PickupInstr, PutNextInstr, BeforeInstr, ObjDesc,
)
import random


# ── New ChildAI level classes ───────────────────────────────────────────

class ChildRedHerring(RoomGridLevel):
    """2 rooms separated by a locked door. Target behind the door.
    4 keys in the first room: only 1 matches the door color.
    The other 3 are decoys. Solver must identify the correct key by
    matching color to the door, not just grabbing the nearest key."""

    def __init__(self, **kwargs):
        super().__init__(room_size=7, num_rows=1, num_cols=2, max_steps=150, **kwargs)

    def gen_mission(self):
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'grey']
        random.shuffle(colors)
        door_color = colors[0]
        decoy_colors = colors[1:4]
        target_color = colors[4]

        self.place_agent(0, 0)
        self.add_door(0, 0, door_idx=0, color=door_color, locked=True)

        # Correct key
        self.add_object(0, 0, 'key', door_color)
        # Decoy keys (wrong colors)
        for c in decoy_colors:
            self.add_object(0, 0, 'key', c)

        # Target behind the locked door
        target, _ = self.add_object(1, 0, 'ball', target_color)

        # Extra distractors in room 1
        try:
            self.add_object(1, 0, 'box', colors[5])
        except Exception:
            pass

        self.check_objs_reachable()
        self.instrs = PickupInstr(ObjDesc(target.type, target.color))


class ChildShuttle(RoomGridLevel):
    """3 rooms in a line. Ball in room 0, box in room 2.
    Door 0->1 locked, door 1->2 locked (different colors).
    Keys for each door in the room before it.

    Mission: put ball next to box. Requires:
    1. Pick up key0, unlock door 0->1, drop key
    2. Go back, pick up ball, carry to room 1, drop ball
    3. Pick up key1, unlock door 1->2, drop key
    4. Go back to room 1, pick up ball, carry to room 2
    5. Drop ball next to box

    ~10 inventory swaps, ~100+ steps. Tests inventory planning."""

    def __init__(self, **kwargs):
        super().__init__(room_size=6, num_rows=1, num_cols=3, max_steps=350, **kwargs)

    def gen_mission(self):
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        random.shuffle(colors)

        self.place_agent(0, 0)

        # Locked doors between rooms
        self.add_door(0, 0, door_idx=0, color=colors[0], locked=True)
        self.add_door(1, 0, door_idx=0, color=colors[1], locked=True)

        # Key for door 0->1 in room 0
        self.add_object(0, 0, 'key', colors[0])
        # Key for door 1->2 in room 1
        self.add_object(1, 0, 'key', colors[1])

        # Ball in room 0
        obj_a, _ = self.add_object(0, 0, 'ball', colors[2])
        # Box in room 2
        obj_b, _ = self.add_object(2, 0, 'box', colors[3])

        # Distractors
        try:
            self.add_object(1, 0, 'ball', colors[4])
        except Exception:
            pass

        self.check_objs_reachable()
        self.instrs = PutNextInstr(
            ObjDesc(obj_a.type, obj_a.color),
            ObjDesc(obj_b.type, obj_b.color),
        )


class ChildOneWayCorridor(RoomGridLevel):
    """1x4 corridor. Each door is locked; key in the current room.

    Used with OneWayDoorWrapper: doors lock behind you after passing.
    The solver MUST pick up the key for the next door BEFORE walking
    through the current door, because it can't go back.

    A naive solver that always grabs the nearest key and opens the
    nearest door will fail: it'll walk through a door without having
    the key for the NEXT door, and get stuck.

    Correct strategy: in each room, pick up the key for the upcoming
    door, then unlock and proceed. Requires look-ahead planning."""

    def __init__(self, **kwargs):
        super().__init__(room_size=6, num_rows=1, num_cols=4, max_steps=250, **kwargs)

    def gen_mission(self):
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'grey']
        random.shuffle(colors)

        self.place_agent(0, 0)

        # Each room i has: locked door to room i+1, key for that door
        for i in range(3):
            self.add_door(i, 0, door_idx=0, color=colors[i], locked=True)
            self.add_object(i, 0, 'key', colors[i])

        # Target in last room
        target, _ = self.add_object(3, 0, 'ball', colors[3])

        # Distractors
        for i in range(4):
            try:
                self.add_object(i, 0, 'box', colors[(i + 4) % len(colors)])
            except Exception:
                pass

        self.check_objs_reachable()
        self.instrs = PickupInstr(ObjDesc(target.type, target.color))


class ChildTightLabyrinth(RoomGridLevel):
    """6x6 room labyrinth with a strict step limit.

    Same layout as ImpossibleLabyrinth (36 rooms, all doors closed),
    but max_steps is much tighter. The solver must find a near-optimal
    path. Brute-force exploration or inefficient routing will time out.

    The BFS-optimal path is typically 150-250 steps depending on seed.
    max_steps=300 leaves very little room for inefficiency."""

    def __init__(self, **kwargs):
        super().__init__(room_size=5, num_rows=6, num_cols=6, max_steps=300, **kwargs)

    def gen_mission(self):
        self.place_agent(0, 0)

        # Doors between all adjacent rooms, all closed (not locked)
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

        self.check_objs_reachable()
        self.instrs = GoToInstr(ObjDesc(target.type, target.color))


# ── Register new levels ─────────────────────────────────────────────────

_CHILD_CLASSES = {
    'ChildAI-RedHerring-v0': ChildRedHerring,
    'ChildAI-Shuttle-v0': ChildShuttle,
    'ChildAI-OneWayCorridor-v0': ChildOneWayCorridor,
    'ChildAI-TightLabyrinth-v0': ChildTightLabyrinth,
}

for _env_id, _cls in _CHILD_CLASSES.items():
    try:
        register(id=_env_id, entry_point=f'{_cls.__module__}:{_cls.__name__}')
    except Exception:
        pass


# ── Test specifications ─────────────────────────────────────────────────
# Each test has: id (env), name, difficulty, mode, seeds, desc
# mode: "open" = full access, "blind" = grid snapshot only, "fog" = partial obs

BLIND_TESTS = [
    {"id": "BabyAI-GoToObjMaze-v0", "name": "Blind: Maze Navigation",
     "difficulty": 30, "seeds": 5, "mode": "blind",
     "desc": "Navigate closed-door maze from grid snapshot. No env.step()."},
    {"id": "BabyAI-UnlockPickup-v0", "name": "Blind: Unlock + Pickup",
     "difficulty": 32, "seeds": 5, "mode": "blind",
     "desc": "Find key, unlock door, pick up target. Plan from grid snapshot."},
    {"id": "BabyAI-NightmareKeyChain-v0", "name": "Blind: Key Chain",
     "difficulty": 35, "seeds": 5, "mode": "blind",
     "desc": "3-key chain puzzle. Plan entire sequence from grid snapshot."},
    {"id": "BabyAI-NightmareCompound-v0", "name": "Blind: Compound Mission",
     "difficulty": 37, "seeds": 5, "mode": "blind",
     "desc": "Multi-step compound mission. Plan from grid snapshot."},
    {"id": "BabyAI-ImpossibleLabyrinth-v0", "name": "Blind: Labyrinth",
     "difficulty": 39, "seeds": 5, "mode": "blind",
     "desc": "36-room labyrinth. Plan 200+ step sequence from grid snapshot."},
]

FOG_TESTS = [
    {"id": "BabyAI-GoToObj-v0", "name": "Fog: GoTo Object",
     "difficulty": 40, "seeds": 5, "mode": "fog",
     "desc": "Find and go to named object. 7x7 partial view only."},
    {"id": "BabyAI-GoToObjMaze-v0", "name": "Fog: Maze Navigation",
     "difficulty": 43, "seeds": 5, "mode": "fog",
     "desc": "Navigate closed-door maze with 7x7 view. Must explore."},
    {"id": "BabyAI-Unlock-v0", "name": "Fog: Unlock Door",
     "difficulty": 45, "seeds": 5, "mode": "fog",
     "desc": "Find key and unlock door. 7x7 view only."},
    {"id": "BabyAI-NightmareMaze-v0", "name": "Fog: Mega Maze",
     "difficulty": 48, "seeds": 5, "mode": "fog",
     "desc": "16-room maze navigation. 7x7 view only. Full exploration needed."},
    {"id": "BabyAI-ImpossibleLabyrinth-v0", "name": "Fog: Labyrinth",
     "difficulty": 50, "seeds": 5, "mode": "fog",
     "desc": "36-room labyrinth. 7x7 view only. Full SLAM-style exploration."},
]

TRAP_TESTS = [
    {"id": "ChildAI-RedHerring-v0", "name": "Trap: Red Herring",
     "difficulty": 55, "seeds": 5, "mode": "open",
     "desc": "4 keys, only 1 unlocks the door. Must match key color to door."},
    {"id": "ChildAI-Shuttle-v0", "name": "Trap: Shuttle",
     "difficulty": 60, "seeds": 5, "mode": "open",
     "desc": "Ferry ball across 2 locked doors. 10+ inventory swaps required."},
    {"id": "ChildAI-OneWayCorridor-v0", "name": "Trap: One-Way Corridor",
     "difficulty": 65, "seeds": 5, "mode": "open", "wrapper": "oneway",
     "desc": "Doors lock behind you. Must grab key before proceeding."},
    {"id": "ChildAI-TightLabyrinth-v0", "name": "Trap: Tight Labyrinth",
     "difficulty": 70, "seeds": 5, "mode": "open",
     "desc": "36-room labyrinth with strict step limit (300). Near-optimal path required."},
]

CHILD_TESTS = BLIND_TESTS + FOG_TESTS + TRAP_TESTS
