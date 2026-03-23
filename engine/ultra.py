"""Ultra-tier level definitions."""
import gymnasium as gym
from gymnasium import register
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import (
    GoToInstr, PickupInstr, PutNextInstr, BeforeInstr, ObjDesc,
)
import random


class UltraImpossibleGauntlet(RoomGridLevel):
    """4x4 room grid with carefully constructed locked-door chains.

    The grid has 16 rooms. A winding path of locked doors connects them,
    with keys placed so you must traverse the ENTIRE grid to collect them.
    The mission is compound: put-next-to then pickup, with objects in
    opposite corners.

    Key design: doors on the critical path are locked. Keys for each door
    are placed in the PREVIOUS room on the path, guaranteeing solvability.
    Extra locked doors (shortcuts) have their keys scattered, creating
    optional but tempting dead ends.
    """

    def __init__(self, **kwargs):
        super().__init__(
            room_size=6,
            num_rows=4,
            num_cols=4,
            max_steps=1500,
            **kwargs,
        )

    def gen_mission(self):
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'grey']
        random.shuffle(colors)

        # Define a winding path through all 16 rooms (snake pattern)
        # (col, row) pairs - snake goes right, down, left, down, right...
        path = []
        for row in range(4):
            cols = range(4) if row % 2 == 0 else range(3, -1, -1)
            for col in cols:
                path.append((col, row))

        # Agent starts at beginning of path
        self.place_agent(path[0][0], path[0][1])

        # Place locked doors along the critical path
        # Key for door N is in room N (before the door)
        critical_doors = []
        for idx in range(len(path) - 1):
            ci, cj = path[idx]
            ni, nj = path[idx + 1]

            # Determine which wall the door is on
            if ni == ci + 1:  # going right
                door_idx = 0  # right wall of current room
                door_room = (ci, cj)
            elif ni == ci - 1:  # going left
                door_idx = 0  # right wall of next room (= left wall of current)
                door_room = (ni, nj)
            elif nj == cj + 1:  # going down
                door_idx = 1  # bottom wall of current room
                door_room = (ci, cj)
            elif nj == cj - 1:  # going up
                door_idx = 1  # bottom wall of next room
                door_room = (ni, nj)
            else:
                continue

            color = colors[idx % len(colors)]
            try:
                self.add_door(door_room[0], door_room[1],
                              door_idx=door_idx, color=color, locked=True)
                critical_doors.append((door_room, door_idx, color, ci, cj))
            except Exception:
                pass

        # Place key for each critical door in the room BEFORE it on the path
        for door_room, door_idx, color, key_ci, key_cj in critical_doors:
            try:
                self.add_object(key_ci, key_cj, 'key', color)
            except Exception:
                # Room full, try adjacent rooms that are earlier on path
                placed = False
                for pi, pj in path:
                    if (pi, pj) == (key_ci, key_cj):
                        break
                    try:
                        self.add_object(pi, pj, 'key', color)
                        placed = True
                        break
                    except Exception:
                        continue

        # Add some non-critical doors (closed, not locked) for shortcuts
        for i in range(4):
            for j in range(4):
                for didx in [0, 1]:
                    if didx == 0 and i >= 3:
                        continue
                    if didx == 1 and j >= 3:
                        continue
                    try:
                        self.add_door(i, j, door_idx=didx, locked=False)
                    except Exception:
                        pass  # Already has a door

        # Mission objects in far-apart rooms
        obj_a, _ = self.add_object(0, 0, 'ball', 'red')
        obj_b, _ = self.add_object(3, 3, 'box', 'blue')
        target, _ = self.add_object(3, 0, 'ball', 'green')

        # Distractors
        dist_items = [('ball', 'purple'), ('box', 'grey'), ('ball', 'yellow'),
                      ('box', 'purple'), ('key', 'grey')]
        for i in range(4):
            for j in range(4):
                if random.random() < 0.3:
                    t, c = random.choice(dist_items)
                    try:
                        self.add_object(i, j, t, c)
                    except Exception:
                        pass

        self.check_objs_reachable()

        # Compound: put red ball next to blue box, then pick up green ball
        sub1 = PutNextInstr(
            ObjDesc(obj_a.type, obj_a.color),
            ObjDesc(obj_b.type, obj_b.color),
        )
        sub2 = PickupInstr(ObjDesc(target.type, target.color))
        self.instrs = BeforeInstr(sub1, sub2)


class UltraImpossibleLabyrinthPlus(RoomGridLevel):
    """6x6 room grid = 36 rooms, with locked doors on a critical path
    and closed doors elsewhere. Keys placed to force a long winding route.
    Compound mission requiring traversal of the entire grid."""

    def __init__(self, **kwargs):
        super().__init__(
            room_size=5,
            num_rows=6,
            num_cols=6,
            max_steps=2000,
            **kwargs,
        )

    def gen_mission(self):
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'grey']
        random.shuffle(colors)

        # Snake path through 36 rooms
        path = []
        for row in range(6):
            cols = range(6) if row % 2 == 0 else range(5, -1, -1)
            for col in cols:
                path.append((col, row))

        self.place_agent(path[0][0], path[0][1])

        # Lock every 3rd door on the critical path
        # This creates ~12 locked doors with keys in prior rooms
        critical_doors = []
        for idx in range(len(path) - 1):
            ci, cj = path[idx]
            ni, nj = path[idx + 1]
            locked = (idx % 3 == 0)

            if ni == ci + 1:
                door_room, door_idx = (ci, cj), 0
            elif ni == ci - 1:
                door_room, door_idx = (ni, nj), 0
            elif nj == cj + 1:
                door_room, door_idx = (ci, cj), 1
            elif nj == cj - 1:
                door_room, door_idx = (ni, nj), 1
            else:
                continue

            color = colors[idx % len(colors)] if locked else None
            try:
                self.add_door(door_room[0], door_room[1],
                              door_idx=door_idx, color=color, locked=locked)
                if locked:
                    critical_doors.append((door_room, door_idx, color, ci, cj))
            except Exception:
                pass

        # Keys in prior rooms
        for door_room, door_idx, color, key_ci, key_cj in critical_doors:
            try:
                self.add_object(key_ci, key_cj, 'key', color)
            except Exception:
                for pi, pj in path:
                    if (pi, pj) == (key_ci, key_cj):
                        break
                    try:
                        self.add_object(pi, pj, 'key', color)
                        break
                    except Exception:
                        continue

        # Fill remaining doorways with closed (not locked) doors
        for i in range(6):
            for j in range(6):
                for didx in [0, 1]:
                    if didx == 0 and i >= 5:
                        continue
                    if didx == 1 and j >= 5:
                        continue
                    try:
                        self.add_door(i, j, door_idx=didx, locked=False)
                    except Exception:
                        pass

        # Mission: compound across the grid
        obj_a, _ = self.add_object(0, 0, 'ball', 'red')
        obj_b, _ = self.add_object(5, 5, 'box', 'blue')
        target, _ = self.add_object(5, 0, 'ball', 'yellow')

        # Distractors
        dist = [('ball', 'green'), ('box', 'grey'), ('ball', 'purple'),
                ('box', 'green'), ('key', 'purple')]
        idx = 0
        for i in range(6):
            for j in range(6):
                if (i, j) not in [(0, 0), (5, 5), (5, 0)]:
                    try:
                        t, c = dist[idx % len(dist)]
                        self.add_object(i, j, t, c)
                        idx += 1
                    except Exception:
                        idx += 1

        self.check_objs_reachable()

        sub1 = PutNextInstr(
            ObjDesc(obj_a.type, obj_a.color),
            ObjDesc(obj_b.type, obj_b.color),
        )
        sub2 = PickupInstr(ObjDesc(target.type, target.color))
        self.instrs = BeforeInstr(sub1, sub2)


# Register
_ULTRA_CLASSES = {
    'BabyAI-UltraGauntlet-v0': UltraImpossibleGauntlet,
    'BabyAI-UltraLabyrinthPlus-v0': UltraImpossibleLabyrinthPlus,
}

for env_id, cls in _ULTRA_CLASSES.items():
    try:
        register(id=env_id, entry_point=f'{cls.__module__}:{cls.__name__}')
    except Exception:
        pass

ULTRA_LEVELS = [
    {
        "id": "BabyAI-UltraGauntlet-v0",
        "name": "Ultra: The Gauntlet",
        "difficulty": 20,
        "desc": "4x4 locked maze, all 15 doors on critical path locked, compound mission (put-next-to + pickup) across opposite corners. 300-600+ steps.",
    },
    {
        "id": "BabyAI-UltraLabyrinthPlus-v0",
        "name": "Ultra: Labyrinth+",
        "difficulty": 25,
        "desc": "6x6 rooms (36 rooms), 12 locked + 48 closed doors, compound mission across grid. 500-1000+ steps.",
    },
]
