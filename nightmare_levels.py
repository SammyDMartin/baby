"""
Secret Boss Levels: designed to exploit LLM spatial reasoning weaknesses.
"""
import gymnasium as gym
from gymnasium import register
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import (
    GoToInstr, PickupInstr, PutNextInstr, OpenInstr,
    ObjDesc, BeforeInstr, AndInstr, AfterInstr
)
from minigrid.envs.babyai.synth import LevelGen
import random


class NightmareKeyChain(RoomGridLevel):
    """
    4-room corridor. Each room separated by a locked door.
    Keys are placed so you must: get key1 → unlock door1 → get key2 → 
    unlock door2 → get key3 → unlock door3 → pickup target.
    Must manage inventory (can only carry 1 item) and backtrack.
    
    Why this is hard: 3-deep key dependency chain, must drop keys after use,
    must mentally track 4 rooms + 3 keys + 3 doors + inventory state.
    """
    def __init__(self, **kwargs):
        super().__init__(
            room_size=6,
            num_rows=1,
            num_cols=4,
            max_steps=300,
            **kwargs
        )
    
    def gen_mission(self):
        colors = ['red', 'blue', 'green', 'yellow']
        random.shuffle(colors)
        
        # Place agent in leftmost room
        self.place_agent(0, 0)
        
        # Create chain: door[i] between room i and i+1, locked with colors[i]
        # Key for door[i] is in room i (accessible before going through)
        for i in range(3):
            door, _ = self.add_door(i, 0, door_idx=0, color=colors[i], locked=True)
            # Key for this door is in the current room
            self.add_object(i, 0, 'key', colors[i])
        
        # Target object in the last room
        target, _ = self.add_object(3, 0, 'ball', colors[3])
        
        # Add distractors to make it visually noisy
        for i in range(4):
            try:
                self.add_object(i, 0, 'box', colors[i])
            except:
                pass
        
        self.check_objs_reachable()
        self.instrs = PickupInstr(ObjDesc(target.type, target.color))


class NightmareMaze(RoomGridLevel):
    """
    4x4 room grid (16 rooms). All doors closed. Target in opposite corner.
    Agent must open 6-8 doors to reach target.
    
    Why this is hard: 16 rooms = huge grid, long action sequences (~100+ steps),
    must plan efficient route through maze, easy to lose track of position.
    """
    def __init__(self, **kwargs):
        super().__init__(
            room_size=6,
            num_rows=4,
            num_cols=4,
            max_steps=600,
            **kwargs
        )
    
    def gen_mission(self):
        # Agent top-left
        self.place_agent(0, 0)
        
        # Add doors between all adjacent rooms (closed, not locked)
        for i in range(4):
            for j in range(4):
                # Right door
                if i < 3:
                    try:
                        self.add_door(i, j, door_idx=0, locked=False)
                    except:
                        pass
                # Bottom door
                if j < 3:
                    try:
                        self.add_door(i, j, door_idx=1, locked=False)
                    except:
                        pass
        
        # Target in bottom-right
        target, _ = self.add_object(3, 3, 'ball', 'red')
        
        # Scatter distractors
        dist_colors = ['blue', 'green', 'purple', 'yellow', 'grey']
        for i in range(4):
            for j in range(4):
                if (i, j) != (3, 3) and random.random() < 0.5:
                    try:
                        c = random.choice(dist_colors)
                        t = random.choice(['ball', 'key', 'box'])
                        self.add_object(i, j, t, c)
                    except:
                        pass
        
        self.check_objs_reachable()
        self.instrs = GoToInstr(ObjDesc(target.type, target.color))


class NightmareTripleTask(RoomGridLevel):
    """
    3x3 rooms. Must: pick up object A AND put object B next to object C,
    THEN open a locked door. Compound sequential mission with inventory management.
    
    Why this is hard: 3 interleaved subgoals, must plan execution order,
    objects spread across rooms behind closed doors, only 1 item at a time.
    """
    def __init__(self, **kwargs):
        super().__init__(
            room_size=7,
            num_rows=3,
            num_cols=3,
            max_steps=500,
            **kwargs
        )
    
    def gen_mission(self):
        self.place_agent(1, 1)  # center room
        
        # Open all doors first, then selectively lock some
        self.open_all_doors()
        
        # Add objects spread across rooms
        obj_a, _ = self.add_object(0, 0, 'key', 'red')     # top-left
        obj_b, _ = self.add_object(2, 0, 'ball', 'blue')    # top-right  
        obj_c, _ = self.add_object(0, 2, 'box', 'green')    # bottom-left
        
        # Lock one door to add a key-finding subgoal
        door, _ = self.add_door(1, 1, door_idx=1, color='yellow', locked=True)
        self.add_object(2, 2, 'key', 'yellow')  # key in bottom-right
        
        # Distractors
        for _ in range(8):
            try:
                r, c = random.randint(0, 2), random.randint(0, 2)
                self.add_object(r, c, random.choice(['ball','box','key']),
                               random.choice(['purple','grey','yellow']))
            except:
                pass
        
        self.check_objs_reachable()
        
        # Mission: put the blue ball next to the green box, then pick up the red key
        sub1 = PutNextInstr(ObjDesc(obj_b.type, obj_b.color), ObjDesc(obj_c.type, obj_c.color))
        sub2 = PickupInstr(ObjDesc(obj_a.type, obj_a.color))
        self.instrs = BeforeInstr(sub1, sub2)


class NightmareSwap(RoomGridLevel):
    """
    Single large room with 6 objects. Must put A next to B AND put C next to D.
    Objects start interleaved so you have to pick up, put down, pick up another,
    navigate around dropped objects.
    
    Why this is hard: Multiple pick-drop cycles in a crowded room. Dropped objects
    change the grid. Must track where you dropped things. Easy to accidentally
    drop in wrong spot and block paths.
    """
    def __init__(self, **kwargs):
        super().__init__(
            room_size=10,
            num_rows=1,
            num_cols=1,
            max_steps=200,
            **kwargs
        )
    
    def gen_mission(self):
        self.place_agent(0, 0)
        
        # Place objects deliberately spread out
        obj_a, _ = self.add_object(0, 0, 'ball', 'red')
        obj_b, _ = self.add_object(0, 0, 'box', 'blue')
        obj_c, _ = self.add_object(0, 0, 'key', 'green')
        obj_d, _ = self.add_object(0, 0, 'ball', 'yellow')
        
        # More distractors to clutter the room
        for _ in range(4):
            try:
                self.add_object(0, 0, random.choice(['ball','box','key']),
                               random.choice(['purple','grey']))
            except:
                pass
        
        self.check_objs_reachable()
        
        sub1 = PutNextInstr(ObjDesc(obj_a.type, obj_a.color), ObjDesc(obj_b.type, obj_b.color))
        sub2 = PutNextInstr(ObjDesc(obj_c.type, obj_c.color), ObjDesc(obj_d.type, obj_d.color))
        self.instrs = AndInstr(sub1, sub2)


class NightmareBacktrack(RoomGridLevel):
    """
    1x5 corridor of rooms. Target is in room 0, but it's behind a locked door.
    The key is in room 4 (far end). All intermediate doors are closed.
    Must traverse entire corridor, get key, traverse back.
    
    Why this is hard: ~150+ step action sequence. Must maintain direction
    tracking through 8 door-open-walk-through sequences. Easy to lose count.
    """
    def __init__(self, **kwargs):
        super().__init__(
            room_size=6,
            num_rows=1,
            num_cols=5,
            max_steps=400,
            **kwargs
        )
    
    def gen_mission(self):
        # Agent in middle
        self.place_agent(2, 0)
        
        # All intermediate doors closed but not locked
        for i in range(4):
            if i == 0:
                # Door between room 0 and 1 is locked
                self.add_door(i, 0, door_idx=0, color='red', locked=True)
            else:
                self.add_door(i, 0, door_idx=0, locked=False)
        
        # Key in room 4 (far right)
        self.add_object(4, 0, 'key', 'red')
        
        # Target in room 0 (far left, behind locked door)
        target, _ = self.add_object(0, 0, 'ball', 'blue')
        
        # Distractors
        for i in range(1, 5):
            try:
                self.add_object(i, 0, random.choice(['box','ball']), 
                               random.choice(['green','yellow','purple','grey']))
            except:
                pass
        
        self.check_objs_reachable()
        self.instrs = PickupInstr(ObjDesc(target.type, target.color))


# Register all nightmare levels
NIGHTMARE_LEVELS = {
    'BabyAI-NightmareKeyChain-v0': NightmareKeyChain,
    'BabyAI-NightmareMaze-v0': NightmareMaze,
    'BabyAI-NightmareTripleTask-v0': NightmareTripleTask,
    'BabyAI-NightmareSwap-v0': NightmareSwap,
    'BabyAI-NightmareBacktrack-v0': NightmareBacktrack,
}

for env_id, cls in NIGHTMARE_LEVELS.items():
    try:
        register(id=env_id, entry_point=f'{cls.__module__}:{cls.__name__}')
    except:
        pass  # already registered


if __name__ == "__main__":
    from babyai_harness import render_grid_text
    
    for env_id in NIGHTMARE_LEVELS:
        print(f"\n{'='*70}")
        print(f"{env_id}")
        print(f"{'='*70}")
        env = gym.make(env_id)
        obs, _ = env.reset(seed=42)
        print(f"Mission: {obs['mission']}")
        print(render_grid_text(env))
        env.close()
