"""
Show multiple BabyAI tasks across difficulty levels so I can solve them.
"""
from babyai_harness import show_task, run_task

# Representative tasks across difficulty spectrum
TASKS = [
    # Level 1: Simple go-to
    ("BabyAI-GoToRedBallNoDists-v0", 42),
    # Level 2: Go to with distractors
    ("BabyAI-GoToObj-v0", 42),
    # Level 3: Go to in local room
    ("BabyAI-GoToLocal-v0", 42),
    # Level 4: Open a door
    ("BabyAI-OpenDoor-v0", 42),
    # Level 5: Pick up an object
    ("BabyAI-PickupLoc-v0", 42),
    # Level 6: Unlock a door
    ("BabyAI-Unlock-v0", 42),
    # Level 7: Put object next to another
    ("BabyAI-PutNextLocalS5N3-v0", 42),
    # Level 8: Multi-room navigation
    ("BabyAI-GoToObjMaze-v0", 42),
    # Boss level
    ("BabyAI-BossLevel-v0", 42),
]

for env_name, seed in TASKS:
    print("=" * 70)
    show_task(env_name, seed)
    print()
