"""Honest manual solve verification.
Shows the grid, then accepts a COMPLETE action sequence and verifies it.
No peeking at intermediate states."""
import sys
import gymnasium as gym
from engine.impossible import *
from engine.grid import ACTION_MAP, DIR_NAMES, DIR_ARROWS, DIR_DELTAS


def show_challenge(seed):
    """Show the grid for a manual solve challenge."""
    env = gym.make('BabyAI-ImpossibleLabyrinth-v0')
    obs, _ = env.reset(seed=seed)
    grid = env.unwrapped.grid
    pos = (int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1]))
    d = int(env.unwrapped.agent_dir)
    w, h = grid.width, grid.height

    # Find target
    target = None
    for y in range(h):
        for x in range(w):
            c = grid.get(x, y)
            if c and c.type == 'ball' and c.color == 'red':
                target = (x, y)

    print(f"CHALLENGE: Seed {seed}")
    print(f"Mission: {obs['mission']}")
    print(f"Agent: ({pos[0]},{pos[1]}) facing {DIR_NAMES[d]}")
    print(f"Target: red ball at ({target[0]},{target[1]})")
    print(f"Grid: {w}x{h}")
    print()

    # Print grid with coordinates
    header = "    " + "".join(f"{x:>3}" for x in range(w))
    print(header)

    for y in range(h):
        row = f"{y:2d}  "
        for x in range(w):
            if (x, y) == pos:
                row += f" A{DIR_ARROWS[d]}"
            elif (x, y) == target:
                row += " *R"
            else:
                cell = grid.get(x, y)
                if cell is None:
                    row += "  ."
                elif cell.type == 'wall':
                    row += " ##"
                elif cell.type == 'door':
                    c = cell.color[0].upper()
                    s = 'O' if cell.is_open else ('L' if cell.is_locked else 'C')
                    row += f" {c}{s}"
                elif cell.type in ('key', 'ball', 'box'):
                    row += f" {cell.color[0].lower()}{cell.type[0]}"
                else:
                    row += "  ?"
        print(row)

    print()
    print("Legend: ## wall, A^ agent, *R target, XC closed door, XO open door")
    print("  xk=key, xb=ball, xx=box (lowercase = color initial)")
    print()

    # Print all doors with positions
    print("DOORS (non-wall boundaries):")
    for y in range(h):
        for x in range(w):
            cell = grid.get(x, y)
            if cell and cell.type == 'door':
                state = 'open' if cell.is_open else 'locked' if cell.is_locked else 'closed'
                print(f"  ({x:2d},{y:2d}): {cell.color} door ({state})")

    env.close()
    return pos, d, target


def verify_actions(seed, actions_str):
    """Verify a space-separated action sequence."""
    actions = actions_str.strip().split()

    env = gym.make('BabyAI-ImpossibleLabyrinth-v0')
    obs, _ = env.reset(seed=seed)

    total_reward = 0
    for i, a in enumerate(actions):
        if a not in ACTION_MAP:
            print(f"INVALID action '{a}' at step {i+1}")
            env.close()
            return False

        obs, reward, done, truncated, _ = env.step(ACTION_MAP[a])
        total_reward += reward

        if done:
            success = reward > 0
            print(f"{'SUCCESS' if success else 'FAILED'} at step {i+1}/{len(actions)}")
            env.close()
            return success

        if truncated:
            print(f"TRUNCATED at step {i+1}")
            env.close()
            return False

    pos = (int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1]))
    print(f"INCOMPLETE: {len(actions)} actions, agent at ({pos[0]},{pos[1]})")
    env.close()
    return False


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python honest_attempt.py show <seed>")
        print("  python honest_attempt.py verify <seed> '<space-separated actions>'")
        sys.exit(1)

    cmd = sys.argv[1]
    seed = int(sys.argv[2])

    if cmd == 'show':
        show_challenge(seed)
    elif cmd == 'verify':
        actions_str = sys.argv[3] if len(sys.argv) > 3 else ''
        verify_actions(seed, actions_str)
