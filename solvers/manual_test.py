"""Manual solve test framework.
Renders the grid and verifies a manually-produced action sequence."""
import sys
import gymnasium as gym
from engine.impossible import *
from engine.grid import ACTION_MAP, DIR_NAMES, DIR_ARROWS


def render_compact(env):
    """Render grid compactly for manual solving."""
    grid = env.unwrapped.grid
    agent_pos = tuple(env.unwrapped.agent_pos)
    agent_dir = env.unwrapped.agent_dir
    w, h = grid.width, grid.height

    lines = []
    lines.append(f"Agent: ({agent_pos[0]},{agent_pos[1]}) facing {DIR_NAMES[agent_dir]}")
    carrying = env.unwrapped.carrying
    if carrying:
        lines.append(f"Carrying: {carrying.color} {carrying.type}")

    header = "   " + "".join(f"{x:>3}" for x in range(w))
    lines.append(header)

    for y in range(h):
        row = f"{y:2d} "
        for x in range(w):
            if (x, y) == agent_pos:
                row += f" A{DIR_ARROWS[agent_dir]}"
            else:
                cell = grid.get(x, y)
                if cell is None:
                    row += "  ."
                elif cell.type == 'wall':
                    row += " ##"
                elif cell.type == 'door':
                    c = cell.color[0].upper()
                    s = 'O' if cell.is_open else ('L' if cell.is_locked else 'C')
                    row += f"{c}D{s}"
                elif cell.type == 'key':
                    row += f" {cell.color[0].upper()}k"
                elif cell.type == 'ball':
                    row += f" {cell.color[0].upper()}o"
                elif cell.type == 'box':
                    row += f" {cell.color[0].upper()}x"
                elif cell.type == 'goal':
                    row += " GL"
                else:
                    row += f" {cell.type[:2]}"
        lines.append(row)
    return "\n".join(lines)


def test_actions(seed, actions):
    """Test an action sequence on the Impossible Labyrinth."""
    env = gym.make('BabyAI-ImpossibleLabyrinth-v0')
    obs, _ = env.reset(seed=seed)
    mission = obs['mission']

    print(f"=== Seed {seed} ===")
    print(f"Mission: {mission}")
    print(render_compact(env))
    print()

    total_reward = 0
    for i, a in enumerate(actions):
        if a not in ACTION_MAP:
            print(f"Step {i+1}: Invalid action '{a}'")
            env.close()
            return False

        obs, reward, done, truncated, _ = env.step(ACTION_MAP[a])
        total_reward += reward

        if done:
            success = reward > 0
            print(f"Step {i+1}/{len(actions)}: {a} -> {'SUCCESS!' if success else 'FAILED'}")
            env.close()
            return success

        if truncated:
            print(f"Step {i+1}: TRUNCATED")
            env.close()
            return False

    # Didn't complete
    pos = tuple(env.unwrapped.agent_pos)
    print(f"Ran {len(actions)} actions, no completion. Agent at {pos}")
    print(render_compact(env))
    env.close()
    return False


def show_grid(seed):
    """Just show the grid for a seed."""
    env = gym.make('BabyAI-ImpossibleLabyrinth-v0')
    obs, _ = env.reset(seed=seed)
    print(f"=== Seed {seed} ===")
    print(f"Mission: {obs['mission']}")
    print(render_compact(env))

    # Show all objects
    grid = env.unwrapped.grid
    print("\nNon-wall objects:")
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get(x, y)
            if cell and cell.type not in ('wall',):
                extra = ""
                if cell.type == 'door':
                    extra = f" ({'open' if cell.is_open else 'locked' if cell.is_locked else 'closed'})"
                color = getattr(cell, 'color', '')
                print(f"  ({x:2d},{y:2d}): {color} {cell.type}{extra}")
    env.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python manual_test.py show <seed>")
        print("       python manual_test.py test <seed> <actions...>")
        sys.exit(1)

    cmd = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    if cmd == 'show':
        show_grid(seed)
    elif cmd == 'test':
        actions = sys.argv[3:]
        test_actions(seed, actions)
