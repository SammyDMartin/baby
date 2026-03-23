"""
BabyAI test harness: renders gridworld as text, accepts action sequences,
reports success/failure. No access to the bot solver.
"""
import gymnasium as gym
import minigrid
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, DIR_TO_VEC
import numpy as np
import json
import sys

OBJECT_NAMES = {0: 'unseen', 1: 'empty', 2: 'wall', 3: 'floor', 4: 'door', 
                5: 'key', 6: 'ball', 7: 'box', 8: 'goal', 9: 'lava', 10: 'agent'}
COLOR_NAMES = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple', 4: 'yellow', 5: 'grey'}
DIR_NAMES = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
DIR_ARROWS = {0: '>', 1: 'v', 2: '<', 3: '^'}

# Door states: 0=open, 1=closed, 2=locked
DOOR_STATES = {0: 'open', 1: 'closed', 2: 'locked'}

ACTION_MAP = {
    'left': 0,    # turn left
    'right': 1,   # turn right  
    'forward': 2, # move forward
    'pickup': 3,  # pick up object
    'drop': 4,    # drop object
    'toggle': 5,  # toggle/open door
    'done': 6,    # declare done
}

def render_grid_text(env):
    """Render the FULL grid as text (god's eye view, not partial obs)."""
    grid = env.unwrapped.grid
    agent_pos = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir
    
    width = grid.width
    height = grid.height
    
    lines = []
    lines.append(f"Grid size: {width}x{height}")
    lines.append(f"Agent position: ({agent_pos[0]}, {agent_pos[1]}) facing {DIR_NAMES[agent_dir]}")
    
    carrying = env.unwrapped.carrying
    if carrying:
        lines.append(f"Carrying: {COLOR_NAMES.get(carrying.color, carrying.color)} {carrying.type}")
    else:
        lines.append("Carrying: nothing")
    
    lines.append("")
    
    # Column headers
    header = "   " + "".join(f"{x:>4}" for x in range(width))
    lines.append(header)
    
    for y in range(height):
        row = f"{y:2d} "
        for x in range(width):
            if (x, y) == tuple(agent_pos):
                row += f"  A{DIR_ARROWS[agent_dir]}"
            else:
                cell = grid.get(x, y)
                if cell is None:
                    row += "   ."
                elif cell.type == 'wall':
                    row += "  ##"
                elif cell.type == 'door':
                    color_char = cell.color[0].upper()
                    state = 'O' if cell.is_open else ('L' if cell.is_locked else 'C')
                    row += f" {color_char}D{state}"
                elif cell.type == 'key':
                    color_char = cell.color[0].upper()
                    row += f" {color_char}Ky"
                elif cell.type == 'ball':
                    color_char = cell.color[0].upper()
                    row += f" {color_char}Bl"
                elif cell.type == 'box':
                    color_char = cell.color[0].upper()
                    row += f" {color_char}Bx"
                elif cell.type == 'goal':
                    row += "  GL"
                elif cell.type == 'lava':
                    row += "  LA"
                else:
                    row += f"  {cell.type[:2]}"
        lines.append(row)
    
    lines.append("")
    lines.append("Legend: ## = wall, A>/Av/A</A^ = agent facing dir")
    lines.append("  XKy = key (X=color initial), XBl = ball, XBx = box")
    lines.append("  XDO/XDC/XDL = door open/closed/locked")
    lines.append("  Color initials: R=red, G=green, B=blue, P=purple, Y=yellow, E=grey")
    
    # Also list all objects with positions
    lines.append("")
    lines.append("Objects in grid:")
    for y in range(height):
        for x in range(width):
            cell = grid.get(x, y)
            if cell and cell.type not in ('wall', 'floor'):
                extra = ""
                if cell.type == 'door':
                    extra = f" ({'open' if cell.is_open else 'locked' if cell.is_locked else 'closed'})"
                if hasattr(cell, 'color') and cell.color:
                    lines.append(f"  ({x},{y}): {cell.color} {cell.type}{extra}")
                else:
                    lines.append(f"  ({x},{y}): {cell.type}{extra}")
    
    return "\n".join(lines)


def run_task(env_name, seed=None, action_sequence=None):
    """Run a BabyAI task and return the initial state + results."""
    env = gym.make(env_name)
    
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()
    
    mission = obs['mission']
    
    result = {
        'env': env_name,
        'mission': mission,
        'initial_grid': render_grid_text(env),
        'seed': seed,
    }
    
    if action_sequence:
        steps = []
        total_reward = 0
        done = False
        truncated = False
        
        for i, action_name in enumerate(action_sequence):
            if done or truncated:
                break
            
            action = ACTION_MAP.get(action_name)
            if action is None:
                steps.append(f"Step {i+1}: Invalid action '{action_name}'")
                break
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            status = ""
            if done and reward > 0:
                status = " [SUCCESS!]"
            elif done:
                status = " [FAILED]"
            elif truncated:
                status = " [TRUNCATED - too many steps]"
            
            steps.append(f"Step {i+1}: {action_name}{status}")
        
        result['steps'] = steps
        result['total_reward'] = total_reward
        result['success'] = done and total_reward > 0
        result['final_grid'] = render_grid_text(env)
    
    env.close()
    return result


def show_task(env_name, seed=None):
    """Just show the initial state of a task."""
    result = run_task(env_name, seed=seed)
    print(f"=== {result['env']} ===")
    print(f"Mission: {result['mission']}")
    print()
    print(result['initial_grid'])
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python babyai_harness.py <env_name> [seed]")
        sys.exit(1)
    
    env_name = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    show_task(env_name, seed)
