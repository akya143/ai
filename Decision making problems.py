import numpy as np
import random
import matplotlib.pyplot as plt
grid_size = (5, 5)  # 5x5 grid
goal_state = (4, 4)  # goal is at the bottom-right corner
start_state = (0, 0)  # start at the top-left corner
actions = ['up', 'down', 'left', 'right']
action_map = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
learning_rate = 0.1    # Alpha
discount_factor = 0.9  # Gamma
epsilon = 0.1         # Epsilon for epsilon-greedy policy
num_episodes = 1000   # Number of episodes for training
max_steps_per_episode = 100  # Maximum steps per episode
q_table = np.zeros((grid_size[0], grid_size[1], len(actions)))
def reward_function(state):
    if state == goal_state:
        return 100  # reward for reaching the goal
    return -1  # penalty for each step
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        state_x, state_y = state
        q_values = q_table[state_x, state_y, :]
        max_q_value = np.max(q_values)
        max_actions = [actions[i] for i in range(len(actions)) if q_values[i] == max_q_value]
        return random.choice(max_actions)
def take_action(state, action):
    state_x, state_y = state
    move = action_map[action]
    new_x = max(0, min(grid_size[0] - 1, state_x + move[0]))  # Ensure we don't go out of bounds
    new_y = max(0, min(grid_size[1] - 1, state_y + move[1]))  # Ensure we don't go out of bounds
    return (new_x, new_y)
def train_q_learning():
    for episode in range(num_episodes):
        state = start_state
        total_reward = 0
        for step in range(max_steps_per_episode):
            action = choose_action(state)
            next_state = take_action(state, action)
            reward = reward_function(next_state)
            state_x, state_y = state
            next_state_x, next_state_y = next_state
            action_index = actions.index(action)
            max_future_q = np.max(q_table[next_state_x, next_state_y, :])
            q_table[state_x, state_y, action_index] += learning_rate * (reward + discount_factor * max_future_q - q_table[state_x, state_y, action_index])
            state = next_state
            total_reward += reward
            if state == goal_state:
                break
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1} completed")
def visualize_policy():
    policy_grid = np.full(grid_size, '', dtype=object)
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            best_action_index = np.argmax(q_table[x, y, :])
            best_action = actions[best_action_index]
            policy_grid[x, y] = best_action[0].upper()  # Show the first letter of the best action
    plt.figure(figsize=(6, 6))
    plt.imshow(np.zeros(grid_size), cmap="Blues", interpolation='none')
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            plt.text(y, x, policy_grid[x, y], ha='center', va='center', color='black', fontsize=12)
    
    plt.title("Learned Policy (Q-learning)")
    plt.xticks(range(grid_size[1]))
    plt.yticks(range(grid_size[0]))
    plt.gca().invert_yaxis()
    plt.show()
train_q_learning()
visualize_policy()
