from MDP_utils import  MDP
import numpy as np
import os
import random
import pickle


def createEnvironment(correct_transition = .85):
    """"
        creates a four room grid world MDP with the specified probability of correct transitions
    """

    dim1 = 19
    dim2 = 19
    state_dim = dim1*dim2
    action_dim = 5
    grid_world = MDP('gridworld')
    grid_world.gridworld_2D(dim=(dim1, dim2), p_correctmove= correct_transition)

    # initial rewards and obstacles
    grid_world.rewards = -4 * np.ones((dim1, dim2, action_dim));
    grid_world.obstacles = np.zeros((dim1, dim2), dtype=bool)

    # initialize agent in top left quadrant
    agent_pos_cords = [np.random.randint(2, 8), np.random.randint(2, 8)]
    agent_pos = agent_pos_cords[0]*dim1 + agent_pos_cords[1]
    # initialize goal in bottom right quadrant
    goal_pos_cords = [np.random.randint(11, 17), np.random.randint(11, 17)]
    goal_pos = goal_pos_cords[0]*dim1 + goal_pos_cords[1]

    # set locations of obstacles (one in each room)
    obstacle_x_locations = np.random.randint(2, 8, size= (2, 2))
    obstacle_y_locations = np.random.randint(2, 8, size=(2, 2))
    for i in range(0, 2):
        for j in range(0, 2):
            obstacle_x_locations[i, j] = int(obstacle_x_locations[i, j] + 9*i);
            obstacle_y_locations[i, j] = int(obstacle_y_locations[i, j] + 9*j);
            location = [obstacle_x_locations[i, j], obstacle_y_locations[i, j]]
            # place obstacles, ensuring that they are not at the goal or agent position
            if (not np.array_equal(location, goal_pos_cords)) and not (np.array_equal(location, agent_pos_cords)):
                grid_world.obstacles[location[0], location[1]] = 1
            else:
                grid_world.obstacles[location[0] + 1, location[1] + 1] = 1


    # randomly define ``doors'' in the walls
    door_top = np.random.randint(1, 9)
    door_bottom = np.random.randint(10, 18)
    door_left = np.random.randint(1, 9)
    door_right = np.random.randint(10, 18)
    # set locations of walls
    for i in range(0, dim1):
        for j in range(0, dim2):
            is_Wall = False
            if i == 0:
                is_Wall = True
            elif i == dim1 - 1:
                is_Wall = True
            elif j == 0:
                is_Wall = True
            elif j == dim2 - 1:
                is_Wall = True
            elif  i  == 9 and j != door_top and j != door_bottom:
                is_Wall = True
            elif j  == 9 and i != door_right and i != door_left:
                is_Wall = True
            if is_Wall:
                grid_world.obstacles[i, j] = 1;

    # large penalty for reaching obstacles
    for i in range(0, dim1):
        for j in range(0, dim2):
            if grid_world.obstacles[i, j]:
                grid_world.rewards[i,j, :] = - 200 * np.ones((action_dim))

    # reshape rewards im standard MDP format
    grid_world.rewards = np.reshape(grid_world.rewards, (state_dim, -1))

    # define reward for reaching goal, goal transitions back to start
    grid_world.enabled_actions[goal_pos] = [True, False, False, False, False]
    grid_world.rewards[goal_pos, 0] = 400
    grid_world.transitions[goal_pos, :, :] = np.zeros((action_dim, state_dim))
    grid_world.transitions[goal_pos, :, agent_pos] = 1

    # save goal and agent position
    grid_world.agent_pos_cords = agent_pos_cords
    grid_world.goal_pos_cords = goal_pos_cords

    # save p correct transition
    grid_world.correct_transition = correct_transition


    return grid_world


# create a set of grid worlds and save in a specified path
save_path = 'Insert my path here'
num_trials = 50


grid_worlds = []
np.random.seed(0)
for i in range(0, num_trials):
    grid_world = createEnvironment()
    grid_worlds.append(grid_world)
with open(save_path, 'wb') as grid_world_file:
    pickle.dump(grid_worlds, grid_world_file)
