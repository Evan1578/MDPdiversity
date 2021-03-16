from MDP_utils import  MDP
import numpy as np
import mdptoolbox
import scipy
import math
import itertools
import math
import os
import random
import unittest
import torch
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from expanded_approach import createGridWorld
import pickle
import matplotlib.pyplot as plt
import multiprocessing
import gym_minigrid
import gym

def createEnvironment(correct_transition = .95):

    dim1 = 19
    dim2 = 19
    state_dim = dim1*dim2
    action_dim = 5
    grid_world = MDP('gridworld')
    grid_world.gridworld_2D(dim=(dim1, dim2), p_correctmove= correct_transition)

    # initial rewards and obstacles
    grid_world.rewards = -1.2 * np.ones((dim1, dim2, action_dim));
    grid_world.obstacles = np.zeros((dim1, dim2), dtype=bool)

    # initialize agent in top left quadrant
    agent_pos_cords = [np.random.randint(2, 5), np.random.randint(2, 5)]
    agent_pos = agent_pos_cords[0]*dim1 + agent_pos_cords[1]
    # initialize goal in bottom right quadrant
    goal_pos_cords = [np.random.randint(14, 17), np.random.randint(14, 17)]
    goal_pos = goal_pos_cords[0]*dim1 + goal_pos_cords[1]

    # set locations of obstacles (one in each room)
    obstacle_x_locations = np.random.randint(2, 5, size= (3, 3))
    obstacle_y_locations = np.random.randint(2, 5, size=(3, 3))
    for i in range(0, 3):
        for j in range(0, 3):
            obstacle_x_locations[i, j] = int(obstacle_x_locations[i, j] + 6*i);
            obstacle_y_locations[i, j] = int(obstacle_y_locations[i, j] + 6*j);
            location = [obstacle_x_locations[i, j], obstacle_y_locations[i, j]]
            # place obstacles, ensuring that they are not at the goal or agent position
            if (not np.array_equal(location, goal_pos_cords)) and (not np.array_equal(location, agent_pos_cords)):
                grid_world.obstacles[location[0], location[1]] = 1
            else:
                grid_world.obstacles[location[0] + 1, location[1] + 1] = 1


    # set walls
    grid_world.obstacles[:, 0] = np.ones((dim1))
    grid_world.obstacles[:, -1] = np.ones((dim1))
    grid_world.obstacles[:, 6] = np.ones((dim1))
    grid_world.obstacles[:, 12] = np.ones((dim1))
    grid_world.obstacles[0, :] = np.ones((dim2))
    grid_world.obstacles[-1, :] = np.ones((dim2))
    grid_world.obstacles[6, :] = np.ones((dim2))
    grid_world.obstacles[12, :] = np.ones((dim2))


    # randomly define ``doors'' in the walls
    vertical_doors = np.random.randint(2, 5, size=(3, 2))
    horizontal_doors = np.random.randint(2, 5, size=(3, 2))
    grid_world.obstacles[int(vertical_doors[0, 0]) , 6] = 0;
    grid_world.obstacles[int(6 + vertical_doors[1, 0]), 6] = 0;
    grid_world.obstacles[int(12 + vertical_doors[2, 0]), 6] = 0;
    grid_world.obstacles[int(vertical_doors[0, 1]) , 12] = 0;
    grid_world.obstacles[int(6 + vertical_doors[1, 1]), 12] = 0;
    grid_world.obstacles[int(12 + vertical_doors[2, 1]), 12] = 0;
    grid_world.obstacles[6, int(horizontal_doors[0, 0]) ] = 0;
    grid_world.obstacles[6, int(6 + horizontal_doors[1, 0])] = 0;
    grid_world.obstacles[6, int(12 + horizontal_doors[2, 0])] = 0;
    grid_world.obstacles[12, int(horizontal_doors[0, 1]) ] = 0;
    grid_world.obstacles[12, int(6 + horizontal_doors[1, 1])] = 0;
    grid_world.obstacles[12, int(12 + horizontal_doors[2, 1])] = 0;


    # large penalty for reaching obstacles
    for i in range(0, dim1):
        for j in range(0, dim2):
            if grid_world.obstacles[i, j]:
                grid_world.rewards[i,j, :] = - 40 * np.ones((action_dim))



    # reshape rewards im standard MDP format
    grid_world.rewards = np.reshape(grid_world.rewards, (state_dim, -1))

    # define reward for reaching goal, goal transitions back to start
    grid_world.enabled_actions[goal_pos] = [True, False, False, False, False]
    grid_world.rewards[goal_pos, 0] = 200
    grid_world.transitions[goal_pos, :, :] = np.zeros((action_dim, state_dim))
    grid_world.transitions[goal_pos, :, agent_pos] = 1


    # save goal and agent position
    grid_world.agent_pos_cords = agent_pos_cords
    grid_world.goal_pos_cords = goal_pos_cords

    # save p correct transition
    grid_world.correct_transition = correct_transition


    return grid_world

grid_worlds = []
np.random.seed(0)
for i in range(0, 50):
    grid_world = createEnvironment()
    grid_worlds.append(grid_world)


save_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data')
grid_path = os.path.join(save_dir, 'nineRooms/grid_worldsICAPSp95')
with open(grid_path, 'wb') as grid_world_file:
    pickle.dump(grid_worlds, grid_world_file)