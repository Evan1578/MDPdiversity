import numpy as np
import itertools
import os
import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def visualize_occupancy_maps(occupancy_maps, grid_world):
    # define colormap
    sequential = cm.get_cmap('Blues', 256)
    check = np.linspace(0, 1, 256)
    newcolors = sequential(check)
    agent = np.array([141 / 256, 2  / 256, 31 / 256, 1])
    goal = np.array([177/256,156/256,217/256, 1])
    obstacle = np.array([0, 0, 0, 1])
    newcolors[-2, :] = agent
    newcolors[-3, :] = goal
    newcolors[-1, :] = obstacle
    newcmp = ListedColormap(newcolors)


    n_policies = len(occupancy_maps)
    # get obstacle data and agent and goal location
    state_dim = len(occupancy_maps[0])
    dim1 = int(np.sqrt(state_dim))
    dim2 = dim1
    action_dim = 5


    X = grid_world.obstacles.astype(float)
    X[grid_world.agent_pos_cords[0], grid_world.agent_pos_cords[1]] = check[-2]
    X[grid_world.goal_pos_cords[0], grid_world.goal_pos_cords[1]] = check[-3]
    fig, axs = plt.subplots(1, n_policies)
    for j in range(0, n_policies):
        to_image = occupancy_maps[j]
        if torch.is_tensor(to_image):
            to_image = to_image.detach().numpy()
        to_image = np.sum(to_image, axis = 1)
        to_image = np.reshape(to_image, (dim1, dim2))
        max = np.max(to_image);
        to_image = np.maximum((to_image/max)*.9, X)

        axs[j].xaxis.set_visible(False)
        axs[j].yaxis.set_visible(False)
        axs[j].imshow(to_image, cmap=newcmp)

    plt.show()



def main(occupancy_path, grid_path, trial, identifier):
    """"
    visualizes occupancy map from 'occupancy_path' file with corresponding grid world 'grid_path'
    'trial' and 'identifier' specify the specific trial and experimental setup to visualize
     """

    with open(occupancy_path, 'rb') as occupancy_file:
        occupancy_maps = pickle.load(occupancy_file)
    with open(grid_path, 'rb') as grid_file:
        grid_worlds = pickle.load(grid_file)

    grid_world = grid_worlds[trial]
    occupancy_maps_viz = occupancy_maps[(identifier, trial)]
    visualize_occupancy_maps(occupancy_maps_viz, grid_world)

