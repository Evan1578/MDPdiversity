from DiverseStochasticPlanning import *
from visualizeOccupancyMaps import visualize_occupancy_maps

# load data
grid_path = 'exampleGridWorld'
with open(grid_path, 'rb') as grid_world_file:
    grid_worlds = pickle.load(grid_world_file)
grid_world = grid_worlds[0]

lam = 10 # lambda value to test
b = 2 # number of policies to return

optimal_occupancies, div_score_two, div_score_js, rewards, iterates, time_elapsed = diversePlanning(grid_world, lam, b, obj_metric= Jensen_Shannon, proj_metric = 'two_Norm')

# visualize results
visualize_occupancy_maps(optimal_occupancies, grid_world)
