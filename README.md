This code is a clean version of the diverse stochastic planning algorithm used in the paper Multiple Plans are Better than One: Diverse Stochastic Planning, appearing at ICAPS 2021.

This only supports the grid world examples used in the paper, but feel free to expand and add your own MDP examples

The main requirements are: numpy, scipy, pandas, matplotlib, and pytorch - see the requirements.txt file for full specifications.

To use this package, first run 'createFourRooms.py' or 'createNineRooms.py' to create the grid worlds.

Then, used the saved grid worlds to test the performance of the diverse stochastic planning algorithm using 'DiverseStochasticPlanning.py'

The resulting occupancy maps can be visualized using 'visualizeOccupancyMaps.py'.

Finally, please get in touch if you have questions or need help using our algorithm. My email is escopec - at - utexas.edu. 
