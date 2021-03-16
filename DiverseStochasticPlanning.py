from MDP_utils import  MDP
import numpy as np
import scipy
import math
import itertools
import math
import time
import os
import pandas
import torch
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
import pickle

def createRandomPolicy(num_states, num_actions, M):
    """"
        creates a random policy for the MDP 'M'
    """
    policy = np.zeros((num_states, num_actions))
    for i in range(0, num_states):
        # compute number of enabled actions
        num_enabled_actions =  np.count_nonzero(M.enabled_actions[i, :])
        # determine random action distribution + normalize
        random_actions = np.random.rand(1, num_enabled_actions)
        random_actions = random_actions / np.sum(random_actions)
        k = 0
        for j in range(0, num_actions):
            if M.enabled_actions[i, j]:
                policy[i, j] = random_actions[0, k]
                k = k + 1
    return policy

def policy2occupancy(policy, num_states, num_actions, M):
    """"
        converts a policy for the MDP 'M' into an occupany map
    """
    induced_transition = np.zeros((num_states, num_states))
    for i in range(0, num_states):
        induced_transition[i, :] = M.transitions[i, :, :].transpose() @ policy[i, :]
    w, vl = scipy.linalg.eig(induced_transition, left=True, right=False) # get left eigenvectors
    w = np.real(w)
    lar_index = [i for i, j in enumerate(w) if j == max(w)]
    # take largest eigenvector
    larg_vector = np.real(vl[:, lar_index[0]])
    larg_vector = larg_vector / np.sum(larg_vector)
    occupancy = np.zeros((num_states, num_actions))
    for i in range(0, num_states):
        occupancy[i, :] = larg_vector[i] * policy[i, :]
    return occupancy

def KL_divergence(policy_1, policy_2):
    """"
        defines KL-divergence distance between two policies
    """
    d = 0.0
    for i in range(0, policy_1.shape[0]):
        for j in range(0, policy_1.shape[1]):
            if policy_1[i, j] > 0 and policy_2[i, j] > 0:
                d  = d + (policy_1[i, j] * torch.log(policy_1[i, j]/policy_2[i, j]))
                d = d - (policy_1[i, j] - policy_2[i, j])
    return d


def Jensen_Shannon(policy_1, policy_2):
    """"
        defines Jensen_Shannon distance between two policies
    """
    d = torch.tensor([0.0])
    M = .5*(policy_1 + policy_2)
    for i in range(0, policy_1.shape[0]):
        for j in range(0, policy_1.shape[1]):
                if policy_1[i, j] != 0:
                    d  = d + .5*(policy_1[i, j] * torch.log(policy_1[i, j]/M[i, j]))
                if torch.isnan(d):
                    print("oh hell no")
    for i in range(0, policy_2.shape[0]):
        for j in range(0, policy_2.shape[1]):
                if policy_2[i, j] != 0:
                    d  = d + .5*(policy_2[i, j] * torch.log(policy_2[i, j]/M[i, j]))
    return d

def two_Norm(policy_1, policy_2):
    return torch.norm(policy_1 - policy_2, p = 'nuc')

def gradProj(occupancy, M, past_occupancy, metric = 'two_Norm'):
    """"
        Solves the gradient projection step in the projected gradient ascent method
    """

    if metric != 'KL_divergence' and metric != 'two_Norm':
        raise Exception("method not coded yet")

    num_states = occupancy.shape[0]
    num_actions = occupancy.shape[1]

    # nonnegative bound + 0 constraint on non-enabled actions
    lb = np.zeros(occupancy.shape)
    ub = np.zeros(occupancy.shape)
    for i in range(0, num_states):
        for j in range(0, num_actions):
            if M.enabled_actions[i, j]:
                ub[i, j] = np.inf

    occupancy = occupancy.flatten()
    lb = lb.flatten()
    ub = ub.flatten()
    bounds = Bounds(lb, ub)

    # add constraints
    constraints = []
    # must sum to one
    constraint = np.ones((1, occupancy.shape[0]))
    constraints.append(LinearConstraint(constraint, 1, 1))
    # flow property
    for i in range(0, num_states):
        rhs = M.transitions[:, :, i].flatten()
        lhs = np.zeros((1, num_states*num_actions))
        for j in range(0, num_actions):
            lhs[0, i*num_actions + j] = 1
        constraint = rhs - lhs
        constraints.append(LinearConstraint(constraint, 0, 0))

    # define objective function
    def KL_projection(x, occupancy):
        d = 0
        for i in range(0, x.shape[0]):
                if x[i] != 0 and occupancy[i] != 0:
                    d = d + (x[i] * np.log(x[i] / occupancy[i]))
                    d = d - (x[i] - occupancy[i])
                    if np.isnan(d):
                        print("oh no")
        return d

    def two_projection(x, occupancy):
        return np.linalg.norm(x - occupancy, ord = 2)

    # define initial guess as occupancy induced by random policy
    x0 = past_occupancy.flatten()

    # solve
    if metric == 'two_Norm':
        res = minimize(two_projection, x0, args = occupancy, method='SLSQP', constraints= constraints,
                   options={'disp': True, 'maxiter': 10}, bounds=bounds)
    elif metric == 'KL_divergence':
        res = minimize(KL_projection, x0, args=occupancy, method='trust-constr', constraints=constraints,
                       options={'verbose': 1}, bounds=bounds)

    occupancy = res.x.reshape(num_states, num_actions)
    return occupancy, np.linalg.norm(past_occupancy - occupancy)

def projectedGradientAscent(num_policies, occupancies, objfctn, M, proj_metric):
    """"
         Implementation of the Projected Gradient Ascent algorithm where the constraints are determined by the feasible
         simplex of the MDP 'M'. 'objfctn' is the specified objective function, 'num_policies' is the number of
         policies in the return set, "occupancies" is the initial guess, and "proj_metric" is the distance metric
         for the projection step
    """

    step_size_0 = .01
    iterate = 0
    num_converged = 2
    converge_number = 10
    tol = .01
    while True:
        print("The current iterate is " + str(iterate))
        # terminate if gradient magnitude is small enough
        step_size = step_size_0/(1 + iterate)
        iterate = iterate + 1

        obj_value = objfctn(occupancies)
        obj_value.backward()

        for i in range(0, num_policies):
            grad_mag = 0
            # gradient update
            past_occupancy = occupancies[i].data.clone()
            grad_mag = grad_mag + np.linalg.norm(np.array(occupancies[i].grad), ord=2)
            occupancies[i] = occupancies[i] + step_size * occupancies[i].grad
            # projection back into policy space
            temp, norm_diff = gradProj(occupancies[i].detach().numpy(), M, past_occupancy.numpy(), metric = proj_metric)
            occupancies[i] = torch.tensor(temp, requires_grad = True)
        # return if convergence
        if norm_diff < tol:
            num_converged = 1 + num_converged
            if num_converged >= converge_number:
                break
        else:
            num_converged = 0

        if iterate > 30:
            print('iterate exceeding 30, terminating')
            break


        print(str(torch.norm(occupancies[0] - occupancies[1], p='nuc')))
        print("The magnitude of the gradients is" + str(grad_mag))
    return occupancies, iterate


def FrankWolfe(num_policies, occupancies, objfctn, M):
    """"
       Implementation of the Frank Wolfe algorithm where the constraints are determined by the feasible
       simplex of the MDP 'M'. 'objfctn' is the specified objective function, 'num_policies' is the number of
       policies in the return set, and "occupancies" is the initial guess.
    """

    # first define the bounds and contstraints of the simplex:
    num_states = occupancies[0].shape[0]
    num_actions = occupancies[0].shape[1]
    length_vector = num_states * num_actions

    # nonnegative bound + 0 constraint on non-enabled actions
    Bounds = []
    for i in range(0, num_states):
        for j in range(0, num_actions):
            if M.enabled_actions[i, j]:
                Bounds.append([0, np.inf])
            else:
                Bounds.append([0, 0])


    # add constraints
    constraint_LHS = np.zeros((num_states + 1, length_vector))
    constraint_RHS = np.zeros((num_states + 1))
    # must sum to one
    constraint_LHS[0, :] = np.ones((1, length_vector))
    constraint_RHS[0] = 1
    # flow property
    for i in range(0, num_states):
        rhs = M.transitions[:, :, i].flatten()
        lhs = np.zeros((1, num_states * num_actions))
        for j in range(0, num_actions):
            lhs[0, i * num_actions + j] = 1
        constraint = rhs - lhs
        constraint_LHS[i +1, :] = constraint



    # then solve the minimization problem

    # constants
    iterate = 0
    tol = .001

    # line search params
    gamma = .5
    c_const = .001

    while True:
        print("The current iterate is " + str(iterate))
        obj_value = objfctn(occupancies)
        obj_value.backward()
        d = []
        gt = 0
        past_occupancies = []
        for i in range(0, num_policies):
            past_occupancies.append(occupancies[i].data.clone())
            # compute s that minimizes <s, - grad f(x) >

            c = occupancies[i].grad.numpy().flatten()
            # initial guess
            guess = createRandomPolicy(num_states, num_actions, M)
            x0 = policy2occupancy(guess, num_states, num_actions, M).flatten()
            res = scipy.optimize.linprog(-c, A_eq=constraint_LHS, b_eq=constraint_RHS, bounds= Bounds, method='interior-point',
                                   callback=None, options=None, x0=x0)
            s = res.x
            s = torch.tensor(s)
            s = torch.reshape(s, (num_states, num_actions))
            d_i = s - past_occupancies[i]
            gt = gt + torch.sum(d_i *  occupancies[i].grad)# compute frank wolfe gap
            d.append(d_i)

        if gt < tol: #break if near stationary point (frank wolfe gap is small)
            break


        # Line Search
        t = 1
        new_guess = []
        for i in range(0, num_policies):
            new_guess.append(past_occupancies[i] + d[i])
        new_f = objfctn(new_guess)
        grad_f = 0
        for i in range(0, num_policies):
            grad_f = grad_f + c_const * torch.sum(d_i *  occupancies[i].grad)
        f = objfctn(past_occupancies)
        while new_f < f + t*grad_f:
            t = gamma*t
            new_guess = []
            for i in range(0, num_policies):
                new_guess.append(past_occupancies[i] + t*d[i])
            new_f = objfctn(new_guess)

        occupancies = []
        for i in range(0, num_policies):
            occupancy = torch.tensor(past_occupancies[i] + (t * d[i]), requires_grad=True)
            occupancies.append(occupancy)

        iterate = iterate + 1
        if iterate > 30:
            print('iterate exceeding 30, terminating')
            break

    return occupancies, iterate



def diversePlanning(M, lam, num_policies, optimal_reward, obj_metric= Jensen_Shannon, proj_metric = 'two_Norm', sol_method = 'Frank-Wolfe'):
    """"
    Main code - solves the divere stochastic planning problem for an MDP M with tradeoff parameter lam,
    num_policies in the return set, a specified optimal_reward value ( to compute summary statistics),
    and metrics specified for the pairwise diversity component of the objective function and the projection step (if using Projected
    Gradient ascent)
    """

    num_states = M.transitions.shape[0]
    num_actions = M.transitions.shape[1]

    # initialize the policies randomly
    policies = []
    for l in range(0, num_policies):
        policy = createRandomPolicy(num_states, num_actions, M)
        policies.append(policy)

    # convert policies into state-action occupancy measures, define occupancy measures as tensors
    occupancies = []
    for l in range(0, num_policies):
        occupancy = policy2occupancy(policies[l], num_states, num_actions, M)
        if (np.sum(occupancy) - 1) > np.exp(-5): # sanity check - ensure occupancy map entries sum to one
            print("Occupancy map not created correctly!")
        occupancies.append(torch.tensor(occupancy, requires_grad = True))

    # define objective function
    def objfctn(occupancies, metric = obj_metric):
        f = 0
        for i in range(0, num_policies):
            f = f - (1/num_policies)*(optimal_reward - torch.sum(torch.mul(occupancies[i], torch.tensor(M.rewards))))**2
            for j in range(i + 1, num_policies):
                f = f + lam * (2/(num_policies*(num_policies - 1)))* obj_metric(occupancies[i], occupancies[j])
        return f


    # iterative updates
    time_start = time.perf_counter()
    if sol_method == 'projectedGradAscent':
        occupancies, iterates = projectedGradientAscent(num_policies, occupancies, objfctn, M, proj_metric)
    elif sol_method == 'Frank-Wolfe':
        occupancies, iterates = FrankWolfe(num_policies, occupancies, objfctn, M)
    else:
        raise Exception("Soution method not coded yet")
    time_elapsed = (time.perf_counter() - time_start)

    # compute summary statistics
    rewards = 0
    squared_rewards = 0
    div_score_two = 0
    div_score_js = 0
    for i in range(0, num_policies):
        rewards = rewards + torch.sum(occupancies[i] * torch.tensor(M.rewards))**2
        squared_rewards = squared_rewards - (optimal_reward - torch.sum(torch.mul(occupancies[i], torch.tensor(M.rewards))))**2
        for j in range(i + 1, num_policies):
            div_score_two = div_score_two + two_Norm(occupancies[i], occupancies[j])
            div_score_js = div_score_js + Jensen_Shannon(occupancies[i], occupancies[j])

    return occupancies, div_score_two, div_score_js, rewards, squared_rewards, iterates, time_elapsed


def exactSolve(M, b):
    """"
     finds an optimal policy (without considering diversity) for MDP M using linear program
     and returns it 'b' times
    """

    time_start = time.perf_counter()
    # first define the bounds and contstraints of the simplex:
    num_states = M.transitions.shape[0]
    num_actions = M.transitions.shape[1]
    length_vector = num_states * num_actions
    # nonnegative bound + 0 constraint on non-enabled actions
    Bounds = []
    for i in range(0, num_states):
        for j in range(0, num_actions):
            if M.enabled_actions[i, j]:
                Bounds.append([0, np.inf])
            else:
                Bounds.append([0, 0])
    # add constraints
    constraint_LHS = np.zeros((num_states + 1, length_vector))
    constraint_RHS = np.zeros((num_states + 1))
    # must sum to one
    constraint_LHS[0, :] = np.ones((1, length_vector))
    constraint_RHS[0] = 1
    # flow property
    for i in range(0, num_states):
        rhs = M.transitions[:, :, i].flatten()
        lhs = np.zeros((1, num_states * num_actions))
        for j in range(0, num_actions):
            lhs[0, i * num_actions + j] = 1
        constraint = rhs - lhs
        constraint_LHS[i + 1, :] = constraint
    # random initial guess
    guess = createRandomPolicy(num_states, num_actions, M)
    x0 = policy2occupancy(guess, num_states, num_actions, M).flatten()
    # solve problem
    c = M.rewards.flatten()
    res = scipy.optimize.linprog(-c, A_eq=constraint_LHS, b_eq=constraint_RHS, bounds=Bounds, method='interior-point',
                                 callback=None, options=None, x0=x0)
    occupancy = res.x

    # compute summary statistics
    rewards = 0
    optimal_occupancies = []
    occupancy = np.reshape(occupancy, (-1, 5))
    for i in range(0, b):
        optimal_occupancies.append(occupancy)
    rewards = b * np.sum(optimal_occupancies[i] * M.rewards)
    iterates = 1
    time_elapsed = (time.perf_counter() - time_start)

    return optimal_occupancies, rewards, iterates, time_elapsed


def test_performance(grid_path, num_trials):
    """"
    loads grid world from specified file. Then runs the diverse stochastic planning algorithm
    for a range of specified lambda values or number of paths + number of trials.
    """

    with open(grid_path, 'rb') as grid_world_file:
        grid_worlds = pickle.load(grid_world_file)
    grid_worlds = grid_worlds[0:num_trials]

    lam_values = np.linspace(30, 50, num = 5) # range of lambda values to test
    b = 6 # number of paths to return

    # solve setup exactly to determine optimal reward
    optimal_rewards = np.zeros(num_trials)
    for idx_trial, grid_world in enumerate(grid_worlds):
        _, optimal_reward,_ , _ = exactSolve(grid_world, b);
        optimal_rewards[idx_trial] = optimal_reward/b;


    # store results and learned occupancy maps
    results = np.zeros((len(lam_values), num_trials, 6))
    occupancy_maps = {}
    num_iterates = np.zeros((len(lam_values), num_trials))
    comp_time = np.zeros((len(lam_values), num_trials))
    # compute average diversity score (two norm),average diversity score (JS norm),
    # average performance, and average objective function value (two norm + JS norm),
    # then variance of all five of these metrics
    for idx_lam, lam in enumerate(lam_values):
        for idx_trial, grid_world in enumerate(grid_worlds):
            if lam == 0 or b == 1: # use exct linear programming if lambda = 0 or b =1
                # solve exactly
                optimal_occupancies, rewards, iterates, time_elapsed = exactSolve(grid_world, b)
                div_score_two = 0
                div_score_js = 0
            else:
                # solve using diverse planning algorithm
                optimal_occupancies, div_score_two, div_score_js, rewards, squared_rewards, iterates, time_elapsed = diversePlanning(grid_world, lam, b, optimal_rewards[idx_trial], obj_metric= Jensen_Shannon, proj_metric = 'two_Norm')

            print("The Lambda Value is: " + str(lam) + ", the trial is: " + str(idx_trial) + ", the Jensen-Shannon Divergence: " + str(div_score_js))
            # save results
            obj_value_two = (lam * div_score_two) + rewards
            obj_value_js = (lam * div_score_js) + rewards
            results[idx_lam, idx_trial, 0:6] = [div_score_two, div_score_js, rewards, squared_rewards, obj_value_two, obj_value_js]
            occupancy_maps.update({(lam, idx_trial): optimal_occupancies})
            num_iterates[idx_lam, idx_trial] = iterates
            comp_time[idx_lam, idx_trial] = time_elapsed

    return results, b, num_iterates, comp_time, lam_values, occupancy_maps, grid_worlds

def main(grid_path, save_grids, results_path, save_results, occupancy_path, save_occupancy_maps, num_trials):
    """"
    runs test performance to get occupancy maps and results for the given problem setup and specified
    number of trials 'num_trials'
    then saves results in the specified files if desired (determined by boolean variables
    save_grids, save_results, save_occupancy_maps
    """
    results, num_paths, num_iterates, comp_time, lam_values, occupancy_maps, grid_worlds = test_performance(grid_path, num_trials)


    if save_grids:
        with open(grid_path, 'wb') as grid_world_file:
            pickle.dump(grid_worlds, grid_world_file)

    if save_results:
        np.savez(results_path, results = results, num_paths = num_paths, num_iterates = num_iterates, comp_time = comp_time, lam_values = lam_values)

    if save_occupancy_maps:
        with open(occupancy_path, 'wb') as occupancy_file:
            pickle.dump(occupancy_maps, occupancy_file)
