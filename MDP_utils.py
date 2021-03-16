import sys
import time
from math import isclose
import numpy as np
import copy
import itertools
from scipy.special import comb
from scipy.stats import entropy
from scipy.spatial import distance
import random
#import gurobipy as grb
import pandas as pd
import matplotlib.pyplot as plt

class MC:
    """Generate an MDP object for a problem"""
    
    def __init__(self, origin=None, mdp=None, policy=None, *args, **kwargs):
        
        self.origin = origin
        
        if self.origin == 'mdp':
            self.induced_mc(mdp, policy)
        elif self.origin == 'book':
            self.book_ex()
        elif self.origin == None:
            pass
        else:
            raise NameError("Given Markov chain origin is not supported")
            
    def book_ex(self):
        """Create a test MC from model-checking book Figure 10.1"""
        
        self.states = np.arange(4) # set of states {0 : 'start', 1 : 'try', 2 : 'lost', 3 : 'delivered'}
        self.init_state = 0 # initial state
        self.transitions = np.array([
                [0,1,0,0],
                [0,0,0.1,0.9],
                [0,1,0,0],
                [1,0,0,0]], dtype=np.float64) # transition function
    
    def induced_mc(self, mdp, policy):
        """Generate the inuced MC from an MDP and a policy"""
        
        self.states = mdp.states # set of states
        self.init_state = mdp.init_state # initial state
        self.transitions = np.zeros((len(self.states),len(self.states)), dtype=np.float64) # transition function
        if policy.memory_use:
            raise NameError("Induced Markov chain for policies with memory is not supported")
        else:
            if policy.randomization:
                for state in self.states:
                    assert np.sum(policy.mapping[state]) == 1, "Policy has improper distribution in state %i" % (state)
                    
                    for ind_a, action in enumerate(mdp.actions[mdp.enabled_actions[state]]):
                        self.transitions[state] += [policy.mapping[state][ind_a] *
                                mdp.transitions[state,action,next_state] 
                                for next_state in self.states]
            else:
                for state in self.states:
                    action = policy.mapping[state]
                    assert mdp.enabled_actions[state,action], \
                           "Policy takes invalid action %i in state %i" % (action,state)
                    self.transitions[state] = [mdp.transitions[state,action,next_state] 
                                               for next_state in self.states]
    
    def make_absorbing(self, state):
        """Make a state of the MC absorbing"""

        self.transitions[state,:] = np.zeros(len(self.states)) # remove all transitions
        self.transitions[state,state] = 1 # add self transition
        

class Policy:
    """Generate a policy object for an MDP"""
    
    def __init__(self, mdp, randomization=False, memory_use=False, mapping=None):
        
        self.mdp = mdp
        self.randomization = randomization
        self.memory_use = memory_use
        if mapping == None:
            self.mapping = dict()
        else:
            self.mapping = mapping
            
    def unif_rand_policy(self):
        """Generate a random policy with uniform distribution"""
        
        self.randomization = True
        self.mapping = {s : np.full(np.sum(self.mdp.enabled_actions[s]),
                                    1.0/np.sum(self.mdp.enabled_actions[s]))
                        for s in self.mdp.states}
        
# note that mapping is only defined for the enabled actions
#        self.mapping = {s : np.array([1.0/np.sum(self.mdp.enabled_actions[s]) 
#                if self.mdp.enabled_actions[s,a] 
#                else 0 for a in self.mdp.actions]) 
#                for s in self.mdp.states}
            
    def rand_policy(self):
        """Generate a random policy"""
        
        if self.memory_use:
            raise NameError("Random policy with memory is not defined")
        else:
            if self.randomization:
                self.mapping = dict()
                for s in self.mdp.states:
                    self.mapping[s] = np.diff(np.concatenate(
                            ([0], np.sort(np.random.uniform(0,1,
                             np.sum(self.mdp.enabled_actions[s])-1)),[1])))
            else:
                self.mapping = {s : np.random.randint(0,np.sum(self.mdp.enabled_actions[s])) 
                                for s in self.mdp.states}
                
    def take_action(self, state, memory=None):
        """Select a single action according to the policy"""
        
        if self.memory_use:
            pass
        else:
            if self.randomization:
                action = np.random.choice(self.mdp.actions[self.mdp.enabled_actions],
                                          p=self.mapping[state])
                next_state = np.random.choice(self.mdp.states, 
                                              p=self.mdp.transitions[state,action])
            else:
                action = self.mapping[state]
                next_state = np.random.choice(self.mdp.states, 
                                              p=self.mdp.transitions[state,action])
                
        return (action, next_state)
    
    def simulate(self, init_state, n_step):
        """Simulate trajectory realizations of a policy"""
        
        trajectory = np.empty(n_step+1, dtype=np.int32)
        trajectory[0] = init_state
        action_hist = np.empty(n_step, dtype=np.int32)
        
        for step in range(n_step):
            (action_hist[step], trajectory[step+1]) = self.take_action(trajectory[step])
        
        return (trajectory, action_hist)
        
    def verify_trajectory(self, trajectory, spec):
        """Check whether a single trajectory satisfies a specification"""
        
        if len(spec) == 2:
            # reach-avoid specification
            assert len(set.intersection(set(spec[0]),set(spec[1]))) == 0, \
                   "The specification creates conflict"
            reach_steps = []
            for r in spec[1]:
                if len(np.nonzero(trajectory==r)[0]) > 0:
                    reach_steps.append(np.nonzero(trajectory==r)[0][0])
            reach_min = min(reach_steps)
            
            if len(reach_steps) == 0:
                return False
            else:
                for t in range(reach_min):
                    if trajectory[t] in spec[0]:
                        return False
                return True
        
        else:
            raise NameError("Given specification is not handled")
            
    def evaluate(self, spec, s_current):
        """Evaluate a policy with respect to a specification"""
        
        ind_mc = MC(origin='mdp', mdp=self.mdp, policy=self)
        (vars_val,_) = verifier(copy.deepcopy(ind_mc), spec)
        
        return vars_val[s_current]
    

class MDP:
    """Generate an MDP object for a problem"""
    
    def __init__(self, problem_type, *args, **kwargs):
        
        self.problem_type = problem_type
        
        if self.problem_type == 'simple':
            self.simple_problem()
        elif self.problem_type == 'book':
            self.book_ex()
        elif self.problem_type == 'gridworld':
            self.gridworld_2D()
        else:
            raise NameError("Given problem type is not supported")
        
    def simple_problem(self):
        """Create a simple MDP with small size"""
        
        self.states = np.arange(3) # set of states
        self.init_state = 0 # initial state
        self.actions = np.arange(3) # set of actions
        self.action_mapping = {0 : 'a', 1 : 'b', 2 : 'c'}
        self.enabled_actions = np.array([[1,1,0],[1,0,1],[0,1,1]], dtype=np.bool) # enabled actions in states
        self.transitions = np.array([
                [[0,1,0],[0,0,1],[0,0,0]],
                [[1,0,0],[0,0,0],[0,1,0]],
                [[0,0,0],[1,0,0],[0,0,1]]], dtype=np.float64) # transition function
    
    def book_ex(self):
        """Create a test MDP from model-checking book Figure 10.21"""
        
        self.states = np.arange(8) # set of states
        self.init_state = 0 # initial state
        self.actions = np.arange(2) # set of actions
        self.action_mapping = {0 : 'alpha', 1 : 'beta'}
        self.enabled_actions = np.array([[1,1],
                                         [1,0],
                                         [1,0],
                                         [1,0],
                                         [1,0],
                                         [1,1],
                                         [1,0],
                                         [1,0]], dtype=np.bool) # enabled actions in states
        self.transitions = np.zeros((len(self.states),len(self.actions),
                                     len(self.states)), dtype=np.float64) # transition function    
        self.transitions[0,0,[1]] = [1]; self.transitions[0,1,[2,4]] = [1/3,2/3]
        self.transitions[1,0,[1,2,3]] = [1/2,7/18,1/9]
        self.transitions[2,0,[2]] = [1]
        self.transitions[3,0,[3]] = [1]
        self.transitions[4,0,[5,6]] = [1/4,3/4]
        self.transitions[5,0,[6]] = [1]; self.transitions[5,1,[2,7]] = [1/3,2/3]
        self.transitions[6,0,[5,6]] = [2/5,3/5]
        self.transitions[7,0,[2,3]] = [1/2,1/2]
        
    def action_effect(self, state, action):
        """Determines the correct and incorrect effect of actions"""
        
        if self.problem_type == 'gridworld':
            incorrect_actions = np.copy(self.enabled_actions[state])
            (s1,s2) = self.state_mapping[state]
            
            if self.enabled_actions[state,action]:
                if action == 0:
                    correct_state = state
                    incorrect_actions[[True, True, True, True, True]] = 0
                    return [correct_state, incorrect_actions]
                elif action == 1:
                    correct_state = (s1-1)*self.dim[1]+s2
                    incorrect_actions[[True, True, False, False, False]] = 0
                    return [correct_state, incorrect_actions]
                elif action == 2:
                    correct_state = s1*self.dim[1]+s2+1
                    incorrect_actions[[True, False, True, False, False]] = 0
                    return [correct_state, incorrect_actions]
                elif action == 3:
                    correct_state = (s1+1)*self.dim[1]+s2
                    incorrect_actions[[True, False, False, True, False]] = 0
                    return [correct_state, incorrect_actions]
                elif action == 4:
                    correct_state = s1*self.dim[1]+s2-1
                    incorrect_actions[[True, False, False, False, True]] = 0
                    return [correct_state, incorrect_actions]
                else:
                    raise NameError("Given action is not defined")
            else:
                return None
        else:
            raise NameError("Given problem type has no defined action effect")
            
    
    def gridworld_2D(self, dim=(8,8), p_correctmove=0.95, init_state=0):
        """Create an MDP for navigation in a 2D gridworld"""
        
        self.dim = dim # dimensions (d1, d2) of the state space
        self.states = np.arange(self.dim[0]*self.dim[1]) # set of states
        self.state_mapping = {i*self.dim[1]+j : (i,j) 
                for i in range(self.dim[0]) 
                for j in range(self.dim[1])}
        self.init_state = init_state # initial state
        self.actions = np.arange(5) # set of actions
        self.action_mapping = {0 : 'stop', 1 : 'up', 2 : 'right', 3 : 'down', 4 : 'left'}
        self.enabled_actions = np.ones((len(self.states),len(self.actions)), dtype=np.bool) # enabled actions in states
        for state, coordinate in self.state_mapping.items():
            if coordinate[0] == 0: # top boundary
                self.enabled_actions[state,1] = 0
            elif coordinate[0] == self.dim[0]-1: # bottom boundary
                self.enabled_actions[state,3] = 0
            if coordinate[1] == 0: # left boundary
                self.enabled_actions[state,4] = 0
            elif coordinate[1] == self.dim[1]-1: # right boundary
                self.enabled_actions[state,2] = 0
                
        self.p_correctmove = p_correctmove # probability of correct execution of action
        self.transitions = np.zeros((len(self.states),len(self.actions),
                                     len(self.states)), dtype=np.float64) # transition function
        for state, coordinate in self.state_mapping.items():
            for action in self.actions[self.enabled_actions[state]]:
                [correct_state, incorrect_actions] = self.action_effect(state, action)
                n_inc_actions = np.sum(incorrect_actions)
                if n_inc_actions == 0:
                    self.transitions[state, action, correct_state] = 1
                else:
                    self.transitions[state, action, correct_state] = self.p_correctmove
                    for inc_action in self.actions[incorrect_actions]:
                        inc_state = self.action_effect(state, inc_action)[0]
                        self.transitions[state, action, inc_state] = (1-self.p_correctmove)/n_inc_actions            

    def semantic_representation(self, property_dist='random', prior_belief='random'):
        """Assign semantics to MDP states"""
        
        if self.problem_type == 'simple':
            self.properties = np.arange(2)
            self.property_mapping = {0 : 'p', 1 : 'q'}
            self.label_true = np.array([[0,0],[0,0],[1,0]], 
                                       dtype=np.bool) # true property labels of states
            self.label_belief = np.array([[0,0],[0,0.7],[0.6,0]], 
                                         dtype=np.float64) # truthness confidence (belief) over property labels
            
        elif self.problem_type == 'book':
            self.properties = np.arange(1)
            self.property_mapping = {0 : 'target'}
            self.label_true = np.zeros((len(self.states),len(self.properties)), 
                                       dtype=np.bool) # true property labels of states
            self.label_true[3,0] = True
            self.label_belief = np.zeros((len(self.states),len(self.properties)), 
                                         dtype=np.float64) # truthness confidence (belief) over property labels
            self.label_belief[3,0] = 0.8; self.label_belief[6,0] = 0.25;
            
        elif self.problem_type == 'gridworld':
            self.properties = np.arange(2)
            self.property_mapping = {0 : 'obstacle', 1 : 'target'}
            
            self.label_true = np.zeros((len(self.states),len(self.properties)), 
                                       dtype=np.bool) # true property labels of states
            if property_dist == 'random':
#                n_obstacle = 10
#                obstacle_pos = np.random.randint(0,len(self.states),n_obstacle)
                obstacle_pos = [6,18,21,23,24,38,39,43,49,60]
                self.label_true[obstacle_pos,0] = 1
#                n_target = 2
#                target_pos = np.random.randint(0,len(self.states),n_target)
                target_pos = 63
                self.label_true[target_pos,1] = 1
                
            self.label_belief = np.zeros((len(self.states),len(self.properties)), 
                                         dtype=np.float64) # truthness confidence (belief) over property labels
            if prior_belief == 'exact':
                self.label_belief[:,:] = self.label_true
            
            elif prior_belief == 'random':
                self.label_belief = 0.5 * np.ones((len(self.states),len(self.properties)), 
                                                  dtype=np.float64)
                
            elif prior_belief == 'noisy-ind':
                noise = 0.25
                for state in self.states:
                    for prop in self.properties:
                        if self.label_true[state,prop]:
                            self.label_belief[state,prop] = 1 - noise
                        else:
                            self.label_belief[state,prop] = noise
                            
            elif prior_belief == 'noisy-dep':
                noise = 0.24
                confusion_level = 1
                for state in self.states:
                    for prop in self.properties:
                        if self.label_true[state,prop]:
                            self.label_belief[state,prop] = 1 - noise
                            neighbors = self.state_neighbor(state, confusion_level)
                            neighbors.remove(state)
                            self.label_belief[[True if s in neighbors 
                                               else False 
                                               for s in self.states],
                                              prop] = noise/len(neighbors)
                        
            else:
                raise NameError("Given prior belief is not defined")
                
        else:
            raise NameError("Given problem type has no defined semantics")
            
    def state_neighbor(self, state, degree):
        """Find neighbors of a state up to a given degree of closeness"""
        
        if self.problem_type == 'gridworld':
            neighbors = {state}
            checked = set()
            for d in range(degree):
                for n in set.difference(neighbors,checked):
                    for a in self.actions[self.enabled_actions[n]]:
                        new_n = self.action_effect(n,a)[0]
                        neighbors.add(new_n)
                    checked.add(new_n)
        else:
            raise NameError("Given problem type has no defined neighbors")
        
        return neighbors
    
    def make_absorbing(self, state):
        """Make a state of the MDP absorbing"""
        
        if self.problem_type == 'simple':
            self.enabled_actions[state] = [0,0,1] # only action "c" is enabled
            self.transitions[state,:,:] = np.zeros((len(self.actions),len(self.states))) # remove all transitions
            self.transitions[state,2,state] = 1 # add transition for "c"
        elif self.problem_type == 'book':
            self.enabled_actions[state] = [1,0] # only action "alpha" is enabled
            self.transitions[state,:,:] = np.zeros((len(self.actions),len(self.states))) # remove all transitions
            self.transitions[state,0,state] = 1 # add transition for "alpha"            
        elif self.problem_type == 'gridworld':
            self.enabled_actions[state] = [1,0,0,0,0] # only action "stop" is enabled
            self.transitions[state,:,:] = np.zeros((len(self.actions),len(self.states))) # remove all transitions
            self.transitions[state,0,state] = 1 # add transition for "stop"
        else:
            raise NameError("Given problem type does not support absorbing states")


if __name__ == "__main__":
    print(str(__file__))