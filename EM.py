"""
Electromagnetism-like Mechanism by Birbil and Fang (2003).

@author: Joost Verlaan
"""

import numpy as np
import os
import gym
from gym import wrappers
import pybullet_envs
import time


class Hp():
    """ Class containing all hyperparameters needed to run the algortihm.
    
    Attributes:
        MAXITER: maximum number of iterations.
        nb_states: dimension of state vector.
        nb_action: dimension of action vector.
        delta: local search parameter, [0,1].
        m: sample size.
        LSITER: maximum number of local search iterations.
        n: total dimension of problem.
        upper_bound: upper bound of matrix of parameters.
        lower_bound: lower bound of matrix of parameters.
        seed: random seed.
        env_name: name of environment.
    """
    def __init__(self):
        self.MAXITER = 1000 
        self.nb_states = 0  # altered at initialization
        self.nb_actions = 0  # altered at initialization
        self.delta = 0.3  
        self.m = 20  
        self.LSITER = 10  # set equal to 0 if local search is not used
        self.n = 0  # altered at initialization
        self.upper_bound = 9.355
        self.lower_bound = -9.355
        self.seed = 901
        self.env_name = 'InvertedDoubleBulletEnv-v0'     
        
def evaluate(theta, x):
    """Calculates the action vector.
    
    Args:
        theta: matrix of parameters.
        x: state vector.
    
    Returns:
        Resulting action vector.
    """
    return theta.dot(x)
    
def explore(env, theta, hp):
    """Explores the environment for a single episode.
    
    Args: 
        env: environment of the task.
        theta: matrix of parameters.
        hp: object containing all parameters.
        
    Returns:
        Reward resulting from a single episode.   
    """
    env.seed(hp.seed)  # reset environment to the same state
    state = env.reset()
    done = False
    sum_rewards = 0
    while not done:
        action = evaluate(theta, state)
        state, reward, done, _ = env.step(action)
        sum_rewards += reward
        
    return sum_rewards


def initialize(env, hp):
    """Initializes the m random solutions.
    
    Args:
        env: environment of the task.
        hp: object containing all parameters.
    
    Returns:
        thetas: 3D numpy array containing the m initialized 
                matrices of parameters.
        rewards: all found rewards for each matrix.
        best_reward: current best reward found.  
    """
    np.random.seed(hp.seed)
    rewards = [0]*hp.m
    thetas = np.zeros((hp.m, hp.nb_actions, hp.nb_states))
    for i in range(hp.m):
        random_theta = hp.lower_bound + \
            np.random.uniform(low=0.0, high=1.0, size=hp.n)* \
            (hp.upper_bound-hp.lower_bound)
        random_theta = random_theta.reshape(hp.nb_actions, hp.nb_states)
        a = explore(env, random_theta, hp) 
        rewards[i] = explore(env, random_theta, hp)        
        thetas[i,:,:] = random_theta
    best_reward = np.max(rewards)
    
    return thetas, best_reward, rewards
       
 
def local(env, thetas, hp, rewards):
    """Performs local search on thetas.
    
    Args:
        env: environment of the task.
        thetas: 3D numpy array containing all matrices of parameters.
        hp: object containing all parameters.
        rewards: all previously found rewards for each matrix.
    
    Returns:
        thetas_new: newly update thetas.
        rewards: new corresponding rewards.
        best_reward: current best reward found.
        index: index of current best reward.    
    """
    length = hp.delta*(hp.upper_bound - hp.lower_bound)  
    thetas_new = np.zeros((hp.m, hp.nb_actions, hp.nb_states))
    for i in range(hp.m):
        x_i = thetas[i,:,:]
        for k in range(hp.n):
            lambda_1 = np.random.rand()
            counter = 0
            while counter < hp.LSITER:
                y = x_i.flatten()
                lambda_2 = np.random.rand()
                if lambda_1 > 0.5:
                    y[k] += lambda_2*length
                else:
                    y[k] -= lambda_2*length
                y = y.reshape(hp.nb_actions, hp.nb_states)
                reward_y = explore(env, y, hp)
                if reward_y >= rewards[i]:
                    x_i = y
                    rewards[i] = reward_y
                    counter = hp.LSITER - 1
                counter += 1
        thetas_new[i,:,:] = x_i
    best_reward = np.max(rewards)
    index = np.where(rewards == best_reward)
    
    return thetas_new, rewards, best_reward, index


def CalcF(thetas, hp, rewards):
    """Calculates the force matrices for each theta.
    
    Args:
        thetas: 3D numpy array containing all matrices of parameters.
        hp: object containing all parameters.
        rewards: all previously found rewards for each matrix.
    
    Returns:
        F: 3D numpy array containing all force matrices.
    """
    best_reward = np.max(rewards)
    denom = sum(rewards - best_reward)
    q = [0]*hp.m
    F = np.zeros((hp.m, hp.nb_actions, hp.nb_states))
    for i in range(hp.m):
        q[i] = np.exp(-hp.n * ((rewards[i]-best_reward)/denom))
    
    for i in range(hp.m):
        for j in range(hp.m):
            if i != j:
                if rewards[j] > rewards[i]:
                    temp1 = q[i]*q[j]
                    temp2 = np.linalg.norm(thetas[j,:,:].flatten() - thetas[i,:,:].flatten())
                    F[i,:,:] += (thetas[j,:,:] - thetas[i,:,:]) * ( float(temp1/temp2) )
                else:
                    temp1 = q[i]*q[j]
                    temp2 = np.linalg.norm(thetas[j,:,:].flatten() - thetas[i,:,:].flatten())
                    F[i,:,:] -= (thetas[j,:,:] - thetas[i,:,:]) * ( float(temp1/temp2) )
    
    return F


def Move(F, thetas, hp, index, iteration):
    """Updates the thetas using the force matrices.
    
    Args: 
        F: 3D numpy array containing all force matrices.
        thetas: 3D numpy array containing all matrices of parameters.
        hp: object containing all parameters.
        index: index of current best reward.
        iteration: current iterations of the algorithm 
                   (used in case of diminishing step size).
    
    Returns:
        thetas_new: 3D numpy array containing all updated 
                    matrices of parameters.
        rewards: new corresponding rewards.
    """
    thetas_new = thetas.copy()
    rewards = [0]*hp.m
    for i in range(hp.m):
        if i != index[0][0]:
            lambda_rand = np.random.rand()
            temp1 = np.linalg.norm(F[i,:,:].flatten())
            F[i,:,:] = F[i,:,:]/temp1
            F_i = F[i,:,:].flatten()
            theta_i = thetas[i,:,:].flatten()
            for k in range(len(theta_i)):
                if F_i[k] > 0:
                    theta_i[k] += lambda_rand*F_i[k]*(hp.upper_bound-theta_i[k])
                else:
                    theta_i[k] += lambda_rand*F_i[k]*(theta_i[k] - hp.lower_bound)
            thetas_new[i,:,:] = theta_i.reshape(hp.nb_actions, hp.nb_states)
        rewards[i] = explore(env, thetas_new[i,:,:], hp)
    
    return thetas_new, rewards
    

def EM(env, hp):
    """Electromagnetism-like Mechanism.
    
    This function performs the complete EM algorithm by calling 
    the previous functions.
    
    Args:
        env: environment of the task.
        hp: object containing all parameters.
    """
    thetas, optimal_reward, rewards = initialize(env, hp)
    iteration = 1
    start = time.time()
    while iteration < hp.MAXITER:
        thetas_new, rewards, optimal_reward, index = local(env, thetas, hp, rewards)
        end = time.time()
        print('Iteration: {}, Reward: {:0.4f}, Elapsed time: {:0.4f}'.format(iteration, optimal_reward, end-start))
        start = time.time()
        F = CalcF(thetas_new, hp, rewards)
        thetas, rewards = Move(F, thetas_new, hp, index, iteration)
        iteration += 1
    


def mkdir(base, name):
    """Creates directory for the clips of the tasks.
    
    Args:
        base: base of the directory.
        name: name of the created map.
    
    Returns:
        path: path to the directory.   
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    
    return path


# Create directory for clips
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

hp = Hp()
np.random.seed(hp.seed)

# Create gym environment
env = gym.make(hp.env_name)
env = wrappers.Monitor(env, monitor_dir, force=True)
env.seed(hp.seed)

hp.nb_states = env.observation_space.shape[0] 
hp.nb_actions = env.action_space.shape[0] 
hp.n = hp.nb_states * hp.nb_actions

# Perform algorithm
EM(env, hp)