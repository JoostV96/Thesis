"""
Augmented Random Search by Mania et al. (2018)

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
        nb_steps: total number of episodes that are performed.
        learning_rate: learning rate of the algorithm, alpha.
        noise: exploration noise, nu.
        nb_directions: total number of directions, N.
        nb_best_directions: number of best reactions that are used, b.
        seed: random seed.
        env_name: name of environment.
    """
    def __init__(self):
        """Inits the class of hyperparameters."""
        self.nb_steps = 1000
        self.learning_rate = 0.02 # alpha
        self.noise = 0.03 # v
        self.nb_directions = 16 # N
        self.nb_best_directions = 8 # b
        assert self.nb_best_directions <= self.nb_directions
        self.seed = 901
        self.env_name = "InvertedDoublePendulumBulletEnv-v0"
        

class Standardizer():
    """Class to standardize the state.
    
    Attributes:
        n: current iteration.
        mean: mean of the state.
        mean_diff: difference between current and last mean.
        var: variance of the state.
    """
    def __init__(self, nb_inputs):
        """Inits the standardizer class."""
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        """Adjusts the mean and the variance when a new state is occured.
        
        Args:
            x: state vector.
        """
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n  # update mean
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)  # update variance

    def standardize(self, x):
        """Normalizes the state vector.
        
        Args:
            x: state vector.
        
        Returns: 
            standardized state vector.            
        """
        return (x - self.mean) / np.sqrt(self.var)
    

class Policy():
    """Class to compute the policy (matrix of parameters).
    
    Attribute:
        theta: matrix of parameters.
    """

    def __init__(self, input_size, output_size):
        """Inits the matrix of parameters, theta, at zero.
        
        Args:
            input_size: dimension of state vector.
            output_size: dimension of action vector.
        """
        self.theta = np.zeros((output_size, input_size))  # 6 x 26 

    def evaluate(self, x, delta=None, direction=None):
        """Calculates the action vector.
        
        Args:
            x: state vector.
            delta: standard normal matrix.
            direction: if "positive" add delta, if "negative" subtract delta.
            
        Returns: 
            Resulting action vector. 
        """
        if direction is None:
            return self.theta.dot(x)
        elif direction == "positive":
            return (self.theta + hp.noise * delta).dot(x)
        else:
            return (self.theta - hp.noise * delta).dot(x)

    def sample_deltas(self):  
        """Sample deltas from standard normal distribution."""
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]

    def update(self, rollouts, sigma_r):  
        """Update matrix of parameters.
        
        Args:
            rollouts: tuple containing reward from the positive adjustment, 
                      reward from the negative adjustment, and delta.
            sigma_r: standard deviation of all rewards.        
        """
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step


def explore(env, standardizer, policy, delta=None, direction=None):
    """Explores the environment for a single episode in a given direction.
    
    Args:
        env: environment of the task.
        standardizer: object to standardize the state vector.
        policy: object containing the matrix of parameters.
        delta: standard normal matrix.
        direction: if "positive" add delta, if "negative" subtract delta.

    Returns: 
        Reward resulting from a single episode. 
    """
    state = env.reset() 
    done = False
    sum_rewards = 0
    while not done:  
        standardizer.observe(state)
        state = standardizer.standardize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        sum_rewards += reward
        
    return sum_rewards


def train(env, policy, standardizer, hp):
    """Trains the model.
    
    Args:
        env: environment of the task.
        policy: matrix of parameters
        standardizer: object to standardize the state vector. 
        hp: object containing all parameters.     
    """
    for step in range(hp.nb_steps):
        start = time.time()
        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions

        # Getting the positive rewards in the positive directions
        for k in range(hp.nb_directions):
            positive_rewards[k] = explore(env, standardizer, policy, 
                            delta=deltas[k], direction="positive")

        # Getting the negative rewards in the negative/opposite directions
        for k in range(hp.nb_directions):
            negative_rewards[k] = explore(env, standardizer, policy, 
                            delta=deltas[k], direction="negative")

        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key=lambda x: scores[x], 
                       reverse=True)[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Updating our policy
        policy.update(rollouts, sigma_r)

        # Printing the final reward of the policy after the update
        reward_evaluation = explore(env, standardizer, policy)
        end = time.time()
        print("Step: {}, Reward: {:0.4f}, Elapsed time: {:0.4f}".format(step, 
              reward_evaluation, end-start))


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

nb_inputs = env.observation_space.shape[0] 
nb_outputs = env.action_space.shape[0]

# Perform algorithm
policy = Policy(nb_inputs, nb_outputs)
standardizer = Standardizer(nb_inputs)
train(env, policy, standardizer, hp)