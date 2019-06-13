#!/usr/bin/env python

import os, sys
import time
import numpy as np  # "pip install numpy" installs numpy
import random
import math
import csv
from numpy.linalg import inv

import de_R1

import gym
from gym import spaces
from gym.utils import seeding

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

#from callbacks import Callback
from rl.agents.dqn import DQNAgent

from rl.memory import SequentialMemory
from rl.util import *

import argparse
from inspect import currentframe, getframeinfo
from pathlib import Path

class PolicyDebug(object):
    """Abstract base class for all implemented policies.

        Each policy helps with selection of action to take on an environment.

        Do not use this abstract base class directly but instead use one of the concrete policies implemented.
        To implement your own policy, you have to implement the following methods:

        - `select_action`

        # Arguments
        agent (rl.core.Agent): Agent used
    """
    def _set_agent(self, agent):
        self.agent = agent

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def select_action(self, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        """Return configuration of the policy

            # Returns
            Configuration as dict
        """
        return {}

class BoltzmannQPolicy(PolicyDebug):
    """Implement the Boltzmann Q Policy

        Boltzmann Q Policy builds a probability law on q values and returns
        an action selected randomly according to this law.
    """
    def __init__(self, tau=1., clip=(-500., 500.)):
        super(BoltzmannQPolicy, self).__init__()
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        """Return the selected action

            # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

            # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        """Return configurations of BoltzmannQPolicy

            # Returns
            Dict of config
        """
        config = super(BoltzmannQPolicy, self).get_config()
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config


class EpsGreedyQPolicy(PolicyDebug):
    """Implement the epsilon greedy policy

        Eps Greedy policy either:

        - takes a random action with probability epsilon
        - takes current best action with prob (1 - epsilon)
        """
    def __init__(self, eps=.1):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values):
        """Return the selected action

            # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

            # Returns
            Selection action
            """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.random_integers(0, nb_actions-1)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy

            # Returns
            Dict of config
            """
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config

def DE(fun, lbounds, ubounds, budget):
    ENV_NAME = 'ea'

    env = de_R1.DEEnv(fun, lbounds, ubounds, budget)

    nb_actions = env.action_space.n

    # Build a sequential model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(nb_actions, activation = 'linear'))
    print("Model Summary: ",model.summary())

    memory = SequentialMemory(limit=100000, window_length=1)#100000

    # Boltzmann Q Policy
    policy = EpsGreedyQPolicy()

    # DQN Agent: Finally, we configure and compile our agent. You can use every built-in Keras optimizer and even the metrics!
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1e4, target_model_update=1e3, policy=policy, enable_double_dqn = True, batch_size = 64) # nb_steps_warmup >= nb_steps 2000
    # Neural Compilation
    dqn.compile(Adam(lr=1e-4), metrics=['mae'])
    dqn.load_weights('dqn_ea_weights.h5f')

    dqn.test(env, nb_episodes=1, visualize=False)
    return env.best_so_far



