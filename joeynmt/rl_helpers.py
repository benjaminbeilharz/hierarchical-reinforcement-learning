# -*- coding: utf-8 -*-
# Authors: Benjamin Beilharz, Tai Mai and Leander Girrbach
"""
Helper classes/functions for RL-functionality in JoeyNMT.
"""

import numpy as np

import torch
from torch import Tensor
from collections import namedtuple
from random import randrange
from typing import List, Dict


class ReplayMemory:
    """
    Buffer for experience replay
    """
    Transition = namedtuple('Transition',
                            ('state', 'action', 'reward', 'policy'))

    def __init__(self, capacity: int, max_episode_length: int):
        """
        :param capacity: Capacity of replay buffer
        :param max_episode_length: Maximum memory length
        """
        self.capacity = capacity
        self.max_episode_length = max_episode_length
        self.memory = []

    def append(self, state, action, reward, policy):
        """
        Append transition to buffer

        :param state: Current state
        :param action: Action taken
        :param reward: Reward gained
        :param policy: Next state taken
        """
        self.memory.append(self.Transition(state, action, reward,
                                           policy))  # saving s_i, a_i, r_i+1
        if action is None:
            self.memory.append(self.memory)
            self.memory = []

    def sample(self, max_length: int = 0) -> List[Transition]:
        """
        Samples one transition from memory

        :param max_length: max trajectory length
        :returns: List[Transition]
        """
        mem = self.memory[randrange(len(self.memory))]
        T = len(mem)
        if max_length > 0 and T > max_length + 1:
            t = randrange(T - max_length - 1)
            return mem[t:t + max_length + 1]
        else:
            return mem

    def sample_batch(self, batch_size: int, max_length: int):
        """
        Samples a batch of transitions from memory

        :param batch_size: Batch-size
        :param max_length: Max trajectory length
        """
        batch = [self.sample(max_length=max_length) for _ in range(batch_size)]
        minimum_size = min(len(self.memory) for trajectory in batch)
        batch = [self.memory[:minimum_size] for trajectory in batch]
        return list(map(list, zip(*batch)))

    @property
    def length(self):
        return len(self.memory)

    def __len__(self):
        return sum(len(episode) for episode in self.memory)


def hrl_prepare_rewards(rewards: List[float],
                        episode_info: Dict[str, list]) -> List[float]:
    """
    For each high level policy reward, set the reward (initially 0) to
    the sum of all rewards of the following low level policy.

    High level policy rewards are sum of rewards for chosen low level policy.
    This calculation cannot be done by the environment, because low level
    policy rewards are returned after the subtask is chosen, and the
    environment cannot return rewards for past actions.
        
    :param rewards: All rewards (of episode)
    :type rewards:  List[float]
    :param episode_info: Mapping containing all necessary information for
                         each timestep
    :type episode_info:  Dict[str, list]
    :returns:       Manipulated rewards
    """
    last_high_level_index = None
    cum_reward = 0.0

    prepared_rewards = []
    for i, (reward, step) in enumerate(zip(rewards, episode_info)):
        if step['agent_id'] == '4':  # Timestep is high level
            if last_high_level_index is None:  # First high level action
                last_high_level_index = i
                cum_reward = 0.0
            else:
                prepared_rewards[last_high_level_index] = cum_reward
                last_high_level_index = i
                cum_reward = 0.0
        else:
            cum_reward += reward

        prepared_rewards.append(reward)

    # Last high level action has no successor, so we have to set it
    # explicitely
    prepared_rewards[last_high_level_index] = cum_reward
    return prepared_rewards


def evaluate_step(agent, env, sampler, episode_info, obs):
    """
    Evaluates a single step in the episode. To be used by `_train_episode()` 
    and `_validate()`.

    :param env: Environment
    :param sampler: Sampling method for sampling the action from probabilities
    :param episode_info: Contains information about the current episode
    :param obs: Observation from the environment. 
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float()
    action_probabilities = agent(obs)
    #if obs[-1] is None:
    #print(action_probabilities)
    action = sampler(action_probabilities)
    obs, reward, done, info = env.step(action)
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float()
    state = agent.state
    agent_id = agent.identity
    episode_info.add_step(state, agent_id, action_probabilities, action, obs,
                          reward, done, info)

    return obs, episode_info, reward, done


class EpisodeInfo:
    """
    Class for saving infos about training episode.
    """
    def __init__(self):
        self._steps = []

    def __len__(self):
        return len(self._steps)

    def __iter__(self):
        return iter(self._steps)

    def __getitem__(self, index):
        return self._steps[index]

    def add_step(self, state, agent_id, probabilities, action, obs, reward,
                 done, info) -> None:
        step_dict = {
            'state': state,
            'agent_id': agent_id,
            'probabilities': probabilities,
            'action': action,
            'observation': obs,
            'reward': reward,
            'terminal': done,
            'info': info
        }
        self._steps.append(step_dict)

    def clear(self, keep_last=True):
        if keep_last:
            self._steps = self._steps[-1:]
            for key, value in self._steps[-1].items():
                if isinstance(value, Tensor):
                    self._steps[-1][key] = value.detach()
        else:
            self._steps = []


# class OpenAIAtariEnvWrapper:
#     def __init__(self, env, dimensions=(210, 160, 3)):
#         self.env = env
#         self.dimensions = dimensions
#         self.cached_screen = np.zeros(dimensions)
    
    
#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         obs_difference = obs - self.cached_screen
#         self.cached_screen = obs
        
#         return obs_difference, reward, done, info
    
#     def __getattr__(self, name):
#         return getattr(self.env, name)


class OpenAIAtariEnvWrapper:
    def __init__(self, env, preprocessing=False):
        self.env = env
        self.preprocessing = preprocessing
        if self.preprocessing:
            self.dimensions = (80, 80, 1)
        else:
            self.dimensions = (210, 160, 3)
        self.cached_screen = np.zeros(self.dimensions)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self.preprocessing:
            obs = self.preprocess(obs)

        obs_difference = obs - self.cached_screen
        self.cached_screen = obs

        return obs_difference, reward, done, info
    
    def reset(self):
        self.cached_screen = None
        return self.env.reset()
    
    def __getattr__(self, name):
        return getattr(self.env, name)

    def preprocess(self, screen):
        """
        Specifically for Pong. Taken from MinPy examples.
        https://github.com/dmlc/minpy/blob/master/examples/rl/parallel_actor_critic/envs.py
        Preprocess a 210x160x3 uint8 frame into a 6400 (80x80) (1 x input_size) 
        float vector.
        """
        # Crop, down-sample, erase background and set foreground to 1.
        # Ref: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
        # print('l.159 img.size()', img.shape)
        screen = screen[35:195]
        screen = screen[::2, ::2, 0]
        screen[screen == 144] = 0
        screen[screen == 109] = 0
        screen[screen != 0] = 1
        screen = np.expand_dims(screen.astype(np.float), axis=2)
        # print('l.159 img.size()', img.shape)
        return screen

    def reset(self):
        self.cached_screen = np.zeros(self.dimensions)
        screen = self.env.reset()
        if self.preprocessing:
            return self.preprocess(screen)
        else: 
            return screen


class OpenAIPongWrapper:
    def __init__(self, env, dimensions=(210, 160, 3)):
        self.env = env
        self.dimensions = dimensions
        self.cached_screen = None
    
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        """ Preprocess a 210x160x3 uint8 frame into a 6400 (80x80) (1 x input_size) float vector."""
        # Crop, down-sample, erase background and set foreground to 1.
        # Ref: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
        obs = obs[35:195]
        obs = obs[::2, ::2, 0]
        obs[obs == 144] = 0
        obs[obs == 109] = 0
        obs[obs != 0] = 1
        obs = np.expand_dims(obs.astype(np.float).ravel(), axis=0)
        # Subtract the last preprocessed image.
        obs_difference = obs - self.cached_screen if self.cached_screen is not None else np.zeros((1, obs.shape[1]))
        self.cached_screen = obs
        return obs_difference
        
        return obs_difference, reward, done, info
    
    def reset(self):
        self.cached_screen = None
        return self.env.reset()

    def __getattr__(self, name):
        return getattr(self.env, name)
