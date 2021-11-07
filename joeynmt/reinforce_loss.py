# -*- coding: utf-8 -*-
# Authors: Leander Girrbach

"""
Implementations of the REINFORCE algorithm (Williams 1992) and baselines.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import List, Dict, Any
from joeynmt.rl_loss import RLLoss, MultiAgentRLLoss
from abc import ABC, abstractclassmethod


class Baseline(ABC, nn.Module):
    """Abstract class implementing the Baseline interface"""
    def __init__(self):
        ABC.__init__(self)
        nn.Module.__init__(self)
    
    @abstractclassmethod
    def get_loss(self, baseline_values: Tensor, discounted_rewards: Tensor) -> Tensor:
        """For calculating the loss if the baseline is trainable"""
        raise NotImplementedError
    

class MultiAgentAverageRewardBaseline(Baseline):
    """
    Average reward baseline:
    For each Agent, store the rewards encountered up to the
    current episode and return the average reward per step for
    the agent.
    
    No trainable parameters.
    """
    def __init__(self):
        super(MultiAgentAverageRewardBaseline, self).__init__()
        self.agent_reward_dict = dict()
        self.agent_count_dict = dict()

    def forward(self, rewards: Tensor, states: List[Tensor], agent_id: Any) -> Tensor:
        """
        Calculates baseline values for all rewards based on agent ID.
        
        :param rewards:  Vector containing reward values
        :type rewards:   Tensor of shape (num_rewards,)
        :param states:   Ignored (for compatibility only)
        :type states:    List[Tensor]
        :param agent_id: Hashable that uniquely identifies the agent that
                         received the rewards
        :type agent_id:  Any
        :returns:        Tensor with baseline values, shape: (num_rewards,)
        """
        baseline_values = []
        self.current_device = rewards.device
        cum_reward = rewards.sum().item()  # Total reward in episode
        # If agent already registered, add rewards to stored rewards
        try:
            agent_total_reward = self.agent_reward_dict[agent_id]
            agent_total_predictions = self.agent_count_dict[agent_id]
            # Average reward / step:
            baseline_value = agent_total_reward / agent_total_predictions
            # All rewards in current episode get same baseline
            baseline_values = [baseline_value for _ in range(len(rewards))]
            # Update accumulated rewards
            self.agent_reward_dict[agent_id] += cum_reward
            self.agent_count_dict[agent_id] += len(rewards)

        # Else, register agent and add reward
        except KeyError:
            baseline_values = [0.0 for _ in range(len(rewards))]
            self.agent_reward_dict[agent_id] = cum_reward
            self.agent_count_dict[agent_id] = len(rewards)

        return torch.tensor(baseline_values, device=self.current_device).float()

    def get_loss(self, baseline_values: Tensor, discounted_rewards: Tensor) -> Tensor:
        """Not trainable"""
        return torch.tensor(0.0, device=self.current_device)


class SingleAgentAverageRewardBaseline(Baseline):
    """
    Average reward baseline:
    Store the rewards encountered up to the current episode and return the
    average reward per step.
    
    No trainable parameters.
    """
    def __init__(self):
        super(SingleAgentAverageRewardBaseline, self).__init__()
        self.cumulative_reward = 0.0
        self.num_rewards = 0

    def forward(self, rewards: Tensor, states: List[Tensor]) -> Tensor:
        """
        Calculates baseline values for all rewards.
        
        :param rewards:  Vector containing reward values
        :type rewards:   Tensor of shape (num_rewards,)
        :param states:   Ignored (for compatibility only)
        :type states:    List[Tensor]
        :returns:        Tensor with baseline values, shape: (num_rewards,)
        """
        baseline_values = []
        self.current_device = rewards.device
        cum_reward = rewards.sum().item()  # Total reward in episode
        
        # Average reward / step:
        try:
            baseline_value = self.cumulative_reward / self.num_rewards
        except ZeroDivisionError:
            baseline_value = 0.0
        # All rewards in current episode get same baseline
        baseline_values = [baseline_value for _ in range(len(rewards))]
        # Update accumulated rewards
        self.cumulative_reward += cum_reward
        self.num_rewards += len(rewards)

        return torch.tensor(baseline_values, device=self.current_device).float()

    def get_loss(self, baseline_values: Tensor, discounted_rewards: Tensor) -> Tensor:
        """Not trainable"""
        return torch.tensor(0.0, device=self.current_device)


class MultiAgentValueBaseline(Baseline):
    """
    Trainable MLP for estimating the reward

    :param models: Value networks
    :type models:  nn.ModuleDict
    """
    def __init__(self, models: nn.ModuleDict):
        super(MultiAgentValueBaseline, self).__init__()
        self.models = models

    def forward(self, rewards: Tensor, states: List[Tensor], agent_id: Any) -> Tensor:
        # Calculate Q-Value for agent + state:
        return self.models[agent_id](torch.stack(states)).reshape(-1)

    def get_loss(self, baseline_values: Tensor, discounted_rewards: Tensor) -> Tensor:
        return discounted_rewards*baseline_values



class SingleAgentValueBaseline(Baseline):
    """
    Trainable MLP for estimating the reward

    :param model: Value network
    :type model:  nn.Module
    """
    def __init__(self, model: nn.Module):
        super(SingleAgentValueBaseline, self).__init__()
        self.hidden_size = hidden_size
        self.model = model

    def forward(self, rewards: Tensor, states: List[Tensor]) -> Tensor:
        # Calculate Q-Value for state:
        return self.model(torch.stack(states)).reshape(-1)

    def get_loss(self, baseline_values: Tensor, discounted_rewards: Tensor) -> Tensor:
        return discounted_rewards*baseline_values


class SingleAgentREINFORCELoss(RLLoss):
    """
    Implements the REINFORCE algorithm (Williams 1992).
    Single Agent version.
    
    :param discount_factor: Factor for discounting future rewards
    :type discount_factor:  float
    :param entropy_weight:  Weight of entropy loss (smaller means less impact)
    :type entropy_weight:   float
    :param eps:             Small constant to avoid division by zero
    :type eps:              float
    :param whitening:       If True, normalises rewards per agent to have 0
                            mean and 1 variance
    :type whitening:        bool
    :param baseline:        Baseline (substracted from rewards)
    :type baseline:         Baseline
    """
    def __init__(self,
                 discount_factor: float=0.95,
                 entropy_weight: float=0.01,
                 eps: float=1e-9,
                 whitening:bool =True,
                 baseline: Baseline =None):
        super(SingleAgentREINFORCELoss, self).__init__()
        self.discount_factor = discount_factor
        self.entropy_weight = entropy_weight
        self.eps = eps
        self.whitening = whitening

        # Use baseline if provided
        self.baseline = baseline
        self.apply_baseline = baseline is not None
        
        self.current_device = 'cpu'

    def forward(self, episode_info: Dict[str, list]) -> Tensor:
        """
        Calculates policy gradient loss as defined by the REINFORCE algorithm
        (optionally using a baseline).
        
        :param episode_info: Mapping containing all necessary information for
                             each timestep:
                              * Action (int)
                              * Probability distribution over all possible
                                actions (Tensor)
                              * Reward (float)
                              * State (-> calculating value baseline) (Tensor)
        :type episode_info:  EpisodeInfo (from rl_training, works like dict)
        :returns:            Scalar Tensor containing the loss
        """
        self.current_device = episode_info[0]['probabilities'].device
        actions = [step['action'] for step in episode_info]
        states = [step['state'] for step in episode_info]
        probs = [step['probabilities'] for step in episode_info]
        log_probs = [
            torch.log(prob[action]) for prob, action in zip(probs, actions)
        ]
        log_probs = torch.stack(log_probs)
        entropies = torch.stack([self.entropy(distrib) for distrib in probs])
        rewards = [step['reward'] for step in episode_info]
        # Preparing rewards is necessary for tasks like HRL when rewards for
        # certain timesteps are composed of the rewards for other timesteps
        rewards = self.prepare_rewards(rewards, episode_info)

        discounted_rewards = self.discount_rewards(rewards)
        discounted_rewards = torch.tensor(discounted_rewards,
                                          device=self.current_device)
        if self.whitening:
            discounted_rewards = self.normalise_rewards(discounted_rewards)

        baseline_loss = torch.tensor(0.0, device=self.current_device)
        if self.apply_baseline:
            baseline_values = self.baseline(discounted_rewards, states)
            # Substract baseline values from each reward
            discounted_rewards -= baseline_values
            discounted_rewards = discounted_rewards.detach()
            # Calculate loss for training the baseline (if trainable)
            baseline_loss = self.baseline.get_loss(baseline_values,
                                                   discounted_rewards)

        loss = -log_probs * discounted_rewards   # Policy Gradient
        loss -= self.entropy_weight * entropies  # Entropy loss
        loss -= baseline_loss                    # Training baseline
        loss = loss.sum()                        # Summing up
        return loss


class MultiAgentREINFORCELoss(MultiAgentRLLoss):
    """
    Implements the REINFORCE algorithm (Williams 1992).
    Note that this implementation enables a multi-agent setting, requiring
    the ID of every agent associated with each reward. Rewards are grouped
    by agent and discounted for each agent individually.
    
    :param discount_factor: Factor for discounting future rewards
    :type discount_factor:  float
    :param entropy_weight:  Weight of entropy loss (smaller means less impact)
    :type entropy_weight:   float
    :param eps:             Small constant to avoid division by zero
    :type eps:              float
    :param whitening:       If True, normalises rewards per agent to have 0
                            mean and 1 variance
    :type whitening:        bool
    :param baseline:        Baseline (substracted from rewards)
    :type baseline:         Baseline
    """
    def __init__(self,
                 discount_factor: float=0.95,
                 entropy_weight: float=0.01,
                 eps: float=1e-9,
                 whitening:bool =True,
                 baseline: Baseline =None):
        super(MultiAgentREINFORCELoss, self).__init__()
        self.discount_factor = discount_factor
        self.entropy_weight = entropy_weight
        self.eps = eps
        self.whitening = whitening

        # Use baseline if provided
        self.baseline = baseline
        self.apply_baseline = baseline is not None
        
        self.current_device = 'cpu'


    def agent_loss(self, agent_id: Any, agent_info: Dict[str, list]) -> Tensor:
        """
        Calculates REINFORCE policy gradient loss for a single agent.
        
        :param agent_id:   Uniquely identifying the agent
        :type agent_id:    Any
        :param agent_info: Mapping containing all relevant information for
                           each timestep:
                             * log_probs: Logits of probability distribution
                                          over all possible actions (Tensor)
                             * rewards: Rewards (float)
                             * entropies: Entropy of each probability
                                          distribution over all actions
                                          (Tensor)
                             * states: Inputs to baseline (Tensor)
                                       States are either the inputs to the
                                       agent or (in the case of parameter
                                       sharing) some intermediate
                                       representation returned by the agent to
                                       calculate value scores.
        :type agent_info: Dict[str, list]
        :returns:         Loss
        """
        log_probs = torch.stack(agent_info['log_probs'])
        entropies = torch.stack(agent_info['entropies'])
        states = agent_info['states']
        # Assuming log_probs are on the device currently worked on
        self.current_device = log_probs.device
        
        rewards = agent_info['rewards']

        discounted_rewards = self.discount_rewards(rewards)
        #print(discounted_rewards)
        discounted_rewards = torch.tensor(discounted_rewards,
                                          device=self.current_device)
        if self.whitening:
            discounted_rewards = self.normalise_rewards(discounted_rewards)
        #print(discounted_rewards)

        baseline_loss = torch.tensor(0.0, device=self.current_device)
        if self.apply_baseline:
            baseline_values = self.baseline(discounted_rewards, states,
                                            agent_id)
            # Substract baseline values from each reward
            discounted_rewards -= baseline_values
            discounted_rewards = discounteded_rewards.detach()
            # Calculate loss for training the baseline (if trainable)
            baseline_loss = self.baseline.get_loss(baseline_values,
                                                   discounted_rewards)
        
        #print(baseline_values)
        #print(log_probs)
        #print(entropies)
        #print(discounted_rewards)
        #print(baseline_loss)
        #print()
        #time.sleep(0.5)

        agent_loss = -log_probs * discounted_rewards   # Policy Gradient
        agent_loss -= self.entropy_weight * entropies  # Entropy loss
        agent_loss -= baseline_loss                    # Training baseline
        agent_loss = agent_loss.sum()                  # Summing up
        return agent_loss
