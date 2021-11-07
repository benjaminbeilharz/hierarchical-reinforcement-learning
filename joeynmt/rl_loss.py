# -*- coding: utf-8 -*-
# Authors: Tai Mai and Leander Girrbach

"""
Defines base classes for all RL-Alorithms. Since we use these algorithms to
calculate differentiable "losses", we call them RL-Losses.
"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import List, Dict, Any, Hashable
from joeynmt.rl_helpers import EpisodeInfo


class RLLoss(nn.Module):
    """
    Base class for RL losses. Subclass of nn.Module, so that submodules
    such as value networks are trainable.
    
    Implements the following common methods:
      * discount_rewards
      * prepare_rewards
      * entropy
    """
    def discount_rewards(self, rewards: List[float],
                         bootstrap_value: float = 0.0) -> List[float]:
        """ Discount rewards """
        running_reward = bootstrap_value
        discounted_rewards = []
        for reward in rewards[::-1]:
            running_reward = reward + self.discount_factor * running_reward
            discounted_rewards.insert(0, running_reward)
        return discounted_rewards

    @staticmethod
    def prepare_rewards(rewards: List[float],
                        episode_info: Dict[str, list]) -> List[float]:
        """ 
        If necessary, manipulate rewards for current task (no-op by default)
        """
        return rewards

    @staticmethod
    def entropy(prob: Tensor) -> Tensor:
        """
        Calculates entropy of a given probability distribution. Returns 0.0
        if entropy is not defined. This can happen, when any probability value
        is exactly zero -> log = -infty.
        
        :param prob: Probability distribution
        :type prob:  Tensor of shape (num_actions,)
        :returns:    Entropy
        """
        entropy = -(prob * torch.log(prob)).sum()
        if torch.isnan(entropy):
            return torch.tensor(0.0, device=prob.device)
        else:
            return entropy
    
    @staticmethod
    def normalise_rewards(rewards: Tensor, eps: float = 1e-6) -> Tensor:
        """
        Normalise rewards to have 0 mean and 1 variance ("whitening").
        Substract mean => 0 mean
        Divide by standard deviation => 1 variance = 1 standard deviation
        
        :param rewards: Rewards to normalise
        :type rewards:  Tensor
        :param eps:     Small constant to avoid dividing by 0
        :type eps:      float
        :returns:       Normalised rewards
        """
        return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)
        


class MultiAgentRLLoss(RLLoss):
    """
    Base class for Multi-Agent-RL losses. Subclass of RLLoss.
    
    Implements the following common methods:
      * forward
      * agent_loss (must be overwritten by subclass)
      * group_by_agent
    """
    def forward(self, episode_info: EpisodeInfo) -> Tensor:
        """
        Extracts relevant information from episode_info. Then calculates
        each per-agent loss individually and returns the sum of all
        per-agent losses.
        
        :param episode_info: EpisodeInfo instance containing all relevant
                             information for calculating per-agent losses.
        :type episode_info:  EpisodeInfo
        :returns:            Sum of all per-agent losses.
        """
        actions = [step['action'] for step in episode_info]
        agent_ids = [step['agent_id'] for step in episode_info]
        states = [step['state'] for step in episode_info]
        probs = [step['probabilities'] for step in episode_info]
        terminals = [step['terminal'] for step in episode_info]
        observations = [step['observation'] for step in episode_info]
        log_probs = [
            torch.log(prob[action]) for prob, action in zip(probs, actions)
            ]
        rewards = [step['reward'] for step in episode_info]
        # Preparing rewards is necessary for tasks like HRL when rewards for
        # certain timesteps are composed of the rewards for other timesteps
        rewards = self.prepare_rewards(rewards, episode_info)
        entropies = [self.entropy(distrib) for distrib in probs]
        
        # Group information by agent
        by_agent = self.group_by_agent(log_probs, rewards, entropies, states,
                                       terminals, observations, agent_ids)
        
        loss = 0.0
        for agent_id, agent_info in by_agent.items():
            # Calculate loss for each involved agent individually
            loss += self.agent_loss(agent_id, agent_info)
        
        return loss
    
    def agent_loss(self, agent_id: Hashable, agent_info: Dict[str, list]) \
            -> Tensor:
        """
        Calculates the per-agent policy gradient loss based on information in
        agent_info. Must be overwritten by subclass.
        
        :param agent_id:   Uniquely identifying the agent
        :type agent_id:    Hashable
        :param agent_info: Mapping containing all relevant information for
                           each timestep.
        :type agent_info:  Dict[str, list]
        :returns:          Loss
        """
        raise NotImplementedError


    def group_by_agent(self, log_probs: List[Tensor], 
                       rewards: List[float],
                       entropies: List[Tensor], states: List[Tensor],
                       terminals: List[bool], 
                       observations: List[Any],
                       agent_ids: List[Hashable]) \
                           -> Dict[Hashable, Dict[str, list]]:
        """
        Group relevant information (logits, rewards, entropies, states,
        terminal (=done), observations follwing the states) by
        agent. Agents are distinguished by agent_ids.
        
        :param log_probs:    Logits of probability distributions over all 
                             actions
        :type log_probs:     List[Tensor]
        :param rewards:      All rewards
        :type rewards:       List[float]
        :param entropies:    Entropies for all action distributions
        :type entropies:     List[Tensor]
        :param states:       States returned by agent for (optionally)
                             calculating baseline
        :type states:        List[Tensor]
        :param terminals:    For each entry, indicates whether the episode
                             finishes ("done" in OpenAi API)
        :type terminals:     List[bool]
        :param observations: Observations received by committing an
                             action in the current state
        :type observations:  List[Any] (Any should be Tensor)
        :param agent_ids:    For each entry, gives the ID of the agent that
                             committed the respective action.
        :type agent_ids:     List[Hashable]
        :returns:            Dictionary mapping agent IDs to dictionaries that
                             contain relevant information only for the 
                             respective agent, e.g. only rewards that the
                             respective agent received.
        """
        values = zip(log_probs, rewards, entropies, states, terminals,
                     observations, agent_ids)
        by_agent = dict()
        for log_prob, reward, entropy, state, done, observation, agent_id \
            in values:
            # See if current Agent ID already registered in by_agent
            try:
                by_agent[agent_id]['log_probs'].append(log_prob)
                by_agent[agent_id]['rewards'].append(reward)
                by_agent[agent_id]['entropies'].append(entropy)
                by_agent[agent_id]['states'].append(state)
                by_agent[agent_id]['terminal'].append(done)
                by_agent[agent_id]['next_states'].append(observation)
            # If not, create new entry
            except KeyError:
                by_agent[agent_id] = {
                    'log_probs': [log_prob],
                    'rewards': [reward],
                    'entropies': [entropy],
                    'states': [state],
                    'terminal': [done],
                    'next_states': [observation]
                    }
        return by_agent
