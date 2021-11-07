# -*- coding: utf-8 -*-
# Authors: Tai Mai and Leander Girrbach

"""
Implementations of Actor-Critic methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch import Tensor
from typing import List, Dict, Any
from copy import deepcopy
from joeynmt.rl_loss import RLLoss, MultiAgentRLLoss
from joeynmt.rl_helpers import EpisodeInfo


class SingleAgentActorCriticLoss(RLLoss):
    """
    Base class for Actor-Critic losses (single agent).
    
    :param value_net:       The value network
    :type value_net:        nn.Module
    :param discount_factor: The discounting factor to use when calculating
                            discounted rewards (mostly named gamma in
                            literature)
    :type discount_factor:  float
    :param entropy_weight:  Weight multiplied to entropy before adding to loss
    :type entropy_weight:   float
    :param eps:             Small constant to avoid division by 0
    :type eps:              float
    :param whitening:       If true, normalises episode rewards to have 0 mean
                            and 1 variance (after discounting)
    :type whitening:        bool
    :param use_target_networks: Whether to use target networks for calculating
                                the advantage (target networks get updated
                                less frequently than the value networks,
                                therefore enabling a more stable learning
                                process)
    :type use_target_networks:  bool
    :param refresh_target_every: Gives the number of forward calls after
                                 which the target networks are reset to the
                                 current state of the value networks
    :param refresh_target_every: int
    :param bootstrap_from_state_value: If True, initialises the reward with
                                       the state value of the last observation
                                       when discounting rewards (as stated in
                                       Mnih et al. 2016). Otherwise,
                                       initialise reward with 0.
    :type bootstrap_from_state_value: bool
    """
    def __init__(self,
                 value_net: nn.Module,
                 discount_factor: float = 0.95, entropy_weight: float = 0.01,
                 eps: float = 1e-9, whitening: bool = True,
                 use_target_networks: bool = False,
                 refresh_target_every: int = 10,
                 bootstrap_from_state_value: bool = True):
        super(SingleAgentActorCriticLoss, self).__init__()
        self.discount_factor = discount_factor
        self.entropy_weight = entropy_weight
        self.eps = eps
        self.whitening = whitening
        self.bootstrap_from_state_value = bootstrap_from_state_value
        
        # Value net: Must be nn.Module, so it gets registered as parameter
        self.value_net = value_net
        
        # Target networks (optional)
        self.use_target_networks = use_target_networks
        self.refresh_target_every = refresh_target_every
        self.target_update_counter = 0
        
        if self.use_target_networks:
            self.target_net = deepcopy(self.value_net)
            self.target_net.load_state_dict(self.value_net.state_dict())
            for param in self.target_net.parameters():
                param.requires_grad = False
    
    def refresh_target_network(self):
        self.target_net.load_state_dict(self.value_net.state_dict())
    
    
class SingleAgentAdvantageActorCriticLoss(SingleAgentActorCriticLoss):
    """Advantage-Actor-Critic, single agent. According to Mnih et al. 2016)"""
    def forward(self, episode_info: EpisodeInfo) -> Tensor:
        """
        Calculates loss according to the Advantage-Actor-Critic alorithm
        in Asynchronous Methods for Deep Reinforcement Learning (Mnih et al.
        2016) (Algorithm S3). Single-Agent version.
        
        :param episode_info: EpisodeInfo instance containing all relevant
                             information for calculating per-agent losses.
        :type episode_info:  EpisodeInfo
        :returns:            Loss (scalar tensor)
        """
        # Extract relevant information
        actions = [step['action'] for step in episode_info]
        agent_ids = [step['agent_id'] for step in episode_info]
        states = torch.stack([step['state'] for step in episode_info])
        probs = [step['probabilities'] for step in episode_info]
        terminals = [step['terminal'] for step in episode_info]
        observations = [step['observation'] for step in episode_info]
        log_probs = [
            torch.log(prob[action]) for prob, action in zip(probs, actions)
            ]
        log_probs = torch.stack(log_probs)
        entropies = torch.stack([self.entropy(distrib) for distrib in probs])
        rewards = [step['reward'] for step in episode_info]
        # Assuming log-probs are on the device currently worked on
        current_device = log_probs.device
        # Preparing rewards is necessary for tasks like HRL when rewards for
        # certain timesteps are composed of the rewards for other timesteps
        rewards = self.prepare_rewards(rewards, episode_info)
        # If episode if not finished, bootstrap from value for
        # next observation
        if not terminals[-1] and self.bootstrap_from_state_value:
            last_observation = observations[-1].to(current_device)
            bootstrap_value = self.value_net(last_observation).item()
        # Else, initialise reward with 0
        else:
            bootstrap_value = 0.0
        
        discounted_rewards = self.discount_rewards(
            rewards, bootstrap_value=bootstrap_value
            )
        discounted_rewards = torch.tensor(discounted_rewards,
                                          device=current_device).float()
        
        if self.whitening:
            discounted_rewards = self.normalise_rewards(discounted_rewards)
        
        state_values = self.value_net(states)
        state_values = state_values.reshape(-1).contiguous()
        
        if self.use_target_networks:
            target_state_values = self.target_net(states)
            target_state_values = target_state_values.reshape(-1).contiguous()
            advantage = (discounted_rewards - target_state_values.detach())

        else:
            advantage = (discounted_rewards - state_values.detach())
            
        # Policy Gradient
        loss = -log_probs*advantage
        loss -= self.entropy_weight * entropies               # Entropy loss
        # Value loss
        loss += F.mse_loss(state_values, discounted_rewards)
        loss = loss.sum()                                     # Summing up
        
        self.target_update_counter += 1
        if self.target_update_counter == self.refresh_target_every \
            and self.use_target_networks:
            self.refresh_target_network()
            self.target_update_counter = 0

        return loss 


class SingleAgentSimpleActorCriticLoss(SingleAgentActorCriticLoss):
    """Actor-Critic, single agent."""
    def forward(self, episode_info: EpisodeInfo) -> Tensor:
        """
        Calculates loss according to the Advantage-Actor-Critic alorithm
        https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html#actor-critic
        Single-Agent version.
        
        :param episode_info: EpisodeInfo instance containing all relevant
                             information for calculating per-agent losses.
        :type episode_info:  EpisodeInfo
        :returns:            Loss (scalar tensor)
        """
        actions = [step['action'] for step in episode_info]
        states = [step['state'] for step in episode_info]
        probs = [step['probabilities'] for step in episode_info]
        entropies = torch.stack([self.entropy(distrib) for distrib in probs])
        log_probs = [
            torch.log(prob[action]) for prob, action in zip(probs, actions)
            ]
        log_probs = torch.stack(log_probs)
        rewards = [step['reward'] for step in episode_info]
        # Assuming log-probs are on the device currently worked on
        current_device = log_probs.device
        # Preparing rewards is necessary for tasks like HRL when rewards for
        # certain timesteps are composed of the rewards for other timesteps
        rewards = self.prepare_rewards(rewards, episode_info)
        
        discounted_rewards = self.discount_rewards(rewards)
        discounted_rewards = torch.tensor(discounted_rewards, 
                                          device=current_device).float()
        if self.whitening:
            discounted_rewards = self.normalise_rewards(discounted_rewards)
            
        # Calculate state-values
        state_values = self.value_net(torch.stack(states))
        state_values = state_values.reshape(-1).contiguous()

        q_s_a = state_values[:-1].contiguous()
        q_s_prime_a_prime = state_values[1:].contiguous()
        g_t = discounted_rewards[:-1] + self.discount_factor * q_s_prime_a_prime
        delta = g_t - q_s_a
        
        actor_loss = -log_probs[:-1] * delta.detach()  # Policy gradient
        critic_loss = (q_s_a - g_t.detach())**2        # Critic loss
        
        total_loss = actor_loss.sum() + critic_loss.sum()
        total_loss -= self.entropy_weight * entropies.sum()  # Entropy loss
        
        return total_loss


class MultiAgentActorCriticLoss(MultiAgentRLLoss):
    """
    Base class for Actor-Critic losses (multi agent).
    
    :param value_net:       Value networks
    :type value_nets:       nn.ModuleDict
    :param discount_factor: The discounting factor to use when calculating
                            discounted rewards (mostly named gamma in
                            literature)
    :type discount_factor:  float
    :param entropy_weight:  Weight multiplied to entropy before adding to loss
    :type entropy_weight:   float
    :param eps:             Small constant to avoid division by 0
    :type eps:              float
    :param whitening:       If true, normalises episode rewards to have 0 mean
                            and 1 variance (after discounting)
    :type whitening:        bool
    :param use_target_networks: Whether to use target networks for calculating
                                the advantage (target networks get updated
                                less frequently than the value networks,
                                therefore enabling a more stable learning
                                process)
    :type use_target_networks:  bool
    :param refresh_target_every: Gives the number of forward calls after
                                 which the target networks are reset to the
                                 current state of the value networks
    :param refresh_target_every: int
    :param bootstrap_from_state_value: If True, initialises the reward with
                                       the state value of the last observation
                                       when discounting rewards (as stated in
                                       Mnih et al. 2016). Otherwise,
                                       initialise reward with 0.
    :type bootstrap_from_state_value: bool
    """
    def __init__(self, value_nets: nn.ModuleDict,
                 discount_factor: float = 0.95, entropy_weight: float = 0.01,
                 eps: float = 1e-9, whitening: bool = True,
                 use_target_networks: bool = False,
                 refresh_target_every: int = 10,
                 bootstrap_from_state_value: bool = True):
        super(MultiAgentActorCriticLoss, self).__init__()
        self.discount_factor = discount_factor
        self.entropy_weight = entropy_weight
        self.eps = eps
        self.whitening = whitening
        self.bootstrap_from_state_value = bootstrap_from_state_value
        
        # Value nets
        self.value_nets = value_nets
        
        # Target networks (optional)
        self.use_target_networks = use_target_networks
        self.refresh_target_every = refresh_target_every
        self.target_update_counter = 0
        
        if self.use_target_networks:
            self.target_nets = nn.ModuleDict()
            # Copy each value network
            for key, net in self.value_nets.items():
                target_net = deepcopy(net)
                target_net.load_state_dict(net.state_dict())
                # Set target network parameters to not trainable
                for param in target_net.parameters():
                    param.requires_grad = False
                
                self.target_nets[key] = target_net
    
    def refresh_target_networks(self):
        for key, net in self.value_nets.items():
            self.target_nets[key].load_state_dict(net.state_dict())

    
class MultiAgentAdvantageActorCriticLoss(MultiAgentActorCriticLoss):
    """Advantage-Actor-Critic, single agent. According to Mnih et al. 2016)"""
    def agent_loss(self, agent_id, agent_info):
        """
        Calculates loss according to the Advantage-Actor-Critic alorithm
        in Asynchronous Methods for Deep Reinforcement Learning (Mnih et al.
        2016) (Algorithm S3). Multi-Agent version.

        :param agent_id:   Uniquely identifying the agent
        :type agent_id:    Any (must be hashable)
        :param agent_info: Mapping containing all relevant information for
                           calculating the loss.
        :type agent_info:  Dict[str, list]
        :returns:          Loss (scalar tensor)
        """
        # Extract relevant information
        value_net = self.value_nets[agent_id]
        log_probs = torch.stack(agent_info['log_probs'])
        rewards = agent_info['rewards']
        entropies = torch.stack(agent_info['entropies'])
        states = torch.stack(agent_info['states'])
        terminals = agent_info['terminal']
        # Assuming log-probs are on the device currently worked on
        current_device = log_probs.device
        # If episode if not finished, bootstrap from value for
        # next observation
        if not terminals[-1] and self.bootstrap_from_state_value:
            last_observation = agent_info['next_states'][-1].to(current_device)
            bootstrap_value = value_net(last_observation).item()
        # Else, initialise reward with 0
        else:
            bootstrap_value = 0.0
        
        discounted_rewards = self.discount_rewards(
            rewards, bootstrap_value=bootstrap_value
            )
        discounted_rewards = torch.tensor(discounted_rewards,
                                          device=log_probs.device).float()
        if self.whitening:
            discounted_rewards = self.normalise_rewards(discounted_rewards)
        
        # Calculate state values
        state_values = value_net(states)
        state_values = state_values.reshape(-1).contiguous()
        
        if self.use_target_networks:
            target_net = self.target_nets[agent_id]
            target_state_values = target_net(states)
            target_state_values = target_state_values.reshape(-1).contiguous()
            advantage = (discounted_rewards - target_state_values.detach())

        else:
            advantage = (discounted_rewards - state_values.detach())
        
        # Policy gradient 
        agent_loss = -log_probs*advantage
        # Entropy loss
        agent_loss -= self.entropy_weight * entropies
        # Value loss
        agent_loss += F.mse_loss(state_values, discounted_rewards)
        agent_loss = agent_loss.sum()  # Summing up
        
        self.target_update_counter += 1
        if self.target_update_counter == self.refresh_target_every and \
            self.use_target_networks:
            self.refresh_target_networks()
            
        return agent_loss


class MultiAgentSimpleActorCriticLoss(MultiAgentActorCriticLoss):
    """Actor-Critic, multi agent."""
    def agent_loss(self, agent_id, agent_info):
        """
        Calculates loss according to the Advantage-Actor-Critic alorithm
        https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html#actor-critic
        Multi-Agent version.

        :param agent_id:   Uniquely identifying the agent
        :type agent_id:    Any (must be hashable)
        :param agent_info: Mapping containing all relevant information for
                           calculating the loss.
        :type agent_info:  Dict[str, list]
        :returns:          Loss (scalar tensor)
        """
        # Extract relevant information
        log_probs = torch.stack(agent_info['log_probs'])
        rewards = agent_info['rewards']
        entropies = torch.stack(agent_info['entropies'])
        states = agent_info['states']
        # Assuming log-probs are on the device currently worked on
        current_device = log_probs.device

        rewards = torch.tensor(rewards, device=current_device).float()
        
        if self.whitening:
            rewards = self.normalise_rewards(rewards)
        
        # Calculate state-values
        value_net = self.value_nets[agent_id]
        state_values = value_net(torch.stack(states)).reshape(-1)

        q_w_s_a = state_values[:-1]
        q_w_s_prime_a_prime = state_values[1:]
        g_t = rewards[:-1] + self.discount_factor * q_w_s_prime_a_prime
        delta = g_t - q_w_s_a
        
        actor_loss = -log_probs[:-1] * delta.detach()  # Policy gradient
        critic_loss = -delta.detach() * q_w_s_a        # Critic loss
        
        total_loss = actor_loss.sum() + critic_loss.sum()
        total_loss -= self.entropy_weight * entropies.sum()
        
        return total_loss
