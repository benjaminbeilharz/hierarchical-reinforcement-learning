# -*- coding: utf-8 -*-
"""
This module implements Actor Critic Loss with functionalities for target
networks and experience replay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch import Tensor

from random import randrange
from collections import namedtuple


class ActorCriticLoss(nn.Module):
    """
    Implements Actor-Critic for Reinforcement Learning.
    """
    def __init__(self,
                 value_nets: dict,
                 gamma: float = 0.95,
                 entropy_weight: float = 0.01,
                 use_experience_replay: bool = False,
                 target_nets=None,
                 update_every: int = 64,
                 update_target_every: int = 4) -> None:
        # update_target_every: int = 4,
        # device: torch.device = torch.device('cuda') if \
        #     torch.cuda.is_available() else torch.device('cpu')) -> None:
        """
        :param value_nets: Dictionary of value networks for every agent_id
        :type value_nets: Dictionary
        :param gamma: Discount factor
        :type gamma: float
        :param entropy_weight: Entropy weight
        :type entropy_weight: float
        :param use_experience_replay: Setting for learning to use experience
                                      replay
        :type use_experience_replay: bool
        :param target_nets: Target networks if desired. None by default.
        :type target_nets: dict
        :param update_every: Update value networks after this many episodes
        :type update_every: int
        :param update_target_every: Update value networks after this many 
                                    episodes
        :type update_target_every: int
        """
        super(ActorCriticLoss, self).__init__()

        # self.device = device
        self.gamma = gamma
        self.value_nets = {}
        # for agent_id, value_net in value_nets.items():
        #     self.value_nets[agent_id] = value_net.to(self.device)
        self.value_nets = nn.ModuleDict(value_nets)
        self.entropy_weight = entropy_weight
        self.use_experience = use_experience_replay
        if target_nets is not None:
            self.target_nets = nn.ModuleDict(target_nets)
            # self.target_nets = {}
            # for agent_id, target_net in target_nets.items():
            #     self.target_nets[agent_id] = target_net.to(self.device)

        else:
            self.target_nets = None
        self.update_every = update_every
        self.update_target_every = update_target_every
        self.total_episodes = 0  # used for target net updates

    def forward(self, episode_info: list) -> Tensor:
        """
        Calculates the Actor-Critic Loss.

        :param episode_info: Contains information about every step in the 
                             episode. list of dictionaries
        :type episode_info: list
        :return: loss
        """
        self.total_episodes += 1
        if self.total_episodes % (self.update_every *
                                  self.update_target_every) == 0\
                and self.target_nets is not None:
            for agent_id, target_net in self.target_nets.items():
                target_net.load_state_dict(
                    self.value_nets[agent_id].state_dict())

        ###
        # Policy Loss
        action_probs = [
            step['probabilities'][step['action']] for step in episode_info
        ]
        action_probs = torch.log(torch.stack(action_probs))
        device = action_probs.device

        #q_values = self._get_q_values(episode_info)
        q_values, q_values_next = self._prepare_q_values(episode_info,
                                                         device=device)

        # print('type q_values:', type(q_values))
        # print('type q_values_next:', type(q_values_next))

        q_values = torch.stack(q_values)
        q_values = q_values.to(device)
        # rewards = rewards.to(action_probs.device)
        entropies = [-(prob * torch.log(prob)).sum() for prob in action_probs]
        entropies = [
            torch.tensor(0.0) if torch.isnan(entropy) else entropy
            for entropy in entropies
        ]
        entropy = torch.stack(entropies).sum()
        policy_loss = -torch.sum(action_probs * (q_values.clone().detach()))
        policy_loss -= self.entropy_weight * entropy
        policy_loss = policy_loss.to(device)

        ###
        # Critic Loss
        with torch.no_grad():
            # !!!!!!!!!!!!!!!!
            q_values_next = torch.stack(q_values_next)
            q_values_next = q_values_next.to(device)
        rewards = [step['reward'] for step in episode_info]

        rewards = torch.tensor(rewards).float()
        rewards = rewards.to(device)
        td_error = rewards + self.gamma * q_values_next - q_values
        td_error = td_error.to(device)
        critic_loss = -torch.sum(td_error * q_values)
        critic_loss = critic_loss.to(device)

        loss = policy_loss + critic_loss
        loss = loss.to(device)

        return loss

    def _get_q_values(self, episode_info: list,
                      use_target_nets: bool = True,
                      device: torch.device = torch.device('cuda') \
                              if torch.cuda.is_available() \
                              else torch.device('cpu')) -> list:
        """
        Calculates and returns a list of the episode's Q values.

        :param episode_info: Contains information about every step in the
                             episode
        :type episode_info: list
        :param use_target_nets: Whether or not to use the target network to 
                                obtain the Q values
        :type use_target_nets: bool
        :return: list of Q values at every step in the episode
        """
        q_values = []
        if use_target_nets is True:
            value_nets = self.target_nets
        else:
            value_nets = self.value_nets

        for step in episode_info:
            action = step['action']
            state = step['state']
            state = state.to(device)
            agent_id = step['agent_id']
            value_net = value_nets[agent_id]
            value_net = value_net.to(device)
            q = value_net(state)[action]
            q_values.append(q)
        return q_values

    def _prepare_q_values(self, episode_info: list,
                          device: torch.device = torch.device('cuda') \
                                  if torch.cuda.is_available()
                                  else torch.device('cpu')) -> (list, list):
        """
        Returns the list of the episode's Q values and the list of the Q values
        of the next state.
        To be overridden in a hierarchical reinforcement language setting.

        :param episode_info: contains information of every step in the episode
        :type episode_info: list
        :return: Q values 
        :return: Q values of the next states
        """

        q_values = self._get_q_values(episode_info, device)

        if self.target_nets is not None:
            with torch.no_grad():
                q_values_next = self._get_q_values(episode_info,
                                                   use_target_nets=True)
        else:
            q_values_next = q_values.copy()

            q_values_next.pop(0)
            q_values_next.append(torch.zeros(1)[0])

        return q_values, q_values_next


class HRLActorCriticLoss(ActorCriticLoss):
    """
    Implements Actor-Critic for Hierarchical Reinforcement Learning.
    """
    def _prepare_q_values(self, episode_info,
                device: torch.device = torch.device('cuda') \
                                      if torch.cuda.is_available()
                                      else torch.device('cpu')) -> (list, list):
        """
        Prepares the Q values to include the total reward of a subtask and
        arranges the list of next Q values accordingly so that the next Q value
        of the high level agent is assigned correctly.
        Overrides the method for vanilla reinforcement learning.

        :param episode_info: contains information of every step in the episode
        :type episode_info: list
        :return: Q values
        :return: Q values of the next states
        """

        q_values = self._get_q_values(episode_info, device=device)
        # q_values = q_values.to(device)

        if self.target_nets is not None:
            with torch.no_grad():
                q_values_next = self._get_q_values(episode_info,
                                                   use_target_nets=True)
        else:
            q_values_next = q_values.copy()

        # q_values_next = q_values_next.to(self.device)

        # We rearrange the `q_values_next` in the loop below to reflect the
        # hierarchical structure

        # rearrange the q_values to reflect hierarchy so that the next value of
        # a high level state is the next high level state, not the next low
        # level state

        high_level_steps = []

        # save the timestep ids t of high level steps
        for t, step in enumerate(episode_info):
            if step['agent_id'] is None:
                high_level_steps.append(t)

        # exclude last because the it has no next high level step
        for i in range(len(high_level_steps) - 1):

            # rearrange high level Q values:

            # timestep positions of the current and next high level step
            cur_hl_t = high_level_steps[i]
            next_hl_t = high_level_steps[i + 1]

            # get the Q value for the next high level step
            next_hl_q_value = q_values_next[next_hl_t]

            # replace the current Q value with that next Q value
            q_values_next[cur_hl_t] = next_hl_q_value

            # replace the next Q value at the next timet with a 0. This zero
            # will be overridden by the next iteration (except for the very
            # last because there is no high level t after the last).
            q_values_next[next_hl_t] = torch.zeros(1)[0]

            # rearrange low level Q values:
            # delete first low level t so the subsequent low level ts move
            # up
            del q_values_next[cur_hl_t + 1]
            # insert 0 for the last low level t because there's no low level
            # t after the last
            q_values_next.insert(next_hl_t - 1, torch.zeros(1)[0])

        return q_values, q_values_next
