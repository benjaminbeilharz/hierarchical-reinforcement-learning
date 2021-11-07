"""
Hierarchical Reinforcement Learning
"""
# Author: Benjamin Beilharz and Leander Girrbach
# -*- coding: utf-8 -*-
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from joeynmt.encoders import Encoder
from joeynmt.agent import HRLBaseAgent
from joeynmt.hrl_env import HRLEnvironment


class SequentialEncoder(Encoder):
    """Bidirectional encoder"""
    def __init__(self,
                 input_size: int,
                 embedding_size: int = 16,
                 hidden_size: int = 128,
                 num_layers: int = 1,
                 rnn_type: str = 'LSTM',
                 bidirectional: bool = True):
        """Initialize bidirectional rnn encoder

        :param input_size: vocab size
        :type: int
        :param embedding_size: embedding dimensions
        :type: int
        :param hidden_size: amount of hidden states
        :type: int
        :param num_layers: stacked number of layers
        :type: int
        :param rnn_type: RNN type, either lstm or gru
        :type: str
        :param bidirectional: bidirectional rnn
        :type: bool
        """
        super(SequentialEncoder, self).__init__()
        self.nlayers = num_layers * 2 if bidirectional else num_layers
        self.hidden = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        rnn = nn.GRU if rnn_type == 'gru' else nn.LSTM
        self.rnn = rnn(embedding_size,
                       hidden_size,
                       num_layers,
                       batch_first=True,
                       bidirectional=bidirectional)
        self._output_size = hidden_size * 2 if bidirectional else hidden_size

    def _init_hidden(self,
                     batch_size: int = 1) -> (torch.Tensor, torch.Tensor):
        """Initializes h_0 and c_0

        :param batch_size: input batch size
        :returns: initial hidden/cell state
        """
        h = c = torch.zeros(self.nlayers, batch_size, self.hidden)
        return h, c

    def forward(self, text):
        h_0, c_0 = self._init_hidden()
        h_0 = h_0.to(text.device)
        c_0 = c_0.to(text.device)
        emb = F.dropout(self.embedding(text).unsqueeze(0), p=0.3)
        _, (h, _) = self.rnn(emb, (h_0, c_0))

        num_directions = 2 if self.rnn.bidirectional else 1
        last_hidden = h.view(self.rnn.num_layers, num_directions, 1,
                             self.rnn.hidden_size)
        last_hidden_directions = [
            last_hidden[-1, 0, :, :], last_hidden[-1, 1, :, :]
        ]
        last_hidden = torch.cat(last_hidden_directions, dim=-1).squeeze(0)
        #pred = h[-2:, :].transpose(1,
        #0).contiguous().view(-1,
        #self.hidden).squeeze(0)
        return last_hidden
        # return torch.cat((pred[0], pred[1]))


class HighLevelAgent(HRLBaseAgent):
    """High-level agent manages low-level agents and their corresponding subtasks
    It declares the order of how to subtasks are processed.
    """
    def __init__(self,
                 low_level_agents: List[HRLBaseAgent],
                 hidden_dim: int = 128,
                 output_dim: int = 4,
                 device: torch.device = torch.device('cuda')
                 if torch.cuda.is_available() else torch.device('cpu')):
        """High-level agent initialization

        :param low_level_agents: list of low-level agents to coordinate
        :type: list
        :param hidden_dim: hidden dimension of low-level agents
        :type: int
        :param output_dim: number of low-level agents
        :type: int
        """
        self.input_dim = sum(agent.hidden_dim
                             for agent in low_level_agents) + 4
        self.output_dim = output_dim
        super(HighLevelAgent, self).__init__(input_dim=self.input_dim,
                                             hidden_dim=hidden_dim,
                                             output_dim=output_dim,
                                             agent_level='high',
                                             device=device)
        self.low_level_agents = nn.ModuleList(low_level_agents)
        self.reset()  # initialize all `Agent` attributes
        self.acting = False

    def forward(self, obs):
        subtask_id = obs[-1]
        text = obs[0]
        if isinstance(text, torch.Tensor):
            text = text.to(self.device)

        # Check whether recipe description was already received
        if self.start_episode:
            assert text is not None
            self.current_recipe = text

            # Pass recipe description to low level agents
            for agent in self.low_level_agents:
                agent.current_recipe = text

            self.start_episode = False

        # Update subtask_id and finish subtasks
        if self.current_subtask_id != subtask_id:
            if self.current_subtask_id is not None:
                condition = self.completed_subtasks.clone().detach()
                condition[self.current_subtask_id] = 1
                self.completed_subtasks = torch.where(
                    condition, self.completed_subtasks.new_ones([1]),
                    self.completed_subtasks).to(self.device)
            self.current_subtask_id = subtask_id

        # High-Level Policy
        if subtask_id is None:
            self.acting = True
            
            scores = self.mlp(self.state)
            scores = torch.where(self.completed_subtasks,
                                 scores.new_full([1], float('-inf')),
                                 scores).to(self.device)
            return self.prediction(scores)

        # Pass on to low-level policy
        else:
            self.acting = False
            states = []
            for i, agent in enumerate(self.low_level_agents):
                if i == subtask_id:
                    states.append(None)
                else:
                    states.append(agent.state)
            return self.low_level_agents[subtask_id](obs, states)

    @property
    def state(self):
        """Retrieves current state for high-level agent

        :returns: high-level agent state
        """

        return torch.cat([agent.state for agent in self.low_level_agents] + \
            [self.completed_subtasks.detach().float().clone()]).to(self.device)


class LowLevelAgent(HRLBaseAgent):
    """Low-level agent for an individual subtask"""
    def __init__(self,
                 recipe_understanding_module: Encoder,
                 user_understanding_module: Encoder,
                 hidden_dim: int,
                 output_dim: int,
                 w_d: float = 0.4,
                 device: torch.device = torch.device('cuda')
                 if torch.cuda.is_available() else torch.device('cpu')):
        """
        :param recipe_understanding_module: Encoder for recipe understanding
        :type: Encoder
        :param user_understanding_module: Encoder for user answer understanding
        :type: Encoder
        :param hidden_dim: hidden dimensionality
        :type: int
        :param output_dim: number of actions given the subtask from `HRLEnvironment`
        :type: int
        :param w_d: Interpolation setting
        :type: float
        """
        self.device = device
        self.input_dim = 5 * hidden_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        super(LowLevelAgent, self).__init__(input_dim=self.input_dim,
                                            hidden_dim=hidden_dim,
                                            output_dim=output_dim,
                                            agent_level='low',
                                            device=device)
        self.recipe_understanding_module = recipe_understanding_module
        self.user_understanding_module = user_understanding_module
        self.w_d = w_d
        self.reset()  # initialize all `Agent` attributes

    def forward(self, obs, states):
        text = obs[0].to(self.device) if obs[0] is not None \
                else obs[0]  # User answer (may be empty)
        # Check whether recipe is available
        if self.current_recipe is None:
            raise AssertionError("Recipe can't be None")

        # Check whether recipe is yet to be encoded
        if self.current_recipe_encoding is None:
            self.current_recipe_encoding = \
                self.recipe_understanding_module(self.current_recipe)

        # If user answer given, extend already received answers
        if isinstance(text, torch.Tensor) and not self.start_episode:
            self.current_user_answer = torch.cat(
                [self.current_user_answer, text], dim=0).to(self.device)
            user_answer_encoding = self.user_understanding_module(
                self.current_user_answer)

        # If answers received:
        if bool(len(self.current_user_answer)):
            v_i = (
                1 - self.w_d
            ) * self.current_recipe_encoding + self.w_d * user_answer_encoding

        # If no anwers received, only use recipe encoding
        else:
            v_i = (1 - self.w_d) * self.current_recipe_encoding

        subtask_id = obs[-1]
        states[subtask_id] = v_i
        states = torch.cat(states).to(self.device)

        state_encoding = self.mlp(states).to(self.device)
        self.current_state_repr = state_encoding
        self.start_episode = False

        return self.softmax(self.prediction(state_encoding))

    @property
    def recipe_description(self):
        return self.current_recipe

    @recipe_description.setter
    def recipe_description(self, description):
        if description and isinstance(description, list) and \
            all(isinstance(word, str) for word in description):
            self.current_recipe = description

        else:
            raise TypeError("{} not a valid description".format(description))

    @property
    def state(self):
        return self.current_state_repr


class HRLAgent(nn.Module):
    """Hierarchical Agent that manages `LowLevelAgent` and `HighLevelAgent`"""
    def __init__(self,
                 agents: List[LowLevelAgent],
                 env: HRLEnvironment,
                 device: torch.device = torch.device('cuda')
                 if torch.cuda.is_available() else torch.device('cpu')):
        """
        :param agents: list of low level agents for subtask
        :type: List[Agent]
        :param env: reinforcement learning environment
        :type: HRLEnvironment
        :param device: device to make computations on
        :type: torch.device
        """
        super(HRLAgent, self).__init__()
        self.device = device
        self.env = env
        self.vocab_size = env.vocab_size
        self.agent_id = {}
        self.low_level_agents = agents
        for i, agent in enumerate(agents):
            self.agent_id[str(agent)] = i
        self.high_level_agent = HighLevelAgent(self.low_level_agents,
                                               device=device).to(device)
        self.agent_id['high_level_agent'] = 4

    def __call__(self, obs) -> torch.Tensor:
        """
        Retrieves action probabilities

        :returns: `torch.Tensor` with action probabilities
        """
        return self.high_level_agent(obs)

    def reset(self):
        """
        Resets all agents states by calling `HighLevelAgent.reset()`
        """
        self.high_level_agent.reset(reset_all=True)

    @property
    def id_state_dict(self) -> dict:
        """
        Returns subtask agents with their ids

        :returns: id - model
        """
        d = {str(4): self.high_level_agent.input_dim}
        for low_lvl_agent_id in list(self.agent_id.values())[:-1]:
            d[str(low_lvl_agent_id
                  )] = self.low_level_agents[low_lvl_agent_id].hidden_dim
        return d

    @property
    def id_action_dict(self) -> dict:
        d = {str(4): self.high_level_agent.output_dim}
        for low_lvl_agent_id in list(self.agent_id.values())[:-1]:
            d[str(low_lvl_agent_id
                  )] = self.low_level_agents[low_lvl_agent_id].output_dim
        return d

    @property
    def identity(self) -> int:
        """
        Returns unique id for current agent

        :returns: unique id
        """
        if self.high_level_agent.acting:
            return str(self.agent_id['high_level_agent'])
        else:
            return str(self.agent_id[str(self.low_level_agents[
                self.high_level_agent.current_subtask_id])])

    @property
    def state(self) -> Tensor:
        """
        Returns last active state

        :returns: Tensor
        """
        if self.high_level_agent.acting:
            return self.high_level_agent.state
        else:
            return self.low_level_agents[
                self.high_level_agent.current_subtask_id].state


def build_agent(
    cfg: dict,
    env: HRLEnvironment,
    warmstart: bool = True,
    pretrained_path: str = '{}/models/'.format(os.getcwd())
) -> HRLAgent:
    """Set up for hierarchical agent

    :param cfg: yaml-parsed configuration
    :type: dict
    :param env: environment to act in
    :type: HRLEnvironment
    :param warmstart: load pre-trained models from state dictionary
    :type: bool
    :param pretrained_path: filepath to pre-trained models
    :type: str

    :returns: Hierarchical Agent
    """
    #Read necessary parameters for HRL Agent and build
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    rnn_type = cfg["encoder"].get("rnn_type", "LSTM")
    hidden_size = cfg["encoder"].get("hidden_size", 128)
    embeddings_size = cfg["encoder"].get("embedding_dim", 16)
    num_layers = cfg["encoder"].get("nlayers", 1)
    bidirectional = cfg["encoder"].get("bidirectional", True)
    interpolation = cfg['encoder'].get('interpolation', 0.4)
    num_subtasks = cfg.get("num_subtasks", 4)
    env = HRLEnvironment()
    low_level_agents = []

    for i in range(num_subtasks):
        # initialize subtask agents
        recipe_encoder = SequentialEncoder(input_size=env.vocab_size,
                                           embedding_size=embeddings_size,
                                           hidden_size=hidden_size,
                                           rnn_type=rnn_type,
                                           num_layers=num_layers,
                                           bidirectional=bidirectional)
        answer_encoder = SequentialEncoder(input_size=env.vocab_size,
                                           embedding_size=embeddings_size,
                                           hidden_size=hidden_size,
                                           rnn_type=rnn_type,
                                           num_layers=num_layers,
                                           bidirectional=bidirectional)
        low_level_agents.append(
            LowLevelAgent(recipe_understanding_module=recipe_encoder,
                          user_understanding_module=answer_encoder,
                          hidden_dim=hidden_size,
                          output_dim=env.low_level_action_space[i].n,
                          w_d=interpolation).to(device))

    # warmstart = True
    if warmstart:
        print("Warmstart")
        for i, low_level_agent in enumerate(low_level_agents):
            name = "agent" + str(i) + ".pth"
            path = os.path.join(pretrained_path, name)
            low_level_agent.load_state_dict(
                torch.load(path, map_location=device))
            low_level_agent.train()

    return HRLAgent(low_level_agents, env, device=device).to(device)
