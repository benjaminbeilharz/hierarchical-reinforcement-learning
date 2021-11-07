# -*- coding: utf-8 -*-
# Authors: Benjamin Beilharz, Leander Girrbach and Shaptarshi Roy
"""
This module contains the agent interface and some implementations that can
be used with used for reinforcement learning on OpenAI environments.
Policy networks learn to choose the best action given the current state.
"""

import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from itertools import cycle
from torch import Tensor


class Agent(ABC):
    """Abstract agent interface"""
    @abstractmethod
    def reset(self) -> None:
        """Resets the agent (clears all cached information)"""
        raise NotImplementedError

    @property
    @abstractmethod
    def state(self) -> Tensor:
        """Returns the current state (for Actor-Critic/Value baseline)"""
        raise NotImplementedError

    @property
    @abstractmethod
    def identity(self) -> Any:
        """
        Returns a hashable, unique object identifying the last (sub-)agent
        that was last called.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def id_action_dict(self) -> Dict[Any, int]:
        """
        Returns a mapping of (Sub-)agent IDs to their respective output
        dimensions.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def id_state_dict(self) -> Dict[Any, int]:
        """
        Returns a mapping of (sub-)agent IDs to their respective state sizes.
        """
        raise NotImplementedError


class ClassicControlAgent(Agent, nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int,
                 identifier: Any = 'ClassicControlAgent',
                 device: torch.device = torch.device('cuda')
                 if torch.cuda.is_available() else torch.device('cpu')):
        nn.Module.__init__(self)
        Agent.__init__(self)

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        modules = []
        current_input_size = input_size
        for _ in range(num_layers):
            modules.append(nn.Linear(current_input_size, hidden_size))
            modules.append(nn.ReLU())
            current_input_size = hidden_size
        modules.append(nn.Linear(current_input_size, output_size))

        self.mlp = nn.Sequential(*modules)
        self.mlp = self.mlp.to(device)
        self.softmax = nn.Softmax(dim=-1)

        self.identifier = identifier
        self.current_state = None

    def forward(self, observation):
        observation = observation.to(self.device)
        self.current_state = observation.detach().clone()
        return self.softmax(self.mlp(observation))

    def reset(self):
        self.current_state = None

    @property
    def state(self):
        return self.current_state

    @property
    def identity(self):
        return self.identifier

    @property
    def id_action_dict(self):
        return {self.identifier: self.output_size}

    @property
    def id_state_dict(self):
        return {self.identifier: self.input_size}


class AtariAgent(Agent, nn.Module):
    """
    Policy Network for Atari
    """
    def __init__(self,
                 input_dim: Tensor,
                 hidden_size: int,
                 output_size: int,
                 kernel_size: int,
                 identifier: Any = 'AtariAgent',
                 device: torch.device = torch.device('cuda')
                 if torch.cuda.is_available() else torch.device('cpu')):
        """
        :attr input_dim: Tensor to derive dimensions, `env.reset()`
        :attr hidden_size: Hidden size of CNN
        :attr output_size: Number of actions
        :attr kernel_size: Kernel size
        :attr identifier: Agent Identity
        :attr device: Device to run model on
        """
        nn.Module.__init__(self)
        Agent.__init__(self)

        self.device = device
        # ensure that input dimension is 4d Tensor
        input_dim = torch.tensor(input_dim).detach()
        self.input_dim = input_dim.size(
        ) if input_dim.ndim == 4 else input_dim.unsqueeze(0).size()
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc_dim = ((self.input_dim[1] // kernel_size) *
                       (self.input_dim[2] // kernel_size))

        self.conv1 = nn.Conv2d(in_channels=self.input_dim[-1],
                               out_channels=hidden_size,
                               kernel_size=kernel_size,
                               padding=2,
                               padding_mode='replicate')
        self.conv2 = nn.Conv2d(in_channels=hidden_size,
                               out_channels=3,
                               kernel_size=kernel_size,
                               padding=2,
                               padding_mode='replicate')
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(self.fc_dim, output_size)
        self.softmax = nn.Softmax(dim=-1)

        self.current_state = None
        self.dimensions = (3, 210, 160)
        self.identifier = identifier

    def forward(self, screen: Tensor) -> Tensor:
        # screen = torch.from_numpy(screen).float()
        screen /= 127.5
        screen -= 1
        screen = screen.to(self.device)
        self.current_state = screen.detach().clone()
        batch_size = screen.shape[0]
        screen = screen.permute(0, 3, 2, 1).contiguous()  # N, C, W, H
        screen = self.avg_pool(torch.relu(self.conv1(screen)))
        screen = self.avg_pool(torch.relu(self.conv2(screen)))
        screen = screen.contiguous().flatten()
        pred = self.softmax(self.fc(screen))

        if batch_size == 1:
            pred = pred.reshape(-1)
        return pred

    def reset(self):
        self.current_state = None

    @property
    def state(self):
        return self.current_state

    @property
    def identity(self):
        return self.identifier

    @property
    def id_action_dict(self):
        return {self.identifier: self.output_size}

    @property
    def id_state_dict(self):
        return {self.identifier: self.dimensions}

class PreprocessingAtariAgent(Agent, nn.Module):
    def __init__(self,
                 input_dims: Tuple,
                 hidden_size: int,
                 output_size: int,
                 identifier: Any = 'NewAtariAgent',
                 device: torch.device = torch.device('cuda')
                 if torch.cuda.is_available() else torch.device('cpu')):
        nn.Module.__init__(self)
        Agent.__init__(self)

        input_channels = input_dims[-1]

        self.device = device
        # self.hidden_size = hidden_size
        # self.output_size = output_size

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=16,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        self.fc1 = nn.Linear(12800, 256)
        self.fc2 = nn.Linear(256, 2)
        self.max_pool = nn.MaxPool2d(2)
        self.softmax = nn.Softmax(dim=-1)

        self.current_state = None
        self.dimensions = (1, 80, 80)
        self.identifier = identifier

    def forward(self, screen: Tensor) -> Tensor:
        # screen = torch.from_numpy(screen).float()
        screen /= 127.5
        screen -= 1
        screen = screen.to(self.device)
        # print('159 screen size', screen.size())
        self.current_state = screen.detach().clone()
        if screen.ndim == 3:
            screen = screen.unsqueeze(0)
        batch_size = screen.shape[0]
        screen = screen.permute(0, 3, 2, 1).contiguous()
        screen = self.max_pool(torch.relu(self.conv1(screen)))
        screen = self.max_pool(torch.relu(self.conv2(screen)))
        screen = torch.relu(self.conv3(screen))
        screen = screen.reshape(batch_size, -1).contiguous()
        screen = torch.relu(self.fc1(screen))
        pred = self.softmax(self.fc2(screen))

        if batch_size == 1:
            pred = pred.reshape(-1)
        return pred

    def reset(self):
        self.current_state = None

    @property
    def state(self):
        return self.current_state

    @property
    def identity(self):
        return self.identifier

    @property
    def id_action_dict(self):
        return {self.identifier: self.output_size}

    @property
    def id_state_dict(self):
        return {self.identifier: self.dimensions}


class CNNAgent(nn.Module):
    """
    CNN Agent based on paper: http://cscubs.cs.uni-bonn.de/2018/proceedings/paper_1.pdf
    """
    def __init__(self,
                 input_dim: Tensor,
                 hidden_size: int = 16,
                 output_size: int = 2,
                 kernel_size: int = 3,
                 device: torch.device = torch.device('cuda')
                 if torch.cuda.is_available() else torch.device('cpu')):
        """
        :attr input_dim: Tensor to derive dimensionality
        :attr hidden_size: Hidden size of CNN
        :attr output_size: Binary prediction of pixel
        :attr kernel_size: Convolution kernel size
        :attr device: Training device
        """
        super(CNNAgent, self).__init__()

        self.device = device
        self.current_state = None
        # ensures 4d Tensor
        self.input_dim = input_dim.size(
        ) if input_dim.ndim == 4 else input_dim.unsqueeze(0).size()
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc_dim = ((self.input_dim[-1] / kernel_size) *
                       (self.input_dim[-2] / kernel_size))

        self.conv = nn.Sequential([
            nn.Conv2d(self.input_dim[1], hidden_size, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_size * 4, hidden_size * 4, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_size * 4, hidden_size * 2, kernel_size)
        ])
        self.fc = nn.Sequential([
            nn.Linear(self.fc_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Softmax(dim=-1)
        ])

    def forward(self, screen):
        screen = screen.to(self.device)
        self.current_state = screen.detach().clone()
        batch_size = screen.shape[0]
        screen = screen.permute(0, 3, 2, 1).contiguous()  # N, w, h, c
        screen = self.conv(screen)
        screen = screen.contiguous().flatten()
        pred = self.fc(screen)
        if batch_size == 1:
            pred = pred.reshape(-1)
        return pred


class HRLBaseAgent(nn.Module):
    """
    HRLBaseAgent class for Hierarchical Reinforcement Learning with low-level and high-level agent
    Based on paper: https://arxiv.org/pdf/1808.06740.pdf
    """
    def __init__(self,
                 agent_level: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 device: torch.device = torch.device('cuda')
                 if torch.cuda.is_available() else torch.device('cpu')):
        """
        :attr agent_level: either high or low
        :type: str
        :attr input_dim: input dimension size
        :type: int
        :attr hidden_dim: hidden dimension size
        :type: int
        :attr output_dim: output dimension size
        :type: int
        :attr device: device to run agent on
        :type: torch.device
        """
        super(HRLBaseAgent, self).__init__()
        self._agent_level = agent_level  #high/low level agent
        self.device = device

        # determines which type of mlp and prediction is used depending on agent_level
        if agent_level == "low":
            self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                     nn.Tanh())

            self.prediction = nn.Linear(hidden_dim, output_dim)
            self.softmax = nn.Softmax(dim=-1)

        if agent_level == "high":
            self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                     nn.Tanh(),
                                     nn.Linear(hidden_dim, output_dim))

            self.prediction = nn.Softmax(dim=-1)

        self.reset()

    def forward(self):
        raise NotImplementedError('Define forward pass in subclasses.')

    def _init_sublevel_network(self, *network: List[nn.Module],
                               agent_level: str):
        """
        Initializes sublevel network given agent_level

        :param network: container of nn.Modules, aka. the network
        :type: List[nn.Module]
        :param agent_level: either low or high
        :type: str
        """
        if agent_level == 'low':
            self.net = nn.Sequential(*network)
        elif agent_level == 'high':
            self.net = nn.Sequential(*network)

    def reset(self, reset_all: bool = False):
        """
        Resets (all) instance variables
        
        :param reset_all: resets all instances of the agents recursively
        :type: bool
        :default: False
        """
        if self._agent_level == "high":
            self.completed_subtasks = torch.zeros(4,
                                                  device=self.device,
                                                  requires_grad=False).byte()
            self.current_subtask_id = None
            self.current_recipe = None
            self.start_episode = True
            self.order = cycle([0, 1, 2, 3])

            if reset_all:
                for agent in self.low_level_agents:
                    agent.reset()

        elif self._agent_level == "low":
            self.current_recipe = None
            self.current_recipe_encoding = None
            self.current_user_answer = torch.LongTensor().to(self.device)
            self.current_state_repr = torch.zeros(self.hidden_dim,
                                                  device=self.device)
            self.start_episode = True
