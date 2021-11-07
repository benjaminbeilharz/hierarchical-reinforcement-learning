# -*- coding: utf-8 -*-
# Author: Benjamin Beilharz
"""
This module contains all value networks used for reinforcement learning.
Value networks learn to evaluate the current state.
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from torch import Tensor


class ValueNetwork(ABC):
    """
    Base class for typing
    """
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass
        """
        raise NotImplementedError('Please redefine this in your subclasses.')


class AtariValueNet(ValueNetwork, nn.Module):
    """
    Atari value network
    """
    def __init__(self,
                 input_dim: Tensor,
                 hidden_size: int,
                 output_size: int = 1,
                 kernel_size: int = 4,
                 device: torch.device = torch.device('cuda')
                 if torch.cuda.is_available() else torch.device('cpu')):
        """
        :attr input_dim: Tensor to derive dimensions
        :attr hidden_size: hidden_size of convolution layers
        :attr output_size: By default a scalar
        :attr kernel_size: Kernel size of convolution layers
        :attr device: Device to train model on
        """

        nn.Module.__init__(self)
        # add batch_size if not a 4d tensor
        self.input_dim = input_dim.size(
        ) if input_dim.ndim == 4 else input_dim.unsqueeze(0).size()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.current_state = None
        self.device = device
        self.fc_dim = ((self.input_dim[1] // kernel_size) *
                       (self.input_dim[2] // kernel_size))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim[-1],
                      out_channels=hidden_size,
                      kernel_size=kernel_size,
                      padding=2,
                      padding_mode='replicate'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=kernel_size),
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=4,
                      kernel_size=kernel_size,
                      padding=2,
                      padding_mode='replicate'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=kernel_size),
        )
        self.fc = nn.Sequential(nn.Linear(self.fc_dim, output_size),
                                nn.Softmax(dim=-1))

    def forward(self, state: Tensor) -> Tensor:
        state.to(self.device)
        state /= 127.5
        state -= 1
        self.current_state = state.detach().clone()
        batch_size = state.size()[0]
        state = state.permute(0, 3, 2, 1).contiguous()  # N, W, H, C
        state = self.conv(state)
        state = state.contiguous().flatten()  # flatten conv layers
        out = self.fc(state)
        return out if batch_size != 1 else out.view(-1)


def build_general_value_network(input_size: int,
                                hidden_size: int,
                                output_size: int,
                                num_layers: int = 1) -> ValueNetwork:
    """
    Build general feed forward value network
    
    :param input_size: Size of the input (vocabulary size)
    :type input_size: int
    :param hidden_size: Size of the hidden dimension
    :type hidden_size: int
    :param output_size: Size of the output (size of action space)
    :type output_size: int
    :param num_layers: Number of hidden layers
    :type num_layers: int
    :returns: nn.Module
    """
    modules = []
    current_input_size = input_size
    for _ in range(num_layers - 1):
        modules.append(nn.Linear(current_input_size, hidden_size))
        modules.append(nn.ReLU())
        current_input_size = hidden_size
    modules.append(nn.Linear(current_input_size, output_size))

    return nn.Sequential(*modules)
