# -*- coding: utf-8 -*-

"""
Contains sampling strategies (subclasses of torch.nn.Module for allowing
trainable parameters).
"""

import torch
import torch.nn as nn
import numpy as np

from torch import Tensor


class Sampler(nn.Module):
    def forward(self, probabilities: Tensor) -> int:
        """
        Samples a Tensor

        :param probabilities: Tensor used for sampling
        :type: Tensor
        :returns: int
        """
        raise NotImplementedError


class GreedySampler(Sampler):
    """Returns the argmax of a given distribution"""
    def forward(self, probabilities: Tensor) -> int:
        """
        Greedy sampling.
        
        :param probabilities: Probability distribution
        :type probabilities:  Tensor (1-dimension)
        """
        with torch.no_grad():
            return torch.argmax(probabilities, dim=-1).item()


class EpsilonGreedySampler(Sampler):
    """
    With a probability of epsilon, samples from the given categorical
    distribution. Otherwise, samples greedily (argmax).
    
    :param epsilon: Probability of sampling from the given distribution
    :type epsilon:  float
    """
    def __init__(self, epsilon: float = 0.05):
        super(EpsilonGreedySampler, self).__init__()
        self.epsilon = epsilon

    def forward(self, probabilities: Tensor) -> int:
        """Epsilon-greedy sampling"""
        with torch.no_grad():
            if np.random.random() > self.epsilon:
                return torch.argmax(probabilities, dim=-1).item()
            else:
                return torch.distributions.Categorical(
                    probabilities).sample().item()

class DecayingEpsilonGreedySampler(Sampler):
    """
    Like EpsilonGreedySampler, but the probability of sampling from the
    categorical distribution (instead of returning the argmax) is decayed
    by a given factor every number of timesteps.
    
    :param epsilon:      Probability of sampling from the given distribution
    :type epsilon:       float
    :param decay_factor: Factor for decaying epsilon
    :type decay_factor:  float
    :param update_every: Number of calls between updates of epsilon
    :type update_every:  int
    """
    def __init__(self, epsilon: float = 0.05, decay_factor: float = 0.99,
                 update_every: int = 10):
        super(EpsilonGreedySampler, self).__init__()
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.update_every = update_every
        
        self.call_counter = 0

    def forward(self, probabilities: Tensor) -> int:
        """Epsilon-greedy sampling"""
        self.call_counter += 1
        
        if self.call_counter == self.update_every:
            self.epsilon *= decay_factor
            self.call_counter = 0

        with torch.no_grad():
            if np.random.random() > self.epsilon:
                return torch.argmax(probabilities, dim=-1).item()
            else:
                return torch.distributions.Categorical(
                    probabilities).sample().item()


class MultinomialSampler(Sampler):
    """
    Samples from the given categorical distribution.
    """
    def forward(self, probabilities: Tensor) -> int:
        with torch.no_grad():
            # Multinomial doesn't have to instantiate a Distribution object
            action = torch.multinomial(probabilities, 1).item()
            return action
