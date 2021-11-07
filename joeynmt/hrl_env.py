# -*- coding: utf-8 -*-
# Author: Leander Girrbach
"""
Defines the RL environment for Hierarchical RL of If-Then-Recipes
(Yao et al. 2018)
Implements the Openai-gym API.
"""

import time
import pickle
import random
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
from gym import Env
from gym.spaces import Discrete, Space
from torch import Tensor

from joeynmt.constants import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN
from joeynmt.user_simulator import UserSimulator
from joeynmt.vocabulary import Vocabulary


def make_vocab(token2idx: Dict[str, int]) -> Vocabulary:
    """
    Converts token-to-index mapping provided by Yao et al. to JoeyNMT
    vocabulary. Given indices are ignored and tokens alphabetically sorted.

    :param token2idx: Token-to-index mapping
    :type token2idx:  Dict[str, int]
    :returns:         Vocabulary object
    """
    # UNK_TOKEN must be first token because DEFAULT_UNK_ID is zero
    tokens = [UNK_TOKEN] + [word for word in sorted(token2idx.keys())]
    return Vocabulary(tokens=tokens)


def text2tensor(text: List[str], vocab: Vocabulary) -> Tensor:
    """
    Converts a token sequence to indices.

    :param text:  Tokens as strings
    :type:        List[str]
    :param vocab: Vocabulary mapping tokens to indices
    :type vocab:  Vocabulary
    :returns:     PyTorch LongTensor with indices
    """
    idx = []
    # Convert tokens to indices
    for token in text:
        # self.vocab.stoi is DefaultDict returning 0 for unknown
        # tokens
        idx.append(vocab.stoi[token])

    # Add BOS and EOS tokens
    idx.insert(0, vocab.stoi[BOS_TOKEN])
    idx.append(vocab.stoi[EOS_TOKEN])
    return torch.LongTensor(idx)


class MakeObservation:
    """
    Functor that converts recipe description/user answer and subtask index
    into a format understood by the agent (currently: Tuple).

    :param vocab: Vocabulary object mapping tokens to indices
    :type vocab:  Vocabulary
    """
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def __call__(self,
                 text: List[str] = None,
                 subtask_id: int = None) -> Tuple[Tensor, int]:
        """
        Functor call converts list of tokens to Tensor holding their indices
        w.r.t. the vocabulary, then puts token indices and subtask index in
        tuple.
        :param text:       Tokens (recipe description / user answer)
        :type text:        List[str]
        :param subtask_id: Subtask ID
        :type subtask_id:  int
        :returns:          Tuple containing token indices and subtask ID
        """
        if text is not None:
            text = text2tensor(text, self.vocab)

        return (text, subtask_id)


class RecipeSpace(Space):
    """
    Space containing all recipe descriptions. 
    """
    def __init__(self, recipes, make_observation):
        super(RecipeSpace, self).__init__()
        self.recipes = recipes
        self.make_observation = make_observation

        self.subtask_ids = [None, 0, 1, 2, 3]

    def __repr__(self):
        return "RecipeSpace({})".format(len(self.recipes))

    def sample(self):
        recipe = np.random.choice(self.recipes)
        return self.make_observation(text=recipe)

    def contains(self, x):
        text, subtask_id = x
        valid_text = text in self.recipes
        valid_subtask_id = subtask_id in self.subtask_ids
        return valid_text and valid_subtask_id


class HRLEnvironment(Env):
    """
    Environment for hierarchical RL of If-Then recipes.
    Holds the dataset provided by Yao et al. (2018) and implements the
    Openai Gym API. It supports the following methods:
      * step
      * reset 
      * render
      * close
      * seed
    and has the `action_space` property, which gives the possible actions for
    the current subtask.
    
    :param mode:             Which subset (train/dev/test) of the dataset to
                             load
    :type mode:              str
    :param beta:             Amount of (negative) reward to return when asking
                             the user
    :type beta:              float
    :param correct_reward:   Reward returned for correct prediction
    :type correct_reward:    float
    :param incorrect_reward: Reward returned for incorrect prediction
    :type incorrect_reward:  float
    :param shuffle:          Whether to shuffle the description order
                             (default: True)
    :type shuffle:  bool
    :param max_user_answers: Maximum number of user answers for each subtask
    :type max_user_answers:  int
    :param datapath:         Path to data (.pkl) provided by Yao et al.
    :type datapath:          str
    """
    def __init__(self,
                 mode: str = "train",
                 beta: float = 0.8,
                 correct_reward: float = 1.0,
                 incorrect_reward: float = -1.0,
                 shuffle: bool = True,
                 max_user_answers: int = 2,
                 dynamic_user_answers: bool = True,
                 fixed_order: bool=False,
                 datapath: str = "data/data_with_noisy_user_ans.pkl"):
        super(HRLEnvironment, self).__init__()

        with open(datapath, 'rb') as f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # sklearn compains about
                # outdated label encoders
                data = pickle.load(f, encoding='latin1')  # No UTF-8 for Py2
            self.data = data[mode]
            self.word2idx = data["word_ids"]
            # We have one more action, that is for requesting user answers
            self.low_level_action_space = \
                [Discrete(n+1) for n in data['num_labels']]

        self.vocab = make_vocab(self.word2idx)
        self.vocab_size = len(self.vocab)
        # Initialise make_observation functor
        # (works like a method/function)
        self.make_observation = MakeObservation(self.vocab)

        all_recipes = [dp['words'] for dp in self.data]
        self.observation_space = RecipeSpace(all_recipes,
                                             self.make_observation)

        self.shuffle = shuffle
        self.recipe_iterator = self._make_iter()  # Decide for recipe order
        # -> ensures data coverage

        self.high_level_action_space = Discrete(4)  # 4 Subtasks

        self.BETA = beta
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        # -1 is reward for wrong prediction, +1 for correct prediction
        self.reward_range = (
            min(self.incorrect_reward, self.BETA),
            max(self.correct_reward, self.BETA)
            )
        self.max_user_answers = max_user_answers
        
        self.fixed_order = fixed_order
        self.subtask_id = None
        self.is_high_level = True
        self.recipe_description = None
        self.true_channels = None
        self.true_channel_names = None
        self.completed = [False, False, False, False]

        self.user_simulator = UserSimulator(dynamic_user_answers)
        # self.user_simulator = None

    def __len__(self) -> int:
        return len(self.data)

    def _make_iter(self) -> None:
        """
        Decide for order of recipe descriptions -> Data coverage
        This ensures each recipe description has a chance of being returned
        """
        if self.shuffle:
            return iter(np.random.default_rng().permutation(len(self.data)))
        else:
            return iter(np.arange(len(self.data)))

    @property
    def action_space(self) -> Discrete:
        """Return the currently active action space"""
        if self.is_high_level:
            return self.high_level_action_space
        else:
            return self.low_level_action_space[self.subtask_id]

    def step(self, action: int) -> Tuple[Tensor, float, bool, None]:
        """
        Perform the given action and return next observation
        (None or user answer), reward and whether the episode is done
        
        :param action: The action to perform
        :type action:  int
        :returns:      Next observation, reward, done, info=None
        """
        action_space = self.action_space
        if self.is_high_level:
            assert action in action_space
            
            # Replaces High_level policy, when the order is fixed
            if self.fixed_order:
                for st in self.completed:
                    if not st:
                        action = self.completed.index(st)
                        break

            self.subtask_id = action  # Action selects next subtask in
            # high level policy
            self.is_high_level = False  # Jump to low level policy
            self.num_subtask_user_answers = 0  # Not yet requested any user
            # answers

            # Only return next subtask index as observation, because
            # * recipe description is given by self.reset()
            # * no user answer requested (yet)
            observation = self.make_observation(subtask_id=self.subtask_id)
            reward = 0  # high level policy does not receive any reward
            done = False
            info = None

        else:
            assert action in action_space

            # Check if action is requesting user answer
            # Can only ask user for self.max_user_answers times
            if action == action_space.n - 1 and \
                self.num_subtask_user_answers < self.max_user_answers:
                # Ask user simulator
                user_answer = self.user_simulator.ask_user_answer(
                    self.recipe_description, self.true_channel_names,
                    self.subtask_id)
                self.num_subtask_user_answers += 1

                # Return provided answer
                observation = self.make_observation(text=user_answer,
                                                    subtask_id=self.subtask_id)
                reward = -self.BETA  # Punish agent for asking
                done = False
                info = None

            else:
                self.completed[self.subtask_id] = True  # Prediction made
                # -> subtask done
                self.is_high_level = True  # Jump to high level policy
                self.num_subtask_user_answers = 0

                # Check whether given label is correct
                correct = self.true_channels[self.subtask_id] == action

                # High level subtask index: None
                # No user answer
                # => next observation is (None, None)
                observation = self.make_observation()
                reward = self.correct_reward if correct \
                    else self.incorrect_reward
                done = all(self.completed)
                info = None
        
        return observation, reward, done, info

    def reset(self) -> Tuple[Tensor, int]:
        """
        Resets all instance variables
        Returns an initial observation (recipe_description, None)

        :returns: recipe_description for next episode
        """
        recipe_description, true_channel_values, true_channel_names =\
            self.sample_recipe_description()
        self.subtask_id = None
        self.is_high_level = True  # Start in high level policy
        self.completed = [False, False, False, False]
        self.true_channels = true_channel_values
        self.true_channel_names = true_channel_names
        self.recipe_description = recipe_description
        self.num_subtask_user_answers = 0

        return self.make_observation(text=recipe_description)

    def render(self) -> str:
        """Just gives current recipe description as string"""
        return "\nRecipe:\n\t{}\n\n".format(self.recipe_description)

    def close(self) -> None:
        """No-op. Nothing to do, but part of Openai Gym API"""
        pass

    def seed(self, seed: int = None) -> None:
        """Seed numpy"""
        np.random.seed(seed)

    def sample_recipe_description(
            self) -> Tuple[List[str], List[int], List[str]]:
        """
        Gives next recipe description together with (sparse) labels
        and label names according to the current ordering of all recipe
        descriptions. If all recipe descriptions have been returned once,
        decides upon a new ordering.
        """
        try:
            next_index = next(self.recipe_iterator)
        except StopIteration:
            self.recipe_iterator = self._make_iter()
            next_index = next(self.recipe_iterator)

        current_sample = self.data[next_index]
        recipe_description = current_sample['words']
        true_channel_values = current_sample['labels']
        # We love Python 2, but we prefer UNICODE
        true_channel_names = [
            name.decode("utf-8") for name in current_sample['label_names']
        ]
        return recipe_description, true_channel_values, true_channel_names
