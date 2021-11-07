# coding: utf-8
import time
import pickle
import random
import warnings
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import torch
from gym import Env
from gym.spaces import Discrete, Space
from torch import Tensor

from joeynmt.constants import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN
from joeynmt.vocabulary import Vocabulary


def prep_data(filepath: str) -> dict:
    pairwise_data = pd.read_csv(filepath, sep='\t')
    src_word2idx = {}
    trg_word2idx = {}

    tokens = [
        ' '.join(pairwise_data['ORIGINAL']), ' '.join(
            pd.concat([
                pairwise_data['TRANSLATION 1'], pairwise_data['TRANSLATION 2']
            ],
                      axis=0))
    ]
    for d in tokens:
        for s in [src_word2idx, trg_word2idx]:
            for k, v in enumerate(d.split()):
                v = v.lower()
                if v in s.keys(): continue
                s[v.lower()] = k

    data = []
    for i, line in pairwise_data.iterrows():
        src = line['ORIGINAL'].lower().split()
        pairs = [
            line['TRANSLATION 1'].lower().split(),
            line['TRANSLATION 2'].lower().split()
        ]
        for i, p in enumerate(pairs):
            ratings = line[4:]
            pref_a = np.array(
                [1 if pref == 'TRANSLATION 1' else 0 for pref in ratings])
            pref_b = np.array(
                [1 if pref == 'TRANSLATION 2' else 0 for pref in ratings])
            if i == 0:
                p_pref = sum(pref_a) / (sum(pref_a) + sum(pref_b))
            else:
                p_pref = sum(pref_b) / (sum(pref_a) + sum(pref_b))

            data.append((src, p, p_pref))

    data = {
        'train': data,
        'dev': data,
        'src_word2idx': src_word2idx,
        'trg_word2idx': trg_word2idx
    }

    pickle.dump(data, open('./data/pairwise.pkl', 'wb'))

    return data


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

    def __call__(self, text: List[str] = None):
        """
        Functor call converts list of tokens to Tensor holding their indices
        w.r.t. the vocabulary, then puts token indices and subtask index in
        tuple.
        :param text:       Tokens 
        :type text:        List[str]
        :param subtask_id: Subtask ID
        :type subtask_id:  int
        :returns:          Tuple containing token indices
        """
        if text is not None:
            text = text2tensor(text, self.vocab)

        return text


class PairwiseEnvironment(Env):
    """
    Environment for pairwise preferences.
    Holds the dataset from Julia Kreuzer's pairwise translations.
    """
    def __init__(self,
                 mode: str = 'train',
                 beta: float = .6,
                 correct_reward: float = 1.,
                 incorrect_reward: float = 0.,
                 shuffle: bool = True,
                 datapath: str = None):
        super(PairwiseEnvironment, self).__init__()

        with open(datapath, 'rb') as f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = pickle.load(f, encoding='utf8')
                self.data = data[mode]
                self.word2idx = data['word2idx']

    #     self.action_space = 2  # pref 1/2/none

        self.vocab = make_vocab(self.word2idx)
        self.vocab_size = len(self.vocab)
        self.make_obs = MakeObservation(self.vocab)
        print(self.make_obs(self.data[0][0][0]))

    #     self.BETA = beta
    #     self.correct_reward = correct_reward
    #     self.incorrect_reward = incorrect_reward
    #     self.reward_range = (min(self.incorrect_reward,
    #                              self.BETA), max(self.correct_reward,
    #                                              self.BETA))

    # def __len__(self) -> int:
    #     return len(self.data)

    # @property
    # def action_space(self) -> Discrete:
    #     return self.action_space

    # def step(self, action: int) -> Tuple[Tensor, float, bool, None]:
    #     action_space = self.action_space
    #     assert action in action_space

    #     raise NotImplementedError('Still missing')
    #     return observation, reward, done, info


if __name__ == '__main__':
    prep_data('./data/pairwise.tsv')
    # PairwiseEnvironment(datapath='../data/pairwise.pkl')