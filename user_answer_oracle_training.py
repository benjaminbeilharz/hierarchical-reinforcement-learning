# -*- coding: utf-8 -*-

"""
This is a variant of standard training. We only alter the dataset
by appending 1 user answer to every recipe description from the beginning
on.

By this we want to explore the performance limits that could be achieved
on the dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

import numpy as np
import warnings
import pickle

from joeynmt.constants import PAD_TOKEN
from joeynmt.constants import EOS_TOKEN
from joeynmt.vocabulary import Vocabulary
from joeynmt.hrl_env import make_vocab
from joeynmt.hrl_env import  text2tensor
from joeynmt.recipe_classifier import RecipeClassifier
from joeynmt.user_simulator import UserSimulator

from standard_training import StandardTrainer


# You can set your parameters here
SEED = 42
MINIBATCHSIZE = 32
HIDDEN_SIZE = 512
EMBEDDING_DIM = 50
NUMLAYERS = 1
EPOCHS = 30

ASK_USER_LABELS = [251, 876, 218, 458]

USE_CUDA = torch.cuda.is_available()

# Seeding for reproducibility:
# Please be aware that reproducibility is NOT ensured across devices
np.random.seed(SEED)
torch.manual_seed(SEED)

# Define alias for frequently used return types:
BatchInfo = Tuple[Tensor, int, int, int, int]


class Trainer(StandardTrainer):
    """
    Variant of standard trainier.
    Because of the added user answers, the input for each subtask (same recipe
    description) are different. Therefore, we have to apply methods from
    sup pretraining, namely bucketing minibatch samples by subtask.
    """
    def run_batch(self, batch: tuple) -> BatchInfo:
        # Unpack batch
        source, lengths, targets, subtasks = batch
        
        if USE_CUDA:
            source = source.cuda()
            lengths = lengths.cuda()
            targets = targets.cuda()
            subtasks = subtasks.cuda()
        
        # Bucket minibatch samples by subtask
        subtask_buckets = [None, None, None, None]
        
        for subtask in range(4):
            subtask_mask = torch.eq(subtasks, subtask)
            subtask_indices = subtask_mask.nonzero().reshape(-1)
            
            if len(subtask_indices):
                subtask_buckets[subtask] = (source[subtask_indices].contiguous(),
                                            lengths[subtask_indices].contiguous(),
                                            targets[subtask_indices].contiguous()
                                            )
                
        total_loss = torch.tensor(0.0, device=source.device)
        num_correct = 0
        num_asks = 0
        batch_size = source.shape[0]
        
        # Get predictions for each subtask (separately)
        for subtask in range(4):
            # Only run agent if we have samples for current subtask
            if subtask_buckets[subtask] is None:
                continue
            else:
                source, lengths, subtask_targets = subtask_buckets[subtask]
    
            agent = self.agents[subtask]
            predictions = agent(source, lengths).contiguous()
            
            # Here, we can just use supervised loss, asking user is
            # not possible
            subtask_loss = self.criterion(predictions, subtask_targets)
            total_loss += subtask_loss
            
            predicted_labels = torch.argmax(predictions, dim=-1)
            correct_mask = torch.eq(predicted_labels, subtask_targets)
            num_correct += correct_mask.sum().item()
        
        # Unfortunately, we cannot easily measure Accuracy C+F because each
        # combination of recipe and subtask is an inidiviudal sample in the
        # dataset.
        num_correct_cf = 0
    
        return total_loss, 4*num_correct, num_correct_cf, num_asks, batch_size


def load_data(datapath: str = "data/data_with_noisy_user_ans.pkl"):
    print("Unpickling dataset")
    with open(datapath, 'rb') as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pickle.load(f, encoding='latin1')
    
    print("Instantiating UserSimulator")
    user_simulator = UserSimulator(False)

    print("Making vocabulary")
    vocab = make_vocab(data['word_ids'])
    padding_value = vocab.stoi[PAD_TOKEN]
    
    # Make datasets
    print("Preparing dataset")
    datasets = dict()
    
    for mode in ['train', 'dev', 'test']:
        source = []
        targets = []
        subtasks = []
        for dp in data['train']:
            words = dp['words']
            labels = dp['labels']
            label_names = [name.decode('utf-8') for name in dp['label_names']]
        
            for subtask in range(4):
                # Append user answer to every recipe description
                user_answer = user_simulator.ask_user_answer(
                            words, label_names, subtask
                            )
                source.append(words + [EOS_TOKEN] + user_answer)
                targets.append(labels[subtask])
                subtasks.append(subtask)
        
        source = [text2tensor(recipe, vocab) for recipe in source]
    
        source = pack_sequence(source, enforce_sorted=False)
        source, lengths = pad_packed_sequence(source,
                                              batch_first=True,
                                              padding_value=padding_value
                                              )
        source = source.long()
        targets = torch.LongTensor(targets)
        subtasks = torch.LongTensor(subtasks)
        datasets[mode] = DataLoader(
            TensorDataset(source, lengths, targets, subtasks),
            batch_size = MINIBATCHSIZE,
            shuffle = (mode == 'train')
            )
    
    return datasets, vocab


def make_agents(vocab, output_sizes=[251, 876, 218, 458]):
    agent1 = RecipeClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_SIZE, NUMLAYERS,
                              output_sizes[0], padding_idx=vocab.stoi[PAD_TOKEN])
    agent2 = RecipeClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_SIZE, NUMLAYERS,
                              output_sizes[1], padding_idx=vocab.stoi[PAD_TOKEN])
    agent3 = RecipeClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_SIZE, NUMLAYERS,
                              output_sizes[2], padding_idx=vocab.stoi[PAD_TOKEN])
    agent4 = RecipeClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_SIZE, NUMLAYERS,
                              output_sizes[3], padding_idx=vocab.stoi[PAD_TOKEN])
    
    return nn.ModuleList([agent1, agent2, agent3, agent4])


if __name__ == '__main__':
    datasets, vocab = load_data()
    agents = make_agents(vocab)
    trainer = Trainer(agents, vocab, cuda=USE_CUDA)
    trainer.train(datasets['train'], datasets['dev'], 30)
    
    print("\n\n----- Test -----\n")
    trainer.evaluate(datasets['test'])
