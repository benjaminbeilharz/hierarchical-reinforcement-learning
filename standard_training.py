# -*- coding: utf-8 -*-

"""
Implementation of the standard (supervised) training setup in Yao et al.
This means predicting the labels for each subtask is only based on the
respective recipe description. User answers are not available.

Yao et al. report accuracy of 64% with accuracy C+F 37% for a LAM-based
classifier. This script allows training our custom predictor to achieve
comparable results.

Note that we train a separate model for each subtask, as suggested in
Yao et al.

Accuracy: Overall accuracy of predicted trigger/function labels
          Each prediction (for each subtask) is counted individually.

Accuracy C+F: Accuracy across all trigger/channel functions/values
              Accuracy for recipe descriptions where all 4 subtask predictions
              are correct.
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

from torch import Tensor
from typing import Tuple, Dict, List, Any
from joeynmt.constants import PAD_TOKEN
from joeynmt.hrl_env import make_vocab
from joeynmt.hrl_env import  text2tensor
from joeynmt.vocabulary import Vocabulary
from joeynmt.recipe_classifier import RecipeClassifier


# You can set your parameters here
SEED = 42
MINIBATCHSIZE = 32
HIDDEN_SIZE = 512
EMBEDDING_DIM = 50
NUMLAYERS = 1
EPOCHS = 30

USE_CUDA = torch.cuda.is_available()
print("CUDA available:", USE_CUDA)

# Seeding for reproducibility:
# Please be aware that reproducibility is NOT ensured across devices
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_data(datapath: str="data/data_with_noisy_user_ans.pkl") \
    -> Tuple[Dict[str, TensorDataset], Vocabulary]:
    """
    Loads the data and stores it in DataLoader instances. This enables
    immediate and comfortable use in subsequent training procedures.
    Each batch holds recipe descriptions, lengths (because of padding),
    and all subtask labels.
    
    :param datapath: Path to data from Yao et al. (2018)
    :type datapath:  str
    :returns:        Mapping from dataset types (train/dev/test) to DataLoader
                     instances and the vocabulary
    """
    print("Unpickling dataset")
    with open(datapath, 'rb') as f:
        with warnings.catch_warnings():  # Because sklearn is annoying
            warnings.simplefilter("ignore")
            data = pickle.load(f, encoding='latin1')

    print("Making vocabulary")
    vocab = make_vocab(data['word_ids'])
    padding_value = vocab.stoi[PAD_TOKEN]
    
    # Make datasets
    print("Preparing dataset")
    datasets = dict()
    for mode in ['train', 'dev', 'test']:
        source = []
        targets = []
        for dp in data[mode]:
            # Get recipe description and convert to indices
            word_indices = text2tensor(dp['words'], vocab)
            labels = torch.LongTensor(dp['labels'])
        
            source.append(word_indices)
            targets.append(labels)
        
        # For calculating lengths and storing data in Tensors, we
        # pack and pad all recipe descriptions
        # Unfortunately, PackedSequence does not allow indexing, so we have
        # to store padding as well, which is not memory efficient.
        source = pack_sequence(source, enforce_sorted=False)
        source, lengths = pad_packed_sequence(source,
                                              batch_first=True,
                                              padding_value=padding_value
                                              )
        source = source.long()
        targets = torch.stack(targets).long()
        # Only shuffle training data + Provide DataLoader for easy
        # minibatching
        datasets[mode] = DataLoader(TensorDataset(source, lengths, targets),
                                    batch_size=MINIBATCHSIZE,
                                    shuffle= (mode=='train')
                                    )
    
    return datasets, vocab


def make_agents(
        vocab: Vocabulary,
        output_sizes: List[int]=[251, 876, 218, 458]
        ) -> nn.ModuleList:
    # Instantiate 4 identical classifiers, 1 for each subtask
    # We call them agents because the main task is Reinforcement Learning
    agent1 = RecipeClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_SIZE, NUMLAYERS,
                              output_sizes[0], padding_idx=vocab.stoi[PAD_TOKEN])
    agent2 = RecipeClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_SIZE, NUMLAYERS,
                              output_sizes[1], padding_idx=vocab.stoi[PAD_TOKEN])
    agent3 = RecipeClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_SIZE, NUMLAYERS,
                              output_sizes[2], padding_idx=vocab.stoi[PAD_TOKEN])
    agent4 = RecipeClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_SIZE, NUMLAYERS,
                              output_sizes[3], padding_idx=vocab.stoi[PAD_TOKEN])
    
    # ModuleList is convenient because we don't need a different optimizer for
    # agent, as ModuleList provides an `parameters` iterable
    return nn.ModuleList([agent1, agent2, agent3, agent4])


class StandardTrainer:
    """
    Training class. Provides train and evaluate methods.
    
    :param agents: Subtask predictors, 1 for each subtask
    :type agents:  nn.ModuleList
    :param cuda:   Whether to use CUDA/GPU
    :type cuda:    bool
    :param prefix: Prefix to prepend to names of saved agent weights
    :param prefix: str
    """
    def __init__(self, agents: nn.ModuleList, cuda: bool=False,
                 prefix: str = 'standard_training_'):
        self.agents = agents.cuda() if cuda else agents
        # Other than for RL, RecipeClassifier directly calculates log-softmax
        # outputs, so we can use NLLLoss
        self.criterion = nn.NLLLoss()
        # We could choose any other optimizer as well, but I happen to like
        # AdamW
        self.optimizer = torch.optim.AdamW(self.agents.parameters())
        
        self.cuda = cuda
        self.eval_mode = False  # Some subclasses use this member
        self.prefix = prefix
    
    def train(self, train_dataset: DataLoader, validation_dataset: DataLoader,
              epochs: int) -> None:
        """
        Training routine. Trains subtask predictors on the given training
        dataset for epochs epochs. Evaluates on the given validation dataset.
        For simplicity, early stopping is not implemented.
        
        :param train_dataset:      Training dataset
        :type train_dataset:       DataLoader
        :param validation_dataset: Validation dataset
        :type validation_dataset:  DataLoader
        :param epochs:             Number of epochs
        :type epochs:              int
        :returns:                  Nothing (None)
        """
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        for epoch in range(epochs):
            print("\n\n----- Epoch {} -----\n".format(epoch+1))
            batch_counter = 0
            total_batch_number = len(train_dataset)
            self.agents.train()
            self.eval_mode = False
            
            for batch in train_dataset:
                batch_counter += 1
                
                # Run batch, some returned values may be zero (not available)
                total_loss, num_correct, num_correct_cf, num_asks, \
                    batch_size = \
                        self.run_batch(batch)
                
                # We have to divide by 4, because for each recipe description
                # in batch, we make 4 predictions, 1 for each subtask
                accuracy = num_correct / (4*batch_size)
                # Accuracy C+F is calculated for each recipe description
                # across all 4 subtasks instead
                accuracy_cf = num_correct_cf / batch_size
                # We give the average number of questions per recipe
                # description, not subtask
                avg_num_asks = num_asks / batch_size
            
                # Update Agents
                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
                print("Epoch {:02}\t Batch {:04}/{}\t Loss: {:.3f}"
                      "\t Accuracy: {:.2f}\t Accuracy C+F: {:.2f}\t # Asks: {:.2f}"\
                          .format(epoch, batch_counter, total_batch_number,
                                  total_loss.item(), accuracy, accuracy_cf, avg_num_asks),
                    end='\r')
            
            # And after each epoch, we evaluate (this is for info only)
            # because we don't do early stopping
            self.evaluate(self.validation_dataset)
        
        # After training, save the agents
        # Please be aware that existing files bearing the same names will
        # be overwritten
        print("\nSaving Agents")
        for i, agent in enumerate(self.agents):
            torch.save(agent.state_dict(), self.prefix + "agent{}.pth".format(i))


    def evaluate(self, validation_dataset: DataLoader) -> None:
        """
        Calculate Loss, Accuracy, Accuracy C+F, and # Asks on the given
        validation dataset.
        
        :param validation_dataset: The validation dataset
        :type validation_dataset:  DataLoader
        :returns:                  Nothing (None)
        """
        eval_loss = 0.0
        eval_num_correct_overall = 0
        eval_num_correct_cf = 0
        eval_recipes = 0
        eval_num_asks = 0
        
        self.eval_mode = True  # Some subclasses need this

        with torch.no_grad():
            self.agents.eval()
    
            for batch in validation_dataset:
                total_loss, num_correct, num_correct_cf, num_asks, batch_size = \
                    self.run_batch(batch)
                # Accumulate scores across batches
                eval_loss += total_loss.item()
                eval_num_correct_overall += num_correct
                eval_num_correct_cf += num_correct_cf
                eval_num_asks += num_asks
                eval_recipes += batch_size
    
        avg_loss = eval_loss / eval_recipes
        avg_num_asks = eval_num_asks / eval_recipes
        # For each recipe in a batch, we make 4 predictions, one for each
        # subtask, so we have to multiply the number of recipes by 4 when
        # calculating accuracy
        accuracy = eval_num_correct_overall / (4*eval_recipes)
        accuracy_cf = eval_num_correct_cf / eval_recipes
        print("\nEvaluation\t Avg Loss: {:.3f}\t Accuracy: {:.2f}"
              "\t Accuracy C+F: {:.2f}\t # Asks: {:.2f}"\
                  .format(avg_loss, accuracy, accuracy_cf, avg_num_asks))
        
        self.eval_mode = False
    
    def run_batch(self, batch: Tuple[Tensor, Tensor, Tensor]) \
        -> Tuple[Tensor, int, int, int, int]:
        """
        Get predictions for a batch of recipes. Records number of
        correct predictions, number of recipes where predictions in all
        subtasks are correct (accuracy c+f), number of questions to the
        user (always 0 in standard training) and the number of recipes in the
        batch.
        
        :param batch: Batch containing recipe descriptions, their lengths and
                      all subtask labels
        :type batch:  Tuple[Tensor, Tensor, Tensor]
        :returns:     Batch loss, number of correct predictions, number of
                      recipes where predictions in all subtasks are correct,
                      number of questions to the user, number of recipes in
                      batch
        """
        # Unpack batch
        source, lengths, targets = batch
        
        if USE_CUDA:
            source = source.cuda()
            lengths = lengths.cuda()
            targets = targets.cuda()

        total_loss = torch.tensor(0.0, device=source.device)
        num_correct = 0
        num_asks = 0

        batch_size = source.shape[0]
        correct_masks = []  # For calculating accuracy c+f
        
        # Iterate over all subtasks
        for subtask in range(4):
            agent = self.agents[subtask]
            predictions = agent(source, lengths).contiguous()
            subtask_targets = targets[:, subtask].view(-1).contiguous()
            
            subtask_loss = self.criterion(predictions, subtask_targets)
            total_loss += subtask_loss
            
            predicted_labels = torch.argmax(predictions, dim=-1)
            correct_mask = (predicted_labels == subtask_targets)
            num_correct += correct_mask.sum().item()
            correct_masks.append(correct_mask)
        
        # For each recipes, we take the logical AND across all subtasks
        # This only yields true if predictions in all subtasks where correct
        num_correct_cf = correct_masks[0] & correct_masks[1] & \
            correct_masks[2] & correct_masks[3]
        num_correct_cf = num_correct_cf.sum().item()

        return total_loss, num_correct, num_correct_cf, num_asks, batch_size


if __name__ == '__main__':
    datasets, vocab = load_data()
    agents = make_agents(vocab)
    trainer = StandardTrainer(agents, cuda=USE_CUDA)
    trainer.train(datasets['train'], datasets['dev'], EPOCHS)
    
    print("\n\n\n\n----- Test -----\n")
    trainer.evaluate(datasets['test'])
