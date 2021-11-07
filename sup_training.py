# -*- coding: utf-8 -*-

"""
Implementation of the sup (supervised) training setup in Yao et al.
This means for many recipes and subtasks, the label is set to "ask user" and
another sample is added to the dataset which consists of an user answer 
concatenated to the original recipe description. Here, the label is the true
subtask label.

Yao et al. report accuracy of 94% and accuracy C+F 81% for a LAM-based
classifier. Unfortunately, the classifier trained by this script will
achieve only less accuracy (~ 78% // accuracy C+F ~0.5).

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
from typing import List, Dict, Tuple, Any
from joeynmt.vocabulary import Vocabulary
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
MINIBATCHSIZE = 128
HIDDEN_SIZE = 512
EMBEDDING_DIM = 50
NUMLAYERS = 1  # For LSTM
EPOCHS = 20

ASK_USER_LABELS = [251, 876, 218, 458]

USE_CUDA = torch.cuda.is_available()

# Seeding for reproducibility:
# Please be aware that reproducibility is NOT ensured across devices
np.random.seed(SEED)
torch.manual_seed(SEED)

# Define alias for frequently used return types:
BatchInfo = Tuple[Tensor, int, int, int, int]


class IFThenRecipeDataset(Dataset):
    """
    Custom dataset to make use of convenient PyTorch functionality 
    (e.g. automatic batching) while storing data as strings (not Tensors).
    
    Only used for training dataset.
    
    :param recipes:     All recipe descriptions
    :type recipes:      List[List[str]]
    :param labels:      All subtask labels for each recipe description
    :type label_names:  List[np.ndarray]
    :param label_names: Human readable subtask labels
    :type label_names:  List[List[str]]
    """
    def __init__(self, recipes: List[List[str]], labels: List[np.ndarray],
                 label_names: List[List[str]]):
        self.recipes = recipes
        self.labels = labels
        self.label_names = label_names
        
        # We need to ensure congruent data components
        assert len(self.labels) == len(self.recipes) == len(self.label_names)

    def __len__(self) -> int:
        return len(self.labels)


    def __getitem__(self, index: int):
        return self.recipes[index], self.labels[index], self.label_names[index]
    
    @staticmethod
    def recipe_collate(batch: list) -> Tuple[list, list, list]:
        """
        Replaces the built-in collate_fn of DataLoader to prevent trying to
        cast batches (which consist of lists of strings) to Tensors.
        """
        recipes, labels, label_names = zip(*batch)
        return list(recipes), list(labels), list(label_names)


class SupTrainer(StandardTrainer):
    """
    Subclass of StandardTrainer. Because training and development/test
    datasets look differently, two separate methods are added for 
    testing and training resspectively.
    
    During training, labels and recipes are given only by the dataset
    (including user answers).
    During testing and evaluation, user answers are requested
    only when the agent predicts "ask user". Also, during evaluation,
    "ask user" is never the true label (the training setup allows this during
    training).
    
    :param agents: Subtask predictors, 1 for each subtask
    :type agents:  nn.ModuleList
    :param vocab:  Vocabulary to convert sequences of tokens to sequences of
                   indices which are interpretable by torch.nn.Embedding
    :type vocab:   Vocabulary
    :param max_local_turns: Maximum number of questions per subtask (only used
                            during evaluation)
    :type max_local_turns:  int
    :param cuda:   Whether to use CUDA/GPU
    :type cuda:    bool
    """
    def __init__(self, agents: nn.ModuleList, vocab: Vocabulary,
                 max_local_turns: int = 3, cuda: bool=False):
        super(SupTrainer, self).__init__(
            agents, cuda=cuda, prefix='sup_training_'
            )
        self.vocab = vocab
        self.max_local_turns = max_local_turns
        # Disable dynamically created user answers
        self.user_simulator = UserSimulator(False)
        self.padding_value = vocab.stoi[PAD_TOKEN]
        
        self.ask_user_labels = [251, 876, 218, 458]
    
    
    def run_batch(self, batch: tuple) -> BatchInfo:
        """
        Depending on whether we are doing training or evaluation, run the
        respective run_batch method on the current batch.
        """
        if self.eval_mode:
            return self.eval_run_batch(batch)
        else:
            return self.train_run_batch(batch)
    
    def train_run_batch(self, batch: tuple) -> BatchInfo:
        """
        Training procedure for sup training. Because for each subtask the
        input might look differently (with user answer or without user 
        answer), we cannot predict all 4 subtasks on the same recipe 
        description. This is reflected in the dataset creation.
        
        To run the correct agent on each sample, we have to bucket the batch
        by subtasks and run each subtask predictor only on the samples
        belonging to the respective subtask.
        
        :param batch: Batch containing inputs (recipe descriptions +
                      optionally user answers), input lengths, labels, and
                      the subtask of each sample
        :type batch:  Tuple[Tensor, Tensor, Tensor, Tensor]
        :returns:     Batch loss, number of correct predictions, number of
                      recipes where predictions in all subtasks are correct,
                      number of questions to the user, number of recipes in
                      batch
        """
        # Unpack batch
        source, lengths, targets, subtasks = batch
        
        if USE_CUDA:
            source = source.cuda()
            lengths = lengths.cuda()
            targets = targets.cuda()
            subtasks = subtasks.cuda()
        
        # Bucket batch samples by subtask
        subtask_buckets = [None, None, None, None]
        
        for subtask in range(4):
            subtask_mask = torch.eq(subtasks, subtask)
            subtask_indices = torch.nonzero(subtask_mask).reshape(-1)
            
            # Only insert samples in bucket if there are samples belonging
            # to the subtask
            if len(subtask_indices):
                subtask_buckets[subtask] = (source[subtask_indices].contiguous(),
                                            lengths[subtask_indices].contiguous(),
                                            targets[subtask_indices].contiguous()
                                            )
                
        total_loss = torch.tensor(0.0, device=source.device)
        num_correct = 0
        num_asks = 0
        batch_size = source.shape[0]
        
        # Get prediction, loss, and scores for each subtask individually
        for subtask in range(4):
            # Don't run if no samples for subtask
            if subtask_buckets[subtask] is None:
                continue
            else:
                source, lengths, subtask_targets = subtask_buckets[subtask]
    
            agent = self.agents[subtask]
            predictions = agent(source, lengths).contiguous()
            
            subtask_loss = self.criterion(predictions, subtask_targets)
            total_loss += subtask_loss
            
            predicted_labels = torch.argmax(predictions, dim=-1)
            correct_mask = torch.eq(predicted_labels, subtask_targets)
            num_correct += correct_mask.sum().item()
            
        num_correct_cf = 0
        # We have to return 4*num_correct because StandardTrainer assumes that
        # we make 4 predictions for each sample in the batch. This is not true
        # in this setting (we make only one prediction per sample). So we have
        # to revert dividing by 4 in StandardTrainer
        return total_loss, 4*num_correct, num_correct_cf, num_asks, batch_size
    
    def make_batch(self, recipes: List[List[str]]) -> Tuple[Tensor, Tensor]:
        """
        Converts a batch of recipe descriptions to Tensors and the lengths of
        the descriptions (needed because of padding).
        
        :param recipes: Recipe descriptions
        :type recipes:  List[List[str]]
        :returns:       Recipes as (padded) Tensors, holding indices of tokens
                        according to self.vocab; Lengths of original recipe
                        descriptions
        """
        recipes = [text2tensor(recipe, self.vocab) for recipe in recipes]
        recipes = pack_sequence(recipes, enforce_sorted=False)
        recipes, lengths = pad_packed_sequence(recipes,
                                               batch_first=True,
                                               padding_value=self.padding_value
                                               )
        recipes = recipes.long()
        return recipes, lengths
    
    def eval_run_batch(self, batch: tuple) -> BatchInfo:
        """
        Evaluation procedure for sup training. For evaluation, we use the same
        input (recipe description) for each subtask. If the agent predicts
        "ask user", we fetch an user answer and concatenate it to the
        original recipe description. Then we predict again. This can be
        repeated max_local_turns many times, or until the subtask agent
        does not request more user answers.
        
        :param batch: Batch holding recipe descriptions, all subtask labels,
                      human readable label descriptions (needed for getting
                      user answers)
        :type batch:  Tuple[List[List[str]], List[List[int]], List[List[str]]]
        :returns:     Batch loss, number of correct predictions, number of
                      recipes where predictions in all subtasks are correct,
                      number of questions to the user, number of recipes in
                      batch
        """
        # Unpack batch
        recipes, labels, label_names = batch
        
        total_loss = 0.0
        num_correct = 0
        num_asks = 0
        correct_masks = [None, None, None, None]
        
        batch_size = len(recipes)
        targets = torch.tensor(labels).long()
        
        # Make predictions for all subtasks
        for subtask in range(4):
            subtask_recipes = recipes.copy()
            # Retain a copy of recipes to attach user answers to
            current_recipes = subtask_recipes.copy()
            subtask_label_names = label_names.copy()
            subtask_targets = targets[:, subtask].view(-1).contiguous()
            # Store active batch indices
            true_indices = torch.arange(batch_size).long()
            
            all_predicted = False
            local_turns = 0
            
            agent = self.agents[subtask]
            ask_user_label = self.ask_user_labels[subtask]
            
            
            # Repeat until we have a prediction for each recipe description
            # or we reach self.max_local_turns
            while not all_predicted and local_turns < self.max_local_turns:
                local_turns += 1
                source, lengths = self.make_batch(current_recipes)
                
                if USE_CUDA and self.cuda:
                    source = source.cuda()
                    lengths = lengths.cuda()
                    subtask_targets = subtask_targets.cuda()
                    true_indices = true_indices.cuda()

                predictions = agent(source, lengths).contiguous()
                
                # We could mask out the loss for samples where "ask user"
                # is predicted, but it's not so important for evaluation,
                # and in this way more questions also means higher loss
                # which intuitively is what we want
                subtask_loss = self.criterion(predictions, subtask_targets)
                total_loss += subtask_loss
            
                predicted_labels = torch.argmax(predictions, dim=-1)
                correct_mask = (predicted_labels == subtask_targets)
                num_correct += correct_mask.sum().item()
                
                # Tells us for which samples a user answer was requested
                ask_user_mask = torch.eq(predicted_labels, ask_user_label)
                
                if correct_masks[subtask] is None:
                    correct_masks[subtask] = correct_mask
                else:
                    correct_masks[subtask][true_indices] += correct_mask
                
                # Only samples for which "ask user" was predicted remain
                # active
                true_indices = true_indices.masked_select(ask_user_mask)
                
                if not torch.any(ask_user_mask):
                    all_predicted = True
                
                else:  # Get user answers and concatenate them to recipe
                       # descriptions
                    ask_user_indices = torch.nonzero(ask_user_mask)
                    ask_user_indices = ask_user_indices.reshape(-1)
                    
                    # Filter samples
                    subtask_recipes = [
                        subtask_recipes[index] for index in ask_user_indices
                        ]
                    subtask_targets = subtask_targets[ask_user_mask] \
                        .contiguous()
                    subtask_label_names = [
                        label_names[index] for index in ask_user_indices
                        ]
                    current_recipes = [
                        current_recipes[index] for index in ask_user_indices
                        ]
                    
                    # Get user answers
                    user_answers = []
                    for recipe, names in zip(subtask_recipes, 
                                             subtask_label_names):
                        user_answer = self.user_simulator.ask_user_answer(
                            recipe, names, subtask
                            )
                        user_answers.append(user_answer)
                    
                    current_recipes = [
                        # We add EOS_TOKEN between original recipe
                        # description and user answer so the classifier
                        # can separate them
                        current_recipe + [EOS_TOKEN] + user_answer
                        for current_recipe, user_answer
                        in zip(current_recipes, user_answers)
                        ]
                    
                    num_asks += len(ask_user_indices)
        
        # For each recipes, we take the logical AND across all subtasks
        # This only yields true if predictions in all subtasks where correct
        num_correct_cf = correct_masks[0] & correct_masks[1] & \
            correct_masks[2] & correct_masks[3]
        num_correct_cf = num_correct_cf.sum().item()
    
        return total_loss, num_correct, num_correct_cf, num_asks, batch_size


def load_data(datapath: str = "data/data_with_noisy_user_ans.pkl"):
    """
    Loads the data and stores it in DataLoader instances. This enables
    immediate and comfortable use in subsequent training procedures.
    
    For sup training, training dataset and validation/test datasets are
    different:
    
    Training dataset: Each sample consists of input (recipe description +
                      optionally user answer), length of input, the label
                      (could be "ask user"), the subtask the sample belongs 
                      to. All data are already converted to Tensor.
    Validation/Test datasets: Each sample consists of recipe description,
                              all subtask labels, subtask labels in human
                              readable form (text). Data are not Tensors,
                              but Lists (of strings/ints)
    
    :param datapath: Path to data from Yao et al. (2018)
    :type datapath:  str
    :returns:        Mapping from dataset types (train/dev/test) to DataLoader
                     instances and the vocabulary
    """
    print("Unpickling dataset")
    with open(datapath, 'rb') as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pickle.load(f, encoding='latin1')
    
    print("Instantiating UserSimulator")
    user_simulator = UserSimulator(True)

    print("Making vocabulary")
    vocab = make_vocab(data['word_ids'])
    padding_value = vocab.stoi[PAD_TOKEN]
    
    # Make datasets
    print("Preparing dataset")
    datasets = dict()
    
    #################################################################
    #  Train dataset                                                #
    #################################################################
    
    source = []
    targets = []
    subtasks = []
    for dp in data['train']:
        words = dp['words']
        pseudo_ask_labels = dp['pseudo_ask_labels']
        labels = dp['labels']
        label_names = [name.decode('utf-8') for name in dp['label_names']]
        
        for subtask, pseudo_ask_label in enumerate(pseudo_ask_labels):
            # Pseudo ask label is the information provided by Yao et al.
            # whether to add the recipe description with user answer
            # or not
            if pseudo_ask_label:
                source.append(words)
                targets.append(ASK_USER_LABELS[subtask])
                subtasks.append(subtask)
                
                user_answer = user_simulator.ask_user_answer(
                            words, label_names, subtask
                            )
                source.append(words + [EOS_TOKEN] + user_answer)
                targets.append(labels[subtask])
                subtasks.append(subtask)
            
            else:
                source.append(words)
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
    datasets['train'] = DataLoader(
        TensorDataset(source, lengths, targets, subtasks),
        batch_size = MINIBATCHSIZE,
        shuffle = True
        )
    
    #################################################################
    #  Test + Development datasets                                  #
    #################################################################
    
    for mode in ['dev', 'test']:
        recipes = [dp['words'] for dp in data[mode]]
        labels = [dp['labels'] for dp in data[mode]]
        label_names = [
            [name.decode("utf-8") for name in dp['label_names']]
            for dp in data[mode]
            ]
        
        datasets[mode] = DataLoader(
            IFThenRecipeDataset(recipes, labels, label_names),
            batch_size = MINIBATCHSIZE,
            shuffle = (mode=='train'),
            # Custom collate_fn to prevent automatic conversion to Tensor
            collate_fn = IFThenRecipeDataset.recipe_collate
            )
    
    return datasets, vocab


def make_agents(vocab, output_sizes=[252, 877, 219, 459]):
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
    trainer = SupTrainer(agents, vocab, cuda=USE_CUDA)
    trainer.train(datasets['train'], datasets['dev'], EPOCHS)
    
    print("\n\n----- Test -----\n")
    trainer.evaluate(datasets['test'])
