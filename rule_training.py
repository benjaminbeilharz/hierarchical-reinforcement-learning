# -*- coding: utf-8 -*-

"""
Implementation of the rule (supervised) training setup in Yao et al.
For each recipe description and subtask, a prediction is made. If the
prediction probability ('confidence') is lower than a given threshold,
an user answer is appended to the recipe description and another prediction
is made on the concatenated input.

Yao et al. report accuracy of 93% with accuracy C+F 76% for a LAM-based
classifier.

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
from typing import Tuple, List, Dict, Any
from joeynmt.constants import PAD_TOKEN
from joeynmt.constants import EOS_TOKEN
from joeynmt.vocabulary import Vocabulary
from joeynmt.hrl_env import make_vocab
from joeynmt.hrl_env import  text2tensor
from joeynmt.recipe_classifier import RecipeClassifier
from joeynmt.user_simulator import UserSimulator

from standard_training import StandardTrainer
from sup_training import IFThenRecipeDataset


# You can set your parameters here
SEED = 42
MINIBATCHSIZE = 32
HIDDEN_SIZE = 512
EMBEDDING_DIM = 50
NUMLAYERS = 1  # For LSTM
DROPOUT = 0.3

USE_CUDA = torch.cuda.is_available()

# Seeding for reproducibility:
# Please be aware that reproducibility is NOT ensured across devices
np.random.seed(SEED)
torch.manual_seed(SEED)

# Define alias for frequently used return types:
BatchInfo = Tuple[Tensor, int, int, int, int]


class RuleTrainer(StandardTrainer):
    """
    Subclass of StandardTrainer.
    
    For each sample in batch and subtask, we make a prediction. If the
    prediction probability (confidence) is below a certain threshold,
    we request an user answer, append it to the recipe description, and
    get another prediction.
    
    :param agents: Subtask predictors, 1 for each subtask
    :type agents:  nn.ModuleList
    :param vocab:  Vocabulary to convert sequences of tokens to sequences of
                   indices which are interpretable by torch.nn.Embedding
    :type vocab:   Vocabulary
    :param confidence_threshold: Predictions are only made if their 
                                 probability exceeds this score
    :type confidence_threshold:  float
    :param max_local_turns: Maximum number of questions per subtask (only used
                            during evaluation)
    :type max_local_turns:  int
    :param cuda:   Whether to use CUDA/GPU
    :type cuda:    bool
    """
    def __init__(self, agents: nn.ModuleList, vocab: Vocabulary,
                 confidence_threshold: float = 0.85, max_local_turns: int = 3,
                 cuda: bool=False):
        super(RuleTrainer, self).__init__(agents, cuda=cuda,
                                          prefix='rule_training_')
        self.vocab = vocab
        self.confidence_threshold = confidence_threshold
        self.max_local_turns = max_local_turns
        self.user_simulator = UserSimulator(False)
        self.padding_value = vocab.stoi[PAD_TOKEN]

    
    def make_batch(self, recipes: List[List[str]]) -> Tuple[Tensor, Tensor]:
        """
        Converts a batch of recipe descriptions to Tensors and gives the
        lengths of the descriptions (needed because of padding).
        
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
        
    
    def run_batch(self, batch: tuple) -> BatchInfo:
        """
        Evaluation procedure for rule training. For evaluation, we use the
        same input (recipe description) for each subtask. If the agent
        predicts "ask user", we fetch an user answer and concatenate it to the
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
        current_recipes = recipes.copy()
        
        total_loss = 0.0
        num_correct = 0
        num_asks = 0
        correct_masks = [None, None, None, None]
        
        batch_size = len(recipes)
        targets = torch.tensor(labels).long()
        
        # Get predictions for each subtask
        for subtask in range(4):
            # Retain copies of samples for iterative filtering of
            # batches
            subtask_recipes = recipes.copy()
            current_recipes = subtask_recipes.copy()
            subtask_label_names = label_names.copy()
            subtask_targets = targets[:, subtask].view(-1).contiguous()
            # We need to track for which samples predictions are not yet made;
            # 'true' refers to the index in the original batch. Because the
            # unfinished batch likely gets smaller with each iteration,
            # indices will not match any more
            true_indices = torch.arange(batch_size).long()
            
            all_predicted = False
            local_turns = 0
            
            agent = self.agents[subtask]
            
            # Repeat until we have predictions for each sample or
            # max_local_turns is reached
            while not all_predicted and local_turns < self.max_local_turns:
                local_turns += 1
                source, lengths = self.make_batch(current_recipes)
                
                if USE_CUDA and self.cuda:
                    source = source.cuda()
                    lengths = lengths.cuda()
                    subtask_targets = subtask_targets.cuda()
                    true_indices = true_indices.cuda()

                predictions = agent(source, lengths).contiguous()
                predicted_labels = torch.argmax(predictions, dim=-1)
                correct_mask = (predicted_labels == subtask_targets)
                
                # Get confidences:
                # We predict the label that gets the highes probability
                # -> confidence is highest score in probability distribution
                # for each sample
                # Because RecipeClassifier outputs log-softmax, we need to
                # exp scores first (otherwise we could apply log to 
                # confidence_threshold, then select the max
                confidences, _ = torch.max(torch.exp(predictions), dim=-1)
                inconfidence_mask = torch.le(
                    confidences, self.confidence_threshold
                    )
                # inconfidence_indices are the indices where we do not have a
                # prediction (prediction probability < confidence_threshold)
                inconfidence_indices = torch.nonzero(inconfidence_mask)
                inconfidence_indices = inconfidence_indices.reshape(-1)
                # confident_indices are the indices where we have predictions
                # (prediction probability > confidence_threshold)
                confident_indices = torch.nonzero(
                    inconfidence_mask.logical_not()
                    )
                confident_indices = confident_indices.reshape(-1)
                
                confident_predictions = predictions[confident_indices] \
                    .contiguous()
                confident_targets = subtask_targets[confident_indices] \
                    .contiguous()
                
                # As long as we can make new predictions, we only calculate
                # loss for those samples where we have (confident) predictions
                if local_turns < self.max_local_turns:
                    if len(confident_indices) > 0:
                        subtask_loss = self.criterion(confident_predictions,
                                                      confident_targets)
                    # If we don't have any confident predictions, we can't
                    # calculate any loss
                    else:
                        subtask_loss = torch.tensor(
                            0.0, device=predictions.device
                            )
                # If we have reached max_local_turns but still some samples
                # don't have confident predictions, we still want to have
                # loss signal for them, so we calculate loss nonetheless
                else:
                    subtask_loss = self.criterion(
                        predictions, subtask_targets
                        )

                total_loss += subtask_loss
                
                # We only record confident predictions
                correct_mask_final = \
                    correct_mask & inconfidence_mask.logical_not()
                num_correct += correct_mask_final.sum().item()
                
                # If this is the first iteration, we can just set
                # correct_mask_final as all correct predictions
                if correct_masks[subtask] is None:
                    correct_masks[subtask] = correct_mask_final
                # Otherwise, we need to insert the correct predictions
                # at the right indices
                else:
                    correct_masks[subtask][true_indices] += correct_mask_final
                
                # And we have to update true_indices, because we remove
                # samples where we now have confident predictions from
                # the active batch
                true_indices = true_indices.masked_select(inconfidence_mask)
                
                
                if not torch.any(inconfidence_mask):
                    # We have predictions for every sample
                    all_predicted = True
                
                # Otherwise, request user answers for samples where we do not
                # have confident predictions, append them to the recipe
                # description and filter all samples where we have confident
                # predictions
                else:
                    subtask_recipes = [
                        subtask_recipes[index] for index in inconfidence_indices
                        ]
                    subtask_targets = subtask_targets[inconfidence_mask]
                    subtask_label_names = [
                        label_names[index] for index in inconfidence_indices
                        ]
                    current_recipes = [
                        current_recipes[index] for index in inconfidence_indices
                        ]
                
                    user_answers = []
                    for recipe, names in \
                            zip(subtask_recipes, subtask_label_names):
                        user_answer = self.user_simulator.ask_user_answer(
                            recipe, names, subtask
                            )
                        user_answers.append(user_answer)
                    
                    current_recipes = [
                        # We add EOS after the recipe description
                        # so the agent can see the boundaries between
                        # description and user answer
                        current_recipe + [EOS_TOKEN] + user_answer
                        for current_recipe, user_answer
                        in zip(current_recipes, user_answers)
                        ]
                    
                    num_asks += len(inconfidence_indices)
        
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
    
    Each sample consists of recipe description, all subtask labels, subtask
    labels in human readable form (text). Data are not Tensors, but Lists
    (of strings/ints).
    
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
    print("Making vocabulary")
    vocab = make_vocab(data['word_ids'])
    padding_value = vocab.stoi[PAD_TOKEN]
    
    # Make datasets
    print("Preparing dataset")
    datasets = dict()
    for mode in ['train', 'dev', 'test']:
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


def make_agents(
    vocab: Vocabulary, output_sizes: List[int] = [251, 876, 218, 458]
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


if __name__ == '__main__':
    datasets, vocab = load_data()
    agents = make_agents(vocab)
    trainer = RuleTrainer(agents, vocab, cuda=USE_CUDA)
    trainer.train(datasets['train'], datasets['dev'], 30)
    
    print("\n\n----- Test -----\n")
    trainer.evaluate(datasets['test'])
