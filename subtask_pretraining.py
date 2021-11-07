# -*- coding: utf-8 -*-

"""
Supervised pretraining of HRL-Subtask agents.
Only recipe descriptions are used for pretraining, no user answers.
Training is realised in a supervised fashion.

Subtask predictions are made in randomised order, so the agents don't
rely on a specific order which may then not match the (possibly variable)
order during RL-training.

Optionally, the agents can be pretrained to predict "ask user" at a certain
ratio. To achieve this, we randomly replace labels by the respective "ask
user" label.

For pretraining, we use custom Agents that support minibatching but have the
same parameters as the modules we use for RL.
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
from joeynmt.constants import PAD_TOKEN
from joeynmt.vocabulary import Vocabulary
from joeynmt.hrl_agent_pretraining import SequentialEncoder
from joeynmt.hrl_agent_pretraining import LowLevelAgent
from joeynmt.hrl_env import make_vocab
from joeynmt.hrl_env import  text2tensor


# You can set your parameters here
SEED = 42
MINIBATCHSIZE = 32
HIDDEN_SIZE = 512
EMBEDDING_DIM = 50
NUMLAYERS = 2
DROPOUT = 0.3
ASK_USER_RATIO = 0.0  # The probability with which to randomly replace a label
                      # by "ask user"
MIN_ASK_USER_RATIO = 0.1

USE_CUDA = torch.cuda.is_available()
print("CUDA available:", USE_CUDA)

# Seeding for reproducibility:
# Please be aware that reproducibility is NOT ensured across devices
np.random.seed(SEED)
torch.manual_seed(SEED)


class BatchedSequentialEncoder(SequentialEncoder):
    """
    Variant of SequentialEncoder that supports minibatches.
    SequentialEncoder is built to only support one sample at a time. This is
    sufficient for RL.
    """
    def forward(self, text: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Applies a bidirectional RNN to the input batch and returns
        the last hidden states (concatenated).
        
        :param text: Inputs with their lengths
        :type text:  Tuple[Tensor, Tensor]
        :returns:    Last hidden states (concatenated)
        """
        source, lengths = text
        batch_size = source.shape[0]
        h_0, c_0 = self._init_hidden(batch_size)
        h_0 = h_0.to(source.device)
        c_0 = c_0.to(source.device)
        emb = F.dropout(self.embedding(source), p=DROPOUT)
        
        packed = pack_padded_sequence(emb, lengths, batch_first=True,
                                      enforce_sorted=False)
        
        out, (h, c) = self.rnn(packed, (h_0, c_0))
        
        num_directions = 2 if self.rnn.bidirectional else 1
        last_hidden = h.view(self.rnn.num_layers,
                             num_directions,
                             batch_size, 
                             self.rnn.hidden_size
                             )
        last_hidden_directions = [
            last_hidden[-1, 0, :, :],
            last_hidden[-1, 1, :, :]
            ]
        last_hidden = torch.cat(last_hidden_directions, dim=-1)
        return last_hidden


def load_data(datapath: str = "data/data_with_noisy_user_ans.pkl") \
        -> Tuple[Dict[str, TensorDataset], Vocabulary, Dict[int, Tensor]]:
    """
    Loads the data and stores it in DataLoader instances. This enables
    immediate and comfortable use in subsequent training procedures.
    Each batch holds recipe descriptions, lengths (because of padding),
    and all subtask labels.
    
    :param datapath: Path to data from Yao et al. (2018)
    :type datapath:  str
    :returns:        Mapping from dataset types (train/dev/test) to DataLoader
                     instances, the vocabulary, and a mapping from subtasks
                     to a vector holding the inverse frequencies of all
                     labels
    """
    with open(datapath, 'rb') as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pickle.load(f, encoding='latin1')
    
    vocab = make_vocab(data['word_ids'])
    padding_value = vocab.stoi[PAD_TOKEN]
    
    # Make datasets
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
    
    # In case we want to use label weigthing (because in fact the labels for
    # this task are quite unbalanced), for each subtask we create a weight
    # vector. For each label, the weight vector holds the inverse of the
    # absolute frequency of the label. Additionally, we apply softmax to the
    # whole vector
    label_count_dict = {index: dict() for index in range(4)}
    for dp in data['train']:
        for index, label in enumerate(dp['labels']):
            if label in label_count_dict[index]:
                label_count_dict[index][label] += 1
            else:
                label_count_dict[index][label] = 1
    
    weight_dict = dict()
    for subtask in range(4):
        max_label = max(label_count_dict[subtask].keys())
        num_labels = max_label + 2  # We add "ask user" as label for each
                                    # subtask
        label_weights = [0 for _ in range(num_labels)]
        for label, count in label_count_dict[subtask].items():
            label_weights[label] = 1/count
        
        label_weights[-1] = ASK_USER_RATIO
        label_weights = torch.tensor(label_weights).float()
        label_weights = torch.softmax(label_weights, dim=-1)
        weight_dict[subtask] = label_weights
    
    return datasets, vocab, weight_dict


def make_fake_user_answer_labels(targets: Tensor,
                                 ask_user_ratio: float) -> Tensor:
    """
    Randomly replace targets with the "ask user" label.
    
    :param targets: The true labels
    :type targets:  Tensor (of shape [batch_size, 4])
    :param ask_user_ratio: Probability of replacing a label by "ask user"
    :type ask_user_ratio:  float
    :returns: The targets where some labels are replaced by "ask user"
    """
    ask_labels = [251, 876, 218, 458]  # "ask user" indices
    # Create empty Tensor for concatenating masked targets
    masked_targets = targets.new_empty((targets.shape[0], 0))
    for subtask in range(4):
        subtask_targets = targets[:, subtask]
        # Create a mask with entries 0 and 1 (1 with ask_user_ratio 
        # probability)
        mask = torch.zeros_like(subtask_targets) \
            .bernoulli_(ask_user_ratio).byte()
        # Replace labels by "ask user" according to mask
        subtask_targets = torch.where(
            mask, subtask_targets.new_full([1], ask_labels[subtask]),
            subtask_targets)
        subtask_targets = subtask_targets.unsqueeze(1)
        
        masked_targets = torch.cat([masked_targets, subtask_targets], dim=-1)
    
    return masked_targets.long()


def make_agents(vocab: Vocabulary) -> nn.ModuleList:
    """
    Instantiates 4 Low-Level-policy agents and stores them in nn.ModuleList
    
    :param vocab: The vocabulary (needed for input size)
    :type vocab:  Vocabulary
    :returns:     Subtask agents as nn.ModuleList
    """
    # Instantiate recipe and user understanding modules
    recipe_understanding1 = BatchedSequentialEncoder(input_size=len(vocab),
                                                     embedding_size=EMBEDDING_DIM,
                                                     hidden_size=HIDDEN_SIZE,
                                                     num_layers=NUMLAYERS)
    recipe_understanding2 = BatchedSequentialEncoder(input_size=len(vocab),
                                                     embedding_size=EMBEDDING_DIM,
                                                     hidden_size=HIDDEN_SIZE,
                                                     num_layers=NUMLAYERS)
    recipe_understanding3 = BatchedSequentialEncoder(input_size=len(vocab),
                                                     embedding_size=EMBEDDING_DIM,
                                                     hidden_size=HIDDEN_SIZE,
                                                     num_layers=NUMLAYERS)
    recipe_understanding4 = BatchedSequentialEncoder(input_size=len(vocab),
                                                     embedding_size=EMBEDDING_DIM,
                                                     hidden_size=HIDDEN_SIZE,
                                                     num_layers=NUMLAYERS)
    user_understanding1 = BatchedSequentialEncoder(input_size=len(vocab),
                                                   embedding_size=EMBEDDING_DIM,
                                                   hidden_size=HIDDEN_SIZE,
                                                   num_layers=NUMLAYERS)
    user_understanding2 = BatchedSequentialEncoder(input_size=len(vocab),
                                                   embedding_size=EMBEDDING_DIM,
                                                   hidden_size=HIDDEN_SIZE,
                                                   num_layers=NUMLAYERS)
    user_understanding3 = BatchedSequentialEncoder(input_size=len(vocab),
                                                   embedding_size=EMBEDDING_DIM,
                                                   hidden_size=HIDDEN_SIZE,
                                                   num_layers=NUMLAYERS)
    user_understanding4 = BatchedSequentialEncoder(input_size=len(vocab),
                                                   embedding_size=EMBEDDING_DIM,
                                                   hidden_size=HIDDEN_SIZE,
                                                   num_layers=NUMLAYERS)
    # Instantiate agents
    low_level_agent1 = LowLevelAgent(recipe_understanding1,
                                     user_understanding1,
                                     HIDDEN_SIZE,
                                     252)
    low_level_agent2 = LowLevelAgent(recipe_understanding2,
                                     user_understanding2,
                                     HIDDEN_SIZE,
                                     877)
    low_level_agent3 = LowLevelAgent(recipe_understanding3,
                                     user_understanding3,
                                     HIDDEN_SIZE,
                                     219)
    low_level_agent4 = LowLevelAgent(recipe_understanding4,
                                     user_understanding4,
                                     HIDDEN_SIZE,
                                     459)

    return nn.ModuleList([low_level_agent1, low_level_agent2,
                          low_level_agent3, low_level_agent4])


def run_batch(batch: Tuple[Tensor, Tensor, Tensor], agents: nn.ModuleList,
              criterion: nn.Module, ask_user_ratio: float) \
                  -> Tuple[Tensor, int, int]:
    """
    Gets predictions for each subtask and sample in batch. Calculates loss.
    
    :param batch: Batch holding recipe descriptions, their lengths, all
                  subtask labels
    :type batch:  Tuple[Tensor, Tensor, Tensor]
    :param agents: Low-level Agents for each subtask
    :type agents:  nn.ModuleList
    :param criterion: Functor for calculating loss based on predictions and
                      true targets
    :type criterion:  nn.Module
    :param ask_user_ratio: Probability with which to replace a label by "ask
                           user"
    :type ask_user_ratio:  float
    :returns: Loss, number of correct predictions, batch size
    """
    source, lengths, targets = batch

    if USE_CUDA:
        source = source.cuda()
        lengths = lengths.cuda()
        targets = targets.cuda()

    total_loss = torch.tensor(0.0, device=source.device)
    num_correct = 0
    num_asks = 0  # Always 0 for this pretraining method
    correct_masks = []
            
    # Reset Agents
    batch_size = source.shape[0]
    for agent in agents:
        agent.reset()
        agent.current_state_repr = \
            source.new_zeros((batch_size, agent.hidden_dim)).float()
            
    # 1. Decide subtask order (random)
    order = np.random.default_rng().permutation(4)
    targets = make_fake_user_answer_labels(targets, ask_user_ratio)
            
    # 2. Get predictions for each subtask + Calculate loss
    for subtask in order:
        # Get states from each agent
        states = [agent.state for agent in agents]
        agent = agents[subtask]

        agent.current_recipe = (source, lengths)
        obs = ((source, lengths), subtask)
        predictions = agent(obs, states).contiguous()

        subtask_targets = targets[:, subtask].view(-1).contiguous()
        subtask_loss = criterion(predictions, subtask_targets)

        total_loss += subtask_loss
        
        predicted_labels = torch.argmax(predictions, dim=-1)
        subtask_correct = torch.eq(predicted_labels, subtask_targets)
        num_correct += subtask_correct.sum().item()
        correct_masks.append(subtask_correct)
    
    correct_c_f_mask = \
        correct_masks[0] & correct_masks[1] & \
            correct_masks[2] & correct_masks[3]
    correct_c_f = correct_c_f_mask.sum().item()
    
    return total_loss, num_correct, correct_c_f, num_asks, batch_size


def evaluate(val_dataset: DataLoader, agents: nn.ModuleList,
             criterion: nn.Module, run_batch_func) -> None:
    """
    Calculates and prints average (batch) loss and accuracy on the given
    validation dataset.
    
    :param val_dataset: Validation dataset
    :type val_dataset:  DataLoader
    :param agents:      All 4 low-level agents
    :type agents:       nn.ModuleList
    :param criterion:   Functor that calculates loss based on predictions
                        and true targets
    :type criterion:    nn.Module
    returns:            Nothing (None)
    """
    eval_loss = 0.0
    eval_num_correct = 0
    eval_correct_c_f = 0
    eval_batches = 0
    eval_num_asks = 0
    with torch.no_grad():
        agents.eval()
        for batch in val_dataset:
            # Set ask_user_ratio to 0 for evaluation
            total_loss, num_correct, num_correct_c_f, num_asks, batch_size = \
                run_batch_func(batch, agents, criterion, 0.0)
            eval_loss += total_loss.item()
            eval_num_correct += num_correct
            eval_batches += batch_size
            eval_correct_c_f += num_correct_c_f
            eval_num_asks += num_asks
    
    avg_loss = eval_loss / eval_batches
    avg_num_asks = eval_num_asks / eval_batches
    # Because for each sample in batch we get 4 predictions, we have to devide
    # by 4 to get the true accuracy
    accuracy = eval_num_correct / (4*eval_batches)
    accuracy_c_f = eval_correct_c_f / eval_batches
    agents.train()
    print("\nEvaluation\t Avg Loss: {:.3f}\t Accuracy: {:.2f}\t Accuracy C+F: {:.2f}\t # Asks: {:.2f}"\
        .format(avg_loss, accuracy, accuracy_c_f, avg_num_asks))


def pretrain(agents: nn.ModuleList, train_dataset: DataLoader,
             val_dataset: DataLoader, criterion: nn.Module,
             ask_user_ratio: float, min_ask_user_ratio: float, 
             epochs: int, run_batch_func, evaluate_func) -> None:
    """
    (Pre-)training procedure. Trains the given agents on the given
    training dataset for the given number of epochs. After each epoch,
    print loss and accuracy on the given validation dataset.
    
    :param agents:         All 4 low-level agents
    :type agents:          nn.ModuleList
    :param train_dataset:  Training dataset
    :type train_dataset:   DataLoader
    :param val_dataset:    Validation dataset
    :type val_dataset:     DataLoader
    :param criterion:      Functor that calculates loss based on predictions
                           and true targets
    :type criterion:       nn.Module
    :param ask_user_ratio: Probability with which to replace a label by "ask
                           user"
    :type ask_user_ratio:  float
    :param epochs:         Number of epochs
    :type epochs:          int
    """
    if USE_CUDA:
        try:
            criterion = criterion.cuda()
        except AttributeError:  # criterion is most likely a function
            pass
        agents = agents.cuda()

    optimizer = torch.optim.AdamW(agents.parameters())

    for epoch in range(epochs):
        print(agents[0].device)
        print("\n\n----- Epoch {} -----\n".format(epoch+1))
        batch_counter = 0
        total_batch_number = len(train_dataset)
        for batch in train_dataset:
            batch_counter += 1
            
            total_loss, num_correct, num_correct_c_f, num_asks, batch_size = \
                run_batch_func(batch, agents, criterion, ask_user_ratio)
            accuracy = num_correct / (4*batch_size)
            accuracy_c_f = num_correct_c_f / batch_size
            avg_num_asks = num_asks / batch_size
            
            # Update Agents
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print("Epoch {:02}\t Batch {:04}/{}\t Loss: {:.3f}\t Accuracy: {:.2f}\t Accuracy C+F: {:.2f}\t # Asks: {:.2f}"\
                .format(epoch, batch_counter, total_batch_number,
                        total_loss.item(), accuracy, accuracy_c_f, avg_num_asks), end='\r')
            ask_user_ratio = min(0.99*ask_user_ratio, min_ask_user_ratio)
        
        evaluate_func(val_dataset, agents, criterion, run_batch_func)
    
    for i, agent in enumerate(agents):
        torch.save(agent.state_dict(), "pretraining_agent{}.pth".format(i))
    

if __name__ == '__main__':
    datasets, vocab, weight_dict = load_data()
    agents = make_agents(vocab)
    criterion = torch.nn.CrossEntropyLoss()
    pretrain(agents, datasets['train'], datasets['dev'], criterion, 
             ASK_USER_RATIO, MIN_ASK_USER_RATIO, 40,
             run_batch, evaluate)
