# -*- coding: utf-8 -*-

"""
Supervised pretraining of HRL-Subtask agents.
Similarly to the sup-training setup in Yao et al. (2018), this pretraining
method enables the subtask agents to learn from user answers. Like in the
sup-pretraining setup the subtask agents are actively trained to predict
"ask user". This is controlled by the probability in ASK_USER_RATIO.
Differently, when an agent predicts "ask user", we actually provide an user
answer and predict again. This is more similar to rule-training. In
particular, the subset of recipe descriptions where we provide user answers
is not static.

Furthermore, if the agent requests an user answer and the subsequent
prediction is correct, "ask user" is treated as the true label for the
recipe description (without user answer). If the agent's prediction is
wrong after receiving the user answer, the true target also for the
original recipe description (without user answer) is set to the true label
(not "ask user").

Also, we anneal ASK_USER_RATIO with each batch by a factor of 0.99 until
it reaches a minimum value. By this, we try to train the agent to ask many
questions in early training and reducing their number as the agent's
predictions get more accurate.

Subtask predictions are made in randomised order, so the agents don't
rely on a specific order which may then not match the (possibly variable)
order during RL-training.

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
from typing import Tuple, List, Dict, Any
from joeynmt.vocabulary import Vocabulary
from joeynmt.constants import PAD_TOKEN
from joeynmt.hrl_agent_pretraining import SequentialEncoder
from joeynmt.hrl_agent_pretraining import LowLevelAgent
from joeynmt.hrl_env import make_vocab
from joeynmt.hrl_env import  text2tensor
from joeynmt.user_simulator import UserSimulator

from subtask_pretraining import BatchedSequentialEncoder
from subtask_pretraining import make_fake_user_answer_labels
from subtask_pretraining import evaluate
from subtask_pretraining import pretrain


# You can set your parameters here
SEED = 42
MINIBATCHSIZE = 128
HIDDEN_SIZE = 512
EMBEDDING_DIM = 50
NUMLAYERS = 2
DROPOUT = 0.3
ASK_USER_RATIO = 0.8 # The starting probability with which to randomly replace
                     # a label by "ask user" (annealed over time)
MIN_ASK_USER_RATIO = 0.2

USE_CUDA = torch.cuda.is_available()
print("CUDA available:", USE_CUDA)

# Seeding for reproducibility:
# Please be aware that reproducibility is NOT ensured across devices
np.random.seed(SEED)
torch.manual_seed(SEED)

ASK_LABELS = [251, 876, 218, 458]
    

def prepare_sequence(sequence: List[Tensor], padding_value: int) \
        -> Tuple[Tensor, Tensor]:
    """
    Pads sequences and calculates their lengths
    
    :param sequence: All input sequences
    :type sequence:  List[Tensor]
    :param padding_value: Value used for padding
    :type padding_value:  int
    :returns: Padded inputs as 1 Tensor and lengths of each sequence
    """
    sequence = pack_sequence(sequence, enforce_sorted=False)
    sequence, lengths = pad_packed_sequence(
        sequence, batch_first=True, padding_value=padding_value
        )
    
    return sequence, lengths


def load_data(datapath: str = "data/data_with_noisy_user_ans.pkl") \
        -> Tuple[Dict[str, TensorDataset], Vocabulary, Dict[int, Tensor]]:
    """
    Loads the data and stores it in DataLoader instances. This enables
    immediate and comfortable use in subsequent training procedures.
    Each batch holds recipe descriptions, lengths (because of padding),
    and all subtask labels.
    
    Here, each batch also includes a user answer for each subtask and sample.
    
    :param datapath: Path to data from Yao et al. (2018)
    :type datapath:  str
    :returns:        Mapping from dataset types (train/dev/test) to DataLoader
                     instances, the vocabulary, and a mapping from subtasks
                     to a vector holding the inverse frequencies of all
                     labels
    """
    print("Unpickling data")
    with open(datapath, 'rb') as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pickle.load(f, encoding='latin1')
    
    print("Building vocab")
    vocab = make_vocab(data['word_ids'])
    padding_value = vocab.stoi[PAD_TOKEN]
    
    print("Instantiating User Simulator")
    user_simulator = UserSimulator(False)
    
    # Make datasets
    print("Preparing dataset")
    datasets = dict()
    for mode in ['train', 'dev', 'test']:
        source = []
        targets = []
        subtask_user_answers = [[] for _ in range(4)]
        for dp in data[mode]:
            word_indices = text2tensor(dp['words'], vocab)
            labels = torch.LongTensor(dp['labels'])
            
            label_names = [name.decode("utf-8") for name in dp['label_names']]
            for subtask_index in range(4):
                user_answer = user_simulator.ask_user_answer(
                    dp['words'], label_names, subtask_index)
                
                user_answer = text2tensor(user_answer, vocab)
                subtask_user_answers[subtask_index].append(user_answer)

            source.append(word_indices)
            targets.append(labels)
        
        # For calculating lengths and storing data in Tensors, we
        # pack and pad all recipe descriptions and user answers
        # Unfortunately, PackedSequence does not allow indexing, so we have
        # to store padding as well, which is not memory efficient.
        source, source_lengths = prepare_sequence(source, padding_value)
        user_answers1, user_answers1_lengths = \
            prepare_sequence(subtask_user_answers[0], padding_value)
        user_answers2, user_answers2_lengths = \
            prepare_sequence(subtask_user_answers[1], padding_value)
        user_answers3, user_answers3_lengths = \
            prepare_sequence(subtask_user_answers[2], padding_value)
        user_answers4, user_answers4_lengths = \
            prepare_sequence(subtask_user_answers[3], padding_value)
    
        source = source.long()
        user_answers1 = user_answers1.long()
        user_answers2 = user_answers2.long()
        user_answers3 = user_answers3.long()
        user_answers4 = user_answers4.long()
        targets = torch.stack(targets).long()
        datasets[mode] = DataLoader(
            TensorDataset(source, source_lengths, targets,
                          user_answers1, user_answers1_lengths,
                          user_answers2, user_answers2_lengths,
                          user_answers3, user_answers3_lengths,
                          user_answers4, user_answers4_lengths
                          ),
            # Only shuffle training data + Provide DataLoader for easy
            # minibatching
            batch_size=MINIBATCHSIZE, shuffle= (mode=='train')
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
    
    #low_level_agent1.load_state_dict(torch.load('models/agent0.pth', map_location=torch.device('cpu')))
    #low_level_agent2.load_state_dict(torch.load('models/agent1.pth', map_location=torch.device('cpu')))
    #low_level_agent3.load_state_dict(torch.load('models/agent2.pth', map_location=torch.device('cpu')))
    #low_level_agent4.load_state_dict(torch.load('models/agent3.pth', map_location=torch.device('cpu')))
        
    
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
    # Unpack batch
    (source, lengths, targets,
     user_answers1, user_answers1_lengths, 
     user_answers2, user_answers2_lengths,
     user_answers3, user_answers3_lengths,
     user_answers4, user_answers4_lengths) \
         = batch
            
    if USE_CUDA:
        source = source.cuda()
        lengths = lengths.cuda()
        targets = targets.cuda()
        user_answers1 = user_answers1.cuda()
        user_answers2 = user_answers2.cuda()
        user_answers3 = user_answers3.cuda()
        user_answers4 = user_answers4.cuda()
    
    user_answers = [
        user_answers1,
        user_answers2,
        user_answers3,
        user_answers4
        ]
    user_answer_lengths = [
        user_answers1_lengths,
        user_answers2_lengths,
        user_answers3_lengths,
        user_answers4_lengths
        ]

    total_loss = torch.tensor(0.0, device=source.device)
    num_correct = 0
    num_asks = 0
            
    # Reset Agents
    batch_size = source.shape[0]
    for agent in agents:
        agent.reset()
        agent.current_state_repr = \
            source.new_zeros((batch_size, agent.hidden_dim)).float()
        agent.start_episode = True  # Don't treat recipe description as user
                                    # answer
            
    # 1. Decide subtask order (random)
    order = np.random.default_rng().permutation(4)
    # Randomly replace labels by "ask user" according to ask_user_ratio
    unmasked_targets = targets.clone().contiguous()
    targets = make_fake_user_answer_labels(targets, ask_user_ratio)
            
    # 2. Get predictions for each subtask + Calculate loss
    correct_masks = []
    for subtask in order:
        # Get states from each agent
        states = [agent.state for agent in agents]
        agent = agents[subtask]
        ask_label = ASK_LABELS[subtask]

        agent.current_recipe = (source, lengths)
        obs = ((source, lengths), subtask)
        predictions = agent(obs, states).contiguous()
        unmasked_subtask_targets = \
            unmasked_targets[:, subtask].view(-1).contiguous()
        subtask_targets = targets[:, subtask].view(-1).contiguous()
        
        predicted_labels = torch.argmax(predictions, dim=-1)
        ask_user_mask = torch.eq(predicted_labels, ask_label)
        
        # Correct predictions in first pass (including "ask user")
        first_pass_correct = torch.eq(predicted_labels, subtask_targets)
        correct_overall = first_pass_correct
        
        # Allow for a second prediction with user answer
        # if the first pass prediction was "ask user"
        if torch.any(ask_user_mask):
            # Save recipe encodings and states from first pass
            first_pass_recipe_encoding = agent.current_recipe_encoding
            first_pass_states = [agent.state for agent in agents]
            # Filter batch
            agent.current_recipe_encoding = \
                first_pass_recipe_encoding[ask_user_mask].contiguous()
            states = [agent.state[ask_user_mask] for agent in agents]

            current_user_answers = user_answers[subtask]
            current_user_answers = current_user_answers[ask_user_mask]
            current_user_answer_lengths = user_answer_lengths[subtask]
            current_user_answer_lengths = \
                current_user_answer_lengths[ask_user_mask]
            
            # Give user answers to agents
            second_pass_observation = (
                (current_user_answers, current_user_answer_lengths),
                 subtask)
        
            second_pass_predictions = agent(second_pass_observation, states)
            # Use fake targets only for first pass
            second_pass_targets = unmasked_subtask_targets[ask_user_mask]
            
            second_pass_loss = criterion(second_pass_predictions,
                                         second_pass_targets)
            
            second_pass_predicted_labels = \
                torch.argmax(second_pass_predictions, dim=-1)
            second_pass_correct = torch.eq(second_pass_predicted_labels,
                                           second_pass_targets)
            second_pass_num_correct = second_pass_correct.sum().item()
            
            # If first pass prediction is "ask user" and the second pass
            # prediction is correct, we use "ask user" as the first pass
            # target
            ask_user_indices = ask_user_mask.nonzero().squeeze(1).long()
            update_mask = torch.zeros_like(subtask_targets).bool()
            update_mask[ask_user_indices] = second_pass_correct
            update_subtask_targets = update_mask & ask_user_mask
            subtask_targets = \
                subtask_targets.masked_fill(update_subtask_targets, ask_label)
            
            # Restore states and recipe encodings for next subtask
            # (which starts with full batch again)
            agent.current_recipe_encoding = first_pass_recipe_encoding
            agent.current_state_repr = \
                first_pass_states[subtask].index_put((ask_user_indices,),
                                                     agent.state)
            
            correct_overall[ask_user_indices] = second_pass_correct
        
        else:
            second_pass_num_correct = 0
            second_pass_loss = 0.0
        
        subtask_loss = criterion(predictions, subtask_targets)
        subtask_loss += second_pass_loss
        total_loss += subtask_loss
        
        first_pass_correct = \
            first_pass_correct & (ask_user_mask.logical_not())
        num_correct += first_pass_correct.sum().item()
        num_correct += second_pass_num_correct
        
        num_asks += ask_user_mask.sum().item()
        correct_masks.append(correct_overall)
    
    # For each recipes, we take the logical AND across all subtasks
    # This only yields true if predictions in all subtasks where correct
    correct_c_f_mask = correct_masks[0] & correct_masks[1] & \
        correct_masks[2] & correct_masks[3]
    correct_c_f = correct_c_f_mask.sum().item()
    
    return total_loss, num_correct, correct_c_f, num_asks, batch_size
    

if __name__ == '__main__':
    datasets, vocab, weight_dict = load_data()
    criterion = torch.nn.CrossEntropyLoss()
    agents = make_agents(vocab)
    if USE_CUDA:
        print("Agents to CUDA")
        agents = agents.cuda()
    pretrain(agents, datasets['train'], datasets['dev'], criterion, 
             ASK_USER_RATIO, MIN_ASK_USER_RATIO, 15, run_batch, evaluate)
    evaluate(datasets['test'], agents, criterion, run_batch)
