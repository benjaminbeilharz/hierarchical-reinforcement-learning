# coding: utf-8
# author: Benjamin Beilharz beilharz@cl.uni-heidelberg.de

import pickle
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch import Tensor, is_tensor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from typing import List, Tuple

from joeynmt.hrl_agent import SequentialEncoder
from joeynmt.vocabulary import Vocabulary
from joeynmt.nmt_env import make_vocab
from joeynmt.nmt_env import text2tensor
from joeynmt.constants import PAD_TOKEN
"""
"""


class PairwiseLoss(nn.Module):
    def __init__(self, batching):
        super(PairwiseLoss, self).__init__()
        self.batching = batching

    def forward(self, prediction: List[Tensor],
                target: List[Tensor]) -> Tensor:
        pref = lambda x, y: torch.exp(x) / (torch.exp(x) + torch.exp(y))
        to_tensor = lambda x: torch.Tensor(x).double().requires_grad_(True)
        if isinstance(prediction, list) and self.batching:
            pred_a, pred_b = prediction
            pref_a = torch.exp(pred_a) / (torch.exp(pred_a) +
                                          torch.exp(pred_b))
            pref_b = torch.exp(pred_b) / (torch.exp(pred_a) +
                                          torch.exp(pred_b))
            y_a, y_b = target
            return -torch.mean(y_a.double() * torch.log(pref_a).double() +
                               y_b.double() * torch.log(pref_b).double())

        # minibatch
        # pref_a = []
        # pref_b = []
        # y_as = []
        # y_bs = []
        # for _, p in enumerate(zip(prediction, target)):
        #     print(p)
        #     a, b = p
        #     y_a, y_b = q
        #     pref_a.append(pref(a, b))
        #     pref_b.append(pref(b, a))
        #     y_as.append(y_a)
        #     y_bs.append(y_b)
        #     # n = len(pref_a)
        # pref_a = to_tensor(y_as) * torch.log(to_tensor(pref_a))
        # pref_b = to_tensor(y_bs) * torch.log(to_tensor(pref_b))
        # print(pref_a, pref_b)


class BradleyTerry(SequentialEncoder):
    def __init__(self,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary,
                 dropout: float = .3,
                 device: torch.device = torch.device('cuda')
                 if torch.cuda.is_available() else torch.device('cpu')):
        SequentialEncoder.__init__(self,
                                   input_size=len(src_vocab),
                                   embedding_size=512,
                                   hidden_size=1024)
        self.device = device
        self.dropout = dropout
        self.target = deepcopy(self.rnn)
        self.tembedding = deepcopy(self.embedding)

        self.convolutions = [nn.Conv1d(4, 1, i) for i in range(1, 16)]

        self.pool = nn.MaxPool2d(1)

        self.out = nn.Sequential(nn.Dropout(0.3),
                                 nn.Linear(1024 * len(self.convolutions), 700), nn.Sigmoid(), nn.Linear(700, 1),
                                 nn.LeakyReLU())

    def forward(self, src_text: Tuple[Tensor],
                trg_text: Tuple[Tensor]) -> Tensor:
        source, slengths = src_text
        batch_size = source.shape[0]
        target, tlengths = trg_text

        # source init
        srch_0, srcc_0 = self._init_hidden(batch_size)
        srch_0 = srch_0.to(source.device)
        srcc_0 = srcc_0.to(source.device)
        srcemb = F.dropout(self.embedding(source), p=self.dropout)

        # target init
        trgh_0, trgc_0 = self._init_hidden(batch_size)
        trgh_0 = trgh_0.to(source.device)
        trgc_0 = trgc_0.to(source.device)
        trgemb = F.dropout(self.tembedding(target), p=self.dropout)

        spacked = pack_padded_sequence(srcemb,
                                       slengths,
                                       batch_first=True,
                                       enforce_sorted=False)

        tpacked = pack_padded_sequence(trgemb,
                                       tlengths,
                                       batch_first=True,
                                       enforce_sorted=False)

        sout, (sh, sc) = self.rnn(spacked, (srch_0, srcc_0))
        tout, (th, tc) = self.target(tpacked, (trgh_0, trgc_0))

        def concat_hidden(hidden: Tensor) -> Tensor:
            num_directions = 2 if self.rnn.bidirectional else 1
            last_hidden = hidden.view(self.rnn.num_layers, num_directions,
                                      batch_size, self.rnn.hidden_size)
            last_hidden_directions = [
                last_hidden[-1, 0, :, :], last_hidden[-1, 1, :, :]
            ]
            last_hidden = torch.cat(last_hidden_directions,
                                    dim=0).to(source.device)
            return last_hidden

        sh = concat_hidden(sh)
        th = concat_hidden(th)

        lstm_concat = torch.cat([sh, th], dim=0).unsqueeze(0).to(source.device)
        conv_out = []
        for i, conv in enumerate(self.convolutions):
            o = F.relu(conv(lstm_concat)).to(source.device)
            if i != 0:
                pad = nn.ConstantPad1d((0, i), 3)
                o = pad(o).to(source.device)
            conv_out.append(o)

        conv_cat = torch.cat(conv_out, axis=1).to(source.device)
        conv_cat = self.pool(conv_cat).view(1, -1).to(source.device)

        return self.out(conv_cat).to(source.device)


def load_data(filepath: str = 'data/pairwise.pkl', batch_size: int = 1):
    with open(filepath, 'rb') as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pickle.load(f, encoding='utf8')

    src_vocab = make_vocab(data['src_word2idx'])
    trg_vocab = make_vocab(data['trg_word2idx'])
    padding_value = src_vocab.stoi[PAD_TOKEN]

    # Make datasets
    tar = ['src', 'trg']
    datasets = dict()
    for mode in ['train', 'dev']:
        for i, v in enumerate([src_vocab, trg_vocab]):
            source = []
            targets = []
            for dp in data[mode]:
                if np.isnan(np.array(dp[-1])): continue
                word_indices = text2tensor(dp[i], v)
                labels = torch.from_numpy(np.array(dp[-1]))
                # print(labels, word_indices)

                source.append(word_indices)
                targets.append(labels)

            # For calculating lengths and storing data in Tensors, we
            # pack and pad all recipe descriptions
            # Unfortunately, PackedSequence does not allow indexing, so we have
            # to store padding as well, which is not memory efficient.
            source = pack_sequence(source, enforce_sorted=False)
            source, lengths = pad_packed_sequence(source,
                                                  batch_first=True,
                                                  padding_value=padding_value)
            source = source.long()
            targets = torch.stack(targets)
            name = tar[i] + '_' + mode
            datasets[name] = DataLoader(TensorDataset(source, lengths,
                                                      targets),
                                        batch_size=batch_size,
                                        shuffle=False)

    return datasets, src_vocab, trg_vocab


def validate(val_dataset: DataLoader,
             model: nn.Module,
             criterion: nn.Module,
             device: torch.device = torch.device('cuda')
             if torch.cuda.is_available() else torch.device('cpu')):
    eval_loss = .0
    eval_batch_counter = 0
    with torch.no_grad():
        model.eval()
        for batch in val_dataset:
            if eval_batch_counter % 2 == 0:
                pred = []
                tar = []

            source, lenghts, target = batch
            source.to(device)
            lenghts.to(device)
            target.to(device)
            predictions = model((source, lenghts))
            pred.append(predictions)
            tar.append(target)

            if eval_batch_counter % 2 == 1:
                loss = criterion(pred, tar)
                eval_loss += loss.item()

            eval_batch_counter += 1

    print("\nEvaluation\t Avg Loss: {:.3f}".format(eval_loss /
                                                   eval_batch_counter))
    model.train()


def pretrain(model: nn.Module,
             src_train_data: DataLoader,
             trg_train_data: DataLoader,
             src_val_data: DataLoader,
             tar_val_data: DataLoader,
             criterion: nn.Module,
             epochs: int,
             device: torch.device = torch.device('cuda')
             if torch.cuda.is_available() else torch.device('cpu')):

    optimizer = torch.optim.AdamW(model.parameters())

    for epoch in range(epochs):
        print(model.device)
        print("\n\n----- Epoch {} -----\n".format(epoch + 1))
        batch_counter = 0
        total_batch_number = len(src_train_data)
        for s, t in zip(src_train_data, trg_train_data):
            if batch_counter % 2 == 0:
                pred = []
                tar = []
            ssource, slengths, target = s
            ssource.to(device)
            slengths.to(device)
            target.to(device)

            tsource, tlengths, _ = t
            tsource.to(device)
            tlengths.to(device)

            prediction = model((ssource, slengths), (tsource, tlengths))
            pred.append(prediction)
            tar.append(target)

            if batch_counter % 2 == 1:
                loss = criterion(pred, tar)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print("Epoch {:02}\t Batch {}/{}\t Loss: {:.3f}"\
                .format(epoch, batch_counter, total_batch_number,
                        loss.item()))

            batch_counter += 1

        # validate(val_data, model, criterion)

        torch.save(model.state_dict(), '1reward_estimator.pth')


if __name__ == '__main__':
    datasets, src_vocab, trg_vocab = load_data()
    model = BradleyTerry(src_vocab, trg_vocab)
    criterion = PairwiseLoss(True)
    pretrain(model,
             datasets['src_train'],
             datasets['trg_train'],
             None,
             None,
             criterion,
             epochs=50)
