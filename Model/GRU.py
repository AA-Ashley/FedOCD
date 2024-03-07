from abc import ABC

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Model.BasicModel import BasicModel

class GRU(BasicModel):
    def __init__(self, parser):
        super(GRU, self).__init__(parser)
        self.Gru = nn.GRU(self.emb_size, self.emb_size, 1, bias=False).to(parser.device)
        self.logistic = torch.nn.Sigmoid().to(parser.device)

    def computer(self, log_seqs):
        seqs = self.emb_item(log_seqs)
        output, temp = self.Gru(seqs)
        output = self.logistic(output)
        return output

    def forward(self, users, pos_items, neg_items, seq):
        log_feats = self.computer(seq)[:, -1, :]
        emb_pos = self.emb_item(pos_items)
        emb_neg = self.emb_item(neg_items)
        pos_score = (log_feats * emb_pos).sum(dim=-1)
        neg_score = (log_feats * emb_neg).sum(dim=-1)
        return pos_score, neg_score

    def get_loss(self, users, pos_items, neg_items, seq):
        pos_scores, neg_scores = self.forward(users, pos_items, neg_items, seq)
        loss = - (pos_scores - neg_scores).sigmoid().log().mean()
        return loss, 0

    def predict(self, users, items, seqs):
        log_feats = self.computer(seqs)
        final_feat = log_feats[:, -1, :]
        self.emb_user.weight.data[users.long()] = final_feat
        # user_ids hasn't been used yet
        # only use last QKV classifier, a waste
        # (U, I, C)
        emb_items = self.emb_item(items)
        emb_users = self.emb_user(users)
        scores = torch.mul(emb_users, emb_items)
        scores = torch.sum(scores, dim=1)
        scores = self.logistic(scores)
        return scores
