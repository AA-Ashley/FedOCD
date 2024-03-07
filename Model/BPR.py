from abc import ABC

import torch.nn as nn
import torch.nn.functional as F
from Model.BasicModel import BasicModel


class BPR(BasicModel, ABC):
    def __init__(self, parser):
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """
        super(BPR, self).__init__(parser)

    def forward(self, users=None, pos_items=None, neg_items=None, seq=None):
        if users != None:
            emb_user = self.emb_user(users)
            emb_pos_item = self.emb_item(pos_items)
            emb_neg_item = self.emb_item(neg_items)
            pos_scores = (emb_user * emb_pos_item).sum(dim=-1)
            neg_scores = (emb_user * emb_neg_item).sum(dim=-1)
            self.final_emb_user.weight.data = self.emb_user.weight.data
            self.final_emb_item.weight.data = self.emb_item.weight.data
            return pos_scores, neg_scores, self.emb_user, self.emb_item
        return self.emb_user, self.emb_item

    def get_loss(self, users, pos_items, neg_items, seq):
        pos_score, neg_score, _, _ = self.forward(users, pos_items, neg_items, seq)
        loss = - (pos_score - neg_score).sigmoid().log().mean()
        return loss, 0

    def predict(self, users, items, seq):
        # embedUser = self.embedUser(users)
        # embedItem = self.embedItem(items)
        emb_user = self.final_emb_user(users)
        emb_item = self.final_emb_item(items)
        scores = (emb_user * emb_item).sum(dim=-1)
        return scores


