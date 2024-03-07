from abc import ABC
import torch
import torch.nn as nn
from Model.BasicModel import BasicModel


class MatrixFactorization(BasicModel, ABC):
    def __init__(self, parser):
        super(MatrixFactorization, self).__init__(parser)
        self.affineOut = nn.Linear(in_features=self.emb_size, out_features=1).to(parser.device)
        self.logistic = nn.Sigmoid().to(parser.device)

    # 与LightGCN相比没有computer中这么复杂
    def forward(self, users=None, pos_items=None, neg_items=None, seq=None):
        if users != None:
            emb_user = self.emb_user(users)
            emb_pos_item = self.emb_item(pos_items)
            emb_neg_item = self.emb_item(neg_items)
            pos_scores, neg_scores = torch.mul(emb_user, emb_pos_item), torch.mul(emb_user, emb_neg_item)
            pos_scores, neg_scores = self.affineOut(pos_scores), self.affineOut(neg_scores)  # 进行线性变换（用于学习样本之间的关系）
            pos_scores, neg_scores = self.logistic(pos_scores), self.logistic(neg_scores)  # 将输入的值压缩到 (0, 1) 的范围内
            self.final_emb_user.weight.data = self.emb_user.weight.data
            self.final_emb_item.weight.data = self.emb_item.weight.data
            return pos_scores, neg_scores, self.emb_user, self.emb_item
        return self.emb_user, self.emb_item

    def predict(self, users, items, seq):
        emb_user = self.final_emb_user(users)
        emb_item = self.final_emb_item(items)
        scores = torch.mul(emb_user, emb_item)
        scores = self.affineOut(scores)
        scores = self.logistic(scores)
        return scores.squeeze()

    # 损失函数和LightGCN一样
    def get_loss(self, users, pos_items, neg_items, seq):
        pos_score, neg_score, _, _= self.forward(users, pos_items, neg_items, seq)
        loss = - (pos_score - neg_score).sigmoid().log().mean()
        return loss, 0