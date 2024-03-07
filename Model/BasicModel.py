import torch
from abc import abstractmethod, ABC
import torch.nn.functional as F
from torch import nn


class BasicModel(torch.nn.Module):
    def __init__(self, parser):
        super().__init__()
        self.user_count = parser.user_count
        self.item_count = parser.item_count
        self.drop_out = parser.dropout
        self.emb_size = parser.emb_size
        # 经过初始化的embedding
        self.emb_user = torch.nn.Embedding(
            num_embeddings=self.user_count, embedding_dim=self.emb_size).to(parser.device)
        self.emb_item = torch.nn.Embedding(
            num_embeddings=self.item_count, embedding_dim=self.emb_size).to(parser.device)
        # 对权重正态分布初始化 std=0.1表示权重分布在均值附近
        nn.init.normal_(self.emb_user.weight, std=0.1)
        nn.init.normal_(self.emb_item.weight, std=0.1)
        # 未经过初始化的embedding:存储最后结果
        self.final_emb_user = torch.nn.Embedding(
            num_embeddings=self.user_count, embedding_dim=self.emb_size).to(parser.device)
        self.final_emb_item = torch.nn.Embedding(
            num_embeddings=self.item_count, embedding_dim=self.emb_size).to(parser.device)

    @abstractmethod
    def forward(self, users, seqs, posItems, negItems):
        pass

    @abstractmethod
    def getLoss(self, users, seqs, posItems, negItems):
        pass

    @abstractmethod
    def computer(self, args):
        pass

    @abstractmethod
    def predict(self, users, seqs, items):
        pass

    #TODO:改函数名
    def getFinalEmbed(self):
        """
        获取模型经过一轮迭代后的嵌入表示
        :return: 2个Embedding类型， self.finalEmbedUser, self.finalEmbedItem
        """
        # finalEmbedUser = F.normalize(self.finalEmbedUser.weight.data, p=2, dim=1)
        # finalEmbedItem = F.normalize(self.finalEmbedItem.weight.data, p=2, dim=1)
        emb_user = self.emb_user.weight.data
        emb_item = self.emb_item.weight.data
        return emb_user, emb_item






