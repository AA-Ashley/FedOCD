
import torch
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp

import util
from Model.BasicModel import BasicModel

class LightGCN(BasicModel):
    def __init__(self, parser, adj):
        super(LightGCN, self).__init__(parser)
        self.parser = parser
        # graph = sp.load_npz(self.parser.lap_matrix)  #加载稀疏矩阵
        # self.graph = util.coo_to_torch(self.parser, graph)  #将numpy的coo矩阵转化为pytorch可以使用的矩阵
        self.layers = parser.layers
        self.keep_prob = 1 - parser.dropout  #保留概率
        # self.graph = self.graph.coalesce()  #将稀疏张量合并(m没有很懂怎么合并的)
        self.graph = adj.to(parser.device)
        self.f = nn.Sigmoid()

    def dropout(self, x, keep_prob):
        size = x.size()  # 稀疏矩阵的大小
        index = x.indices().t()  # 对稀疏矩阵中非零元素坐标进行转置
        values = x.values()  # 矩阵中非零元素值
        # 生成与values长度相同的随机张量，每个值+keep_prob（得到用于dropout的掩码）
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        # 使用dropout掩码对坐标和值进行筛选，只保留被掩码保留的坐标
        index = index[random_index]
        values = values[random_index] / keep_prob  # 除以keepProb是为了在测试时保持期望值的一致性
        # 使用筛选后的坐标和值重新构建一个稀疏张量矩阵
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def computer(self, args):
        """
        propagate methods for lightGCN
        """
        emb_user = self.emb_user.weight
        emb_item = self.emb_item.weight
        all_emb = torch.cat([emb_user, emb_item])
        emb_list = [all_emb]
        # 对lap_matrix的稀疏矩阵进行dropout操作（被重新筛选坐标和值）
        # graph = self.dropout(self.graph, keep_prob=self.keep_prob)
        # 多层的图embedding构建
        for layer in range(self.layers):
            temp_emb = [torch.sparse.mm(self.graph, all_emb)]  # 将上述矩阵与embedding相乘
            side_emb = torch.cat(temp_emb, dim=0)  # 垂直方向拼接:当前层embedding
            all_emb = side_emb  # 更新当前embedding为拼接后结果
            emb_list.append(all_emb)
        emb_list = torch.stack(emb_list, dim=1)  # 将embedding list沿着维度1堆叠
        output = torch.mean(emb_list, dim=1)
        users, items = torch.split(output, [self.user_count, self.item_count])
        return users, items

    def get_loss(self, users, pos_item, neg_item, seq):
        pos_scores, neg_scores, _, _ = self.forward(users, pos_item=pos_item, neg_item=neg_item, seq=seq)
        loss = - (pos_scores - neg_scores).sigmoid().log().mean()  # 负对数似然损失 使正样本得分高于负样本
        return loss, 0

    def forward(self, users=None, pos_item=None, neg_item=None, seq=None):
        # compute embedding
        all_users, all_items = self.computer(args=None)
        if users != None:
            emb_user = all_users[users]
            emb_pos_item = all_items[pos_item]
            emb_neg_item = all_items[neg_item]
            pos_scores = torch.mul(emb_user, emb_pos_item)  # 逐元素相乘
            pos_scores = torch.sum(pos_scores, dim=1)  # 在emb_size这个维度上相加
            neg_scores = torch.mul(emb_user, emb_neg_item)
            neg_scores = torch.sum(neg_scores, dim=1)
            self.final_emb_user.weight.data = self.emb_user.weight.data
            self.final_emb_item.weight.data = self.emb_item.weight.data
            return pos_scores, neg_scores, all_users, all_items
        return all_users, all_items

    def predict(self, users, items, seq):
        all_users, all_items = self.computer(args=None)
        emb_user = all_users[users]
        emb_item = all_items[items]
        score = torch.mul(emb_user, emb_item)
        score = torch.sum(score, dim=1)
        return score
