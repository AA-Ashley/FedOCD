from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import scipy.sparse as sp



class MLP(nn.Module):
    def __init__(self, layers=None, parser=None):
        super().__init__()
        if layers is None:
            layers = [64, 32]
        self.layers = layers
        self.fcLayers = nn.ModuleList()  # 存储多个全连接层
        # 创建线性层
        for idx, (inSize, outSize) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fcLayers.append(torch.nn.Linear(inSize, outSize).to(parser.device))
        self.fcLayers.append(nn.ReLU().to(parser.device))  # 最后一层
        # 对线性层的权重进行初始化
        for m in self.fcLayers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, embed):
        for idx, _ in enumerate(range(len(self.fcLayers))):
            embed = self.fcLayers[idx](embed)
        return embed

class Aggregation(nn.Module):
    def __init__(self, layers=None, parser=None):
        super().__init__()
        if layers is None:
            layers = [64, 32]
        self.layers = layers
        self.fcLayers = nn.ModuleList()
        for idx, (inSize, outSize) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fcLayers.append(torch.nn.Linear(inSize, outSize).to(parser.device))
        self.func = nn.PReLU().to(parser.device)

        for m in self.fcLayers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, embed):
        for idx, _ in enumerate(range(len(self.fcLayers))):
            embed = self.fcLayers[idx](embed)
        embed = self.func(embed)
        return embed

class GRU(nn.Module):
    def __init__(self, emb_size, device):

        super().__init__()
        self.gru = nn.GRU(emb_size, emb_size, 1, bias=False).to(device)
        self.logistic = torch.nn.Sigmoid().to(device)

    def forward(self, seq_emb):
        output, temp = self.gru(seq_emb)
        output = self.logistic(output)[:, -1, :]
        return output


class GCN(nn.Module):
    def __init__(self, parser):
        super(GCN, self).__init__()
        self.parser = parser
        graph = sp.load_npz(self.parser.lap_matrix)
        self.graph = coo_to_torch(parser, graph)
        self.graph = self.graph.coalesce()


        self.keep_prob = 1 - parser.dropout
        self.layers = parser.layers
        self.f = nn.Sigmoid()

    def dropout(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def forward(self, embedUser, embedItem):
        allEmbed = torch.cat([embedUser, embedItem])
        embeds = [allEmbed]
        graph = self.dropout(self.graph, keep_prob=self.keep_prob)
        for layer in range(self.layers):
            tempEmbed = [torch.sparse.mm(graph, allEmbed)]
            side_emb = torch.cat(tempEmbed, dim=0)
            allEmbed = side_emb
            embeds.append(allEmbed)
        embeds = torch.stack(embeds, dim=1)
        output = torch.mean(embeds, dim=1)
        users, items = torch.split(output, [self.parser.user_count, self.parser.item_count])
        return users, items

class WideAndDeep(nn.Module):
    def __init__(self, embedding_dim):
        super(WideAndDeep, self).__init__()
        self.fc_wide = nn.Linear(embedding_dim * 3, embedding_dim)
        self.fc_deep = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, embed):
        wide_out = self.fc_wide(embed)
        deep_out = self.fc_deep(embed)
        return wide_out + deep_out

class WideAndDeep_client(nn.Module):
    def __init__(self, embedding_dim):
        super(WideAndDeep_client, self).__init__()
        self.fc_wide = nn.Linear(embedding_dim, embedding_dim)
        self.fc_deep = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, embed):
        wide_out = self.fc_wide(embed)
        deep_out = self.fc_deep(embed)
        return wide_out + deep_out

class LightningAggregation(pl.LightningModule):
    def __init__(self, layers=None):
        super(LightningAggregation, self).__init__()
        if layers is None:
            layers = [64, 32]

        # 构建线性层和激活函数的序列
        layer_list = [nn.Linear(inSize, outSize) for inSize, outSize in zip(layers[:-1], layers[1:])]
        layer_list.append(nn.PReLU())

        # 使用 Sequential 定义层的堆叠
        self.fcLayers = nn.Sequential(*layer_list)

        # 对线性层的权重进行初始化
        for m in self.fcLayers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, embed):
        return self.fcLayers(embed)

class Score(nn.Module):
    def __init__(self, embedding_dim):
        super(Score, self).__init__()
        self.weight_emb = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Softmax(dim=1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, item_emb):
        weight = self.weight_emb(item_emb)

        # score = (user_emb[user] * item_emb[item]).sum(dim=-1)
        weight_score = weight * item_emb
        return weight_score


class Generator_VAE(nn.Module):
    def __init__(self, embedding_dim):
        super(Generator_VAE, self).__init__()
        self.weight_emb = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.Sigmoid(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )

    def forward(self, cur_user, user):
        distance = torch.norm(cur_user - user, p=2, dim=-1, keepdim=True)
        dis_scores = F.softmax(-distance, dim=-1)
        weight = self.weight_emb(dis_scores)

        # score = (user_emb[user] * item_emb[item]).sum(dim=-1)
        weight_score = weight * cur_user
        return weight_score

class AttentionWeightGenerator(nn.Module):
    def __init__(self, embedding_size):
        super(AttentionWeightGenerator, self).__init__()
        self.query_embedding = nn.Linear(embedding_size, embedding_size)
        self.key_embedding_1 = nn.Linear(embedding_size, embedding_size)
        self.key_embedding_2 = nn.Linear(embedding_size, embedding_size)
        self.key_embedding_3 = nn.Linear(embedding_size, embedding_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, user_embedding, item_embedding_1, item_embedding_2, item_embedding_3):
        # 计算查询嵌入
        query = self.query_embedding(user_embedding)

        # 计算键嵌入
        key_1 = self.key_embedding_1(item_embedding_1)
        key_2 = self.key_embedding_2(item_embedding_2)
        key_3 = self.key_embedding_3(item_embedding_3)

        # 计算相似性分数
        similarity_1 = torch.matmul(query, key_1.t())
        similarity_2 = torch.matmul(query, key_2.t())
        similarity_3 = torch.matmul(query, key_3.t())

        # 使用softmax计算权重
        weights_1 = self.softmax(similarity_1)
        weights_2 = self.softmax(similarity_2)
        weights_3 = self.softmax(similarity_3)

        return weights_1, weights_2, weights_3

class Sequential_Aggregation(nn.Module):
    def __init__(self, layers=None, parser=None):
        super(Sequential_Aggregation, self).__init__()
        if layers is None:
            layers = [64, 32]

        # 构建线性层和激活函数的序列
        layer_list = [nn.Linear(inSize, outSize) for inSize, outSize in zip(layers[:-1], layers[1:])]
        layer_list.append(nn.PReLU())

        # 使用 Sequential 定义层的堆叠
        self.fcLayers = nn.Sequential(*layer_list)

        # 对线性层的权重进行初始化
        for m in self.fcLayers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, embed):
        return self.fcLayers(embed)
# class NCF(nn.Module):
#     def __init__(self, embedding_dim):
#         super(NCF, self).__init__()
#         self.fc1 = nn.Linear(embedding_dim * 3, embedding_dim)
#         self.fc2 = nn.Linear(embedding_dim, 32)
#         self.prelu = nn.PReLU()
#
#     def forward(self, emb):
#         x = self.fc1(emb)
#         x = self.prelu(x)
#         x = self.fc2(x)
#         return x.squeeze()
#
# class NCF_client(nn.Module):
#     def __init__(self, embedding_dim):
#         super(NCF_client, self).__init__()
#         self.fc1 = nn.Linear(embedding_dim, embedding_dim)
#         self.fc2 = nn.Linear(embedding_dim, 32)
#         self.prelu = nn.PReLU()
#
#     def forward(self, emb):
#         x = self.fc1(emb)
#         x = self.prelu(x)
#         x = self.fc2(x)
#         return x.squeeze()


def coo_to_torch(parser, matrix: sp.coo_matrix):
    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))  # ndarray:垂直堆叠矩阵的行和列
    i = torch.LongTensor(indices)  # 将行和列转换为tensor
    v = torch.Tensor(values)  # 将数据转换为tensor
    shape = matrix.shape
    torch_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))  # 转换为torch可使用的矩阵
    torch_matrix = torch_matrix.float()
    return torch_matrix.to(parser.device)
