from torch import nn
import torch
import torch.nn.functional as F
from sklearn.ensemble import VotingClassifier

from MyModules import *
from util import bpr_loss, ali_loss, uni_loss, loss_distance, l2_reg_loss


class FEA(nn.Module):
    def __init__(self, parser, server_user_emb, server_item_emb, client_uer_emb, client_item_emb):
        super().__init__()
        # 基本信息
        self.parser = parser
        self.user_count = parser.user_count
        self.item_count = parser.item_count
        self.emb_size = parser.emb_size
        # 创建的自己嵌入
        self.emb_user = torch.nn.Embedding(
            num_embeddings=self.user_count, embedding_dim=self.emb_size).to(parser.device)
        self.emb_item = torch.nn.Embedding(
            num_embeddings=self.item_count, embedding_dim=self.emb_size).to(parser.device)
        # 初始化server和client的embedding
        self.client_user_emb_list = []
        # self.client_item_emb_list = []
        self.set_server_emb(server_user_emb, server_item_emb)
        self.set_client_emb(client_uer_emb, client_item_emb)
        # 客户端数量
        self.client_count = len(self.client_user_emb_list) + 1
        # 一些工具
        self.logistic = torch.nn.Sigmoid().to(parser.device)
        self.layers = [self.emb_size * self.client_count, self.emb_size]  # MLP的layers
        self.DNN = MLP(layers=self.layers, parser=parser).to(parser.device)
        self.DNN_item = MLP(layers=[self.emb_size, self.emb_size], parser=parser).to(parser.device)
        # 每一个客户端设置一个decoder
        self.client_decoder_list = []
        for i in range(self.client_count - 1):
            # self.client_decoder_list.append(MLP(layers=[self.emb_size, self.emb_size], parser=parser))
            self.client_decoder_list.append(WideAndDeep_client(self.emb_size).to(parser.device))
        # self.sigmoid = nn.Sigmoid()
        self.score_list = []
        self.score_opt = []
        for i in range(self.client_count):
            # self.score_list.append(Score(self.emb_size).to(parser.device))
            self.score_list.append(Generator_VAE(self.emb_size).to(parser.device))
            self.score_opt.append(torch.optim.Adam(self.score_list[i].parameters(), lr=parser.lr))

    def set_server_emb(self, emb_user, emb_item):
        # 设置自己的embedding权重（与训练好的server embedding相同）
        self.emb_user.weight.data = emb_user
        self.emb_item.weight.data = emb_item

    def set_client_emb(self, emb_user_list, emb_item_list):
        #将离散的用户和物品特征映射到连续的嵌入空间中
        for i in range(len(emb_user_list)):
            emb_user = torch.nn.Embedding(num_embeddings=self.user_count, embedding_dim=self.emb_size).to(self.parser.device)
            emb_user.weight.data = emb_user_list[i]
            self.client_user_emb_list.append(emb_user)


    # 总的一个forward
    def forward(self, users=None, pos_items=None, neg_items=None):
        emb_user_list, emb_item = self.computer_DNN_list()

        pos_score_list, neg_score_list = [], []
        user_weight_list = []
        pos_score, neg_score = 0, 0
        # for i in range(self.client_count):
        #     user_weight = self.score_list[i](emb_user_list[i], self.emb_user.weight)
        #     user_weight_list.append(user_weight)
        for i in range(self.client_count):
            pos_score += (emb_user_list[i][users] * emb_item[pos_items]).sum(dim=-1)
            neg_score += (emb_user_list[i][users] * emb_item[neg_items]).sum(dim=-1)
            pos_score_list.append(pos_score)
            neg_score_list.append(neg_score)

        return pos_score, neg_score, pos_score_list, neg_score_list

    # 总的一个forward
    def predict(self, users, items, seq):
        emb_user_list, emb_item = self.computer_DNN_list()

        scores = 0
        user_weight_list = []
        # for i in range(self.client_count):
        #     user_weight = self.score_list[i](emb_user_list[i], self.emb_user.weight)
        #     user_weight_list.append(user_weight)
        for i in range(self.client_count):
            scores += (emb_user_list[i][users] * emb_item[items]).sum(dim=-1)

        return scores / 4

    # 计算server的DNN
    def computer_DNN(self):
        # server和client的user embedding拼接
        emb_user_list = [self.emb_user.weight]  #收集服务器的用户嵌入权重
        for e in self.client_user_emb_list:  #收集客户端的用户嵌入权重
            emb_user_list.append(e.weight)
        emb_user = torch.cat(emb_user_list, dim=-1)
        # emb_user = self.emb_user.weight
        # server和client的item embedding拼接
        emb_item = self.emb_item.weight
        # emb_item_list = [self.emb_item.weight]
        # for e in self.client_item_emb_list:
        #     emb_item_list.append(e.weight)
        # emb_item = torch.cat(emb_item_list, dim=-1)

        # 放入DNN，进行线性变换和非线性激活
        emb_user = self.DNN(emb_user)
        emb_item = self.DNN_item(emb_item)
        return emb_user, emb_item

    # 计算client的Decoder
    def computer_decoder(self, client_index, client_emb_user):
        # 和server一样的DNN计算
        decoder = self.client_decoder_list[client_index]
        client_emb_user = decoder(client_emb_user)
        return client_emb_user

    def computer_DNN_list(self):
        # 计算client和server的DNN
        server_emb_user_dnn, server_emb_item_dnn = self.computer_DNN()
        # emb_user_list, emb_item_list = [server_emb_user_dnn], [server_emb_item_dnn]  #0为server，1、2为两个client的decoder
        emb_user_list = [server_emb_user_dnn]
        for i in range(self.client_count - 1):
            # dec
            # client_emb_user_dec, client_emb_item_dec = self.computer_decoder(i, self.client_user_emb_list[i].weight, self.client_item_emb_list[i].weight)
            client_emb_user_dec = self.computer_decoder(i, self.client_user_emb_list[i].weight)
            emb_user_list.append(client_emb_user_dec)
        return emb_user_list, server_emb_item_dnn


    def get_loss(self, users, pos_items, neg_items, seq):
        pos_score, neg_score, pos_scores_list, neg_scores_list = self.forward(users, pos_items, neg_items)
        # loss = 0
        loss = - (pos_score - neg_score).sigmoid().log().mean()

        # loss += nn.MSELoss()(pos_score, neg_score) * 0.01
        # if torch.isnan(loss):


        # for i in range(self.client_count):
        #     criterion = nn.MSELoss()
        #     loss += criterion(pos_scores_list[i], neg_scores_list[i]) * 0.01
            # loss += -(pos_scores_list[i] - neg_scores_list[i]).sigmoid().log().mean() * 0.01
        return loss, 0



