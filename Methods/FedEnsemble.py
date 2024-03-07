from torch import nn
import torch
import torch.nn.functional as F
from sklearn.ensemble import VotingClassifier

from MyModules import *
from util import bpr_loss, ali_loss, uni_loss, loss_distance, l2_reg_loss


class FE(nn.Module):
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
        # self.client_user_emb_list = []
        self.client_item_emb_list = []
        self.set_server_emb(server_user_emb, server_item_emb)
        self.set_client_emb(client_uer_emb, client_item_emb)
        # 客户端数量
        self.client_count = len(self.client_item_emb_list) + 1
        # 一些工具
        self.logistic = torch.nn.Sigmoid().to(parser.device)
        self.layers = [self.emb_size * self.client_count, self.emb_size]  # MLP的layers
        self.DNN = MLP(layers=self.layers, parser=parser).to(parser.device)
        self.DNN_user = MLP(layers=[self.emb_size, self.emb_size], parser=parser).to(parser.device)
        # self.DNN = WideAndDeep(self.emb_size).to(parser.device)
        # self.DNN_user = WideAndDeep_client(self.emb_size).to(parser.device)
        # 每一个客户端设置一个decoder
        self.client_decoder_list = []
        for i in range(self.client_count - 1):
            self.client_decoder_list.append(MLP(layers=[self.emb_size, self.emb_size], parser=parser))
            # self.client_decoder_list.append(WideAndDeep_client(self.emb_size).to(parser.device))
        #通常用于特征抽取和转换
        # self.agg = Aggregation(layers=self.layers, parser=parser).to(parser.device)
        # self.agg = WAD_Aggregation(layers=self.layers, parser=parser).to(parser.device)
        self.agg = LightningAggregation(layers=self.layers).to(parser.device)
        #通常用于加权或调整某些输入特征的重要性，以用于后续的任务，例如注意力机制
        self.weightGate = nn.Sequential(
            nn.Linear(self.emb_size * self.client_count, self.emb_size * self.client_count * 2),
            nn.Linear(self.emb_size * self.client_count * 2, self.emb_size * self.client_count),
            nn.Linear(self.emb_size * self.client_count, self.client_count),
            nn.Softmax(dim=1)
        ).to(parser.device)
        self.sigmoid = nn.Sigmoid()
        self.score_list = []
        for i in range(self.client_count):
            self.score_list.append(Score(self.emb_size).to(parser.device))

    def set_server_emb(self, emb_user, emb_item):
        # 设置自己的embedding权重（与训练好的server embedding相同）
        self.emb_user.weight.data = emb_user
        self.emb_item.weight.data = emb_item

    def set_client_emb(self, emb_user_list, emb_item_list):
        #将离散的用户和物品特征映射到连续的嵌入空间中
        # for i in range(len(emb_user_list)):
        #     if self.is_normal_emb_user(emb_user_list[i]):
        #         emb_user = torch.nn.Embedding(num_embeddings=self.user_count, embedding_dim=self.emb_size).to(self.parser.device)
        #         emb_user.weight.data = emb_user_list[i]
        #         self.client_user_emb_list.append(emb_user)
        for i in range(len(emb_item_list)):
            if self.is_normal_emb_item(emb_item_list[i]):
                emb_item = torch.nn.Embedding(num_embeddings=self.item_count, embedding_dim=self.emb_size).to(self.parser.device)
                emb_item.weight.data = emb_item_list[i]
                self.client_item_emb_list.append(emb_item)

    # def is_normal_emb_user(self, emb_user):
    #     proximal_term = (emb_user.sum(axis=-1) - self.emb_user.weight.sum(axis=-1)).norm(2)
    #     proximal_term /= self.user_count
    #     if proximal_term < 0.1:
    #         return True
    #     return False

    def is_normal_emb_item(self, emb_item):
        proximalT_term = (emb_item.sum(axis=-1) - self.emb_item.weight.data.sum(axis=-1)).norm(2)
        proximalT_term /= self.item_count
        if proximalT_term < 0.1:
            return True
        return False

    # 总的一个forward
    def forward(self, users=None, pos_items=None, neg_items=None):
        emb_user, emb_item_list = self.computer_DNN_list()

        # all_emb_item = torch.cat(emb_item_list, dim=1)
        # weight_item = self.weightGate(all_emb_item)
        # weight_items = weight_item.chunk(self.client_count, dim=1)
        # pos_model_list, neg_model_list, pos_scores_list, neg_scores_list = [], [], [], []
        # weight_list = [1, 0.8, 0.9]

        pos_score_list, neg_score_list = [], []
        pos_item_weight_list, neg_item_weight_list = [], []
        pos_score, neg_score = 0, 0
        for i in range(self.client_count):
            pos_item_weight = self.score_list[i](emb_item_list[i])
            # neg_item_weight = self.score_list[i](emb_user, emb_item_list[i], users, neg_items)
            pos_item_weight_list.append(pos_item_weight)
            # neg_item_weight_list.append(neg_item_weight)
        for i in range(self.client_count):
            pos_score += (pos_item_weight_list[i][pos_items] * emb_user[users]).sum(dim=-1)
            neg_score += (pos_item_weight_list[i][neg_items] * emb_user[users]).sum(dim=-1)
            pos_score_list.append(pos_score)
            neg_score_list.append(neg_score)
            # p_score = (pos_item_weight_list[i][pos_items] * emb_user[users]).sum(dim=-1)
            # pos_score += p_score
            # n_score = (pos_item_weight_list[i][neg_items] * emb_user[users]).sum(dim=-1)
            # neg_score += n_score
            # pos_score_list.append(p_score)
            # neg_score_list.append(n_score)

        # for i in range(self.client_count):
        #     pos_score += self.score_list[i](emb_user, emb_item_list[i], users, pos_items)
        #     neg_score += self.score_list[i](emb_user, emb_item_list[i], users, neg_items)
        #     pos_score_list.append(pos_score)
        #     neg_score_list.append(neg_score)

        # ens_pos_scores, ens_neg_scores = 0, 0
        # for i in range(self.client_count):
        #     # ens_pos_scores += weight_list[i] * pos_model_list[i]
        #     # ens_neg_scores += weight_list[i] * neg_model_list[i]
        #     ens_pos_scores += weight_items[i][pos_items].squeeze() * pos_model_list[i]
        #     ens_neg_scores += weight_items[i][neg_items].squeeze() * neg_model_list[i]
        #     # ens_pos_scores += weight_items[i][users].squeeze() * pos_model_list[i]
        #     # ens_neg_scores += weight_items[i][users].squeeze() * neg_model_list[i]
        #     pos_scores_list.append(ens_pos_scores)
        #     neg_scores_list.append(ens_neg_scores)

        # score_weightgate = [1, 0]
        # all_emb_item = self.computer(emb_item_list)
        # emb_pos_item = all_emb_item[pos_items]
        # emb_neg_item = all_emb_item[neg_items]
        # emb_user = emb_user[users]
        # pos_scores = (emb_user * emb_pos_item).sum(dim=-1)
        # neg_scores = (emb_user * emb_neg_item).sum(dim=-1)
        #
        # pos_scores = score_weightgate[0] * pos_scores + score_weightgate[1] * ens_pos_scores
        # neg_scores = score_weightgate[0] * neg_scores + score_weightgate[1] * ens_neg_scores
        # return pos_scores, neg_scores
        return pos_score, neg_score, pos_score_list, neg_score_list,

    # 总的一个forward
    def predict(self, users, items, seq):
        emb_user, emb_item_list = self.computer_DNN_list()

        # all_emb_item = torch.cat(emb_item_list, dim=1)
        # weight_item = self.weightGate(all_emb_item)
        # weight_items = weight_item.chunk(self.client_count, dim=1)
        # model_list = []
        # weight_list = [1, 0.8, 0.9]
        # for i in range(self.client_count):
        #     model = self.score(emb_user[users], emb_item_list[i][items])
        #     model_list.append(model)
        scores = 0
        item_weight_list = []
        for i in range(self.client_count):
            item_weight = self.score_list[i](emb_item_list[i])
            item_weight_list.append(item_weight)
        for i in range(self.client_count):
            scores += (item_weight_list[i][items] * emb_user[users]).sum(dim=-1)



        # for i in range(self.client_count):
        #     scores += self.score_list[i](emb_user, emb_item_list[i], users, items)
        #     # ens_scores += weight_items[i][items].squeeze() * model_list[i]

        # score_weightgate = [1, 0]
        # all_emb_item = self.computer(emb_item_list)
        # emb_user = emb_user[users]
        # emb_item = all_emb_item[items]
        # scores = (emb_user * emb_item).sum(dim=-1)
        #
        # scores = score_weightgate[0] * scores + score_weightgate[1] * ens_scores
        # return scores
        return scores

    # 计算server的DNN
    def computer_DNN(self):
        # server和client的user embedding拼接
        # emb_user_list = [self.emb_user.weight]  #收集服务器的用户嵌入权重
        emb_user = self.emb_user.weight
        # for e in self.client_user_emb_list:  #收集客户端的用户嵌入权重
        #     emb_user_list.append(e.weight)
        # emb_user = torch.cat(emb_user_list, dim=-1)
        # server和client的item embedding拼接
        emb_item_list = [self.emb_item.weight]
        for e in self.client_item_emb_list:
            emb_item_list.append(e.weight)
        emb_item = torch.cat(emb_item_list, dim=-1)
        # 放入DNN，进行线性变换和非线性激活
        # emb_item = self.agg(emb_item)
        # emb_item = self.emb_item.weight
        emb_user = self.DNN_user(emb_user)
        emb_item = self.DNN(emb_item)
        # emb_item = self.DNN_user(emb_item)
        return emb_user, emb_item

    # 计算client的Decoder
    def computer_decoder(self, client_index, client_emb_item):
        # 和server一样的DNN计算
        decoder = self.client_decoder_list[client_index]
        # client_emb_user = decoder(client_emb_user)
        client_emb_item = decoder(client_emb_item)
        # return client_emb_user, client_emb_item
        return client_emb_item

    def computer_DNN_list(self):
        # 计算client和server的DNN
        server_emb_user_dnn, server_emb_item_dnn = self.computer_DNN()
        # emb_user_list, emb_item_list = [server_emb_user_dnn], [server_emb_item_dnn]  #0为server，1、2为两个client的decoder
        emb_item_list = [server_emb_item_dnn]
        for i in range(self.client_count - 1):
            # dec
            # client_emb_user_dec, client_emb_item_dec = self.computer_decoder(i, self.client_user_emb_list[i].weight, self.client_item_emb_list[i].weight)
            client_emb_item_dec = self.computer_decoder(i, self.client_item_emb_list[i].weight)
            # emb_user_list.append(client_emb_user_dec)
            emb_item_list.append(client_emb_item_dec)
        return server_emb_user_dnn, emb_item_list

    def computer(self, emb_item_list):
        all_emb_item = torch.cat(emb_item_list, dim=1)
        # 将这些embedding放入聚合器中
        agg_emb_item = self.agg(all_emb_item)  # 对item embedding进行变换和聚合
        weight_item = self.weightGate(all_emb_item)
        weight_items = weight_item.chunk(self.client_count, dim=1)
        # 将聚合后的embedding和权重信息进行加权
        # emb_item = 0
        # emb_item = agg_emb_item
        emb_item = emb_item_list[0] * weight_items[0]
        for i in range(1, len(weight_items)):
            emb_item += emb_item_list[i] * weight_items[i]
        # for i in range(len(self.weight_list)):
        #     emb_item += emb_item_list[i] * self.weight_list[i]
        emb_item = 0.5 * agg_emb_item + 0.5 * emb_item
        return emb_item

    def get_loss(self, users, pos_items, neg_items, seq):
        pos_score, neg_score, pos_scores_list, neg_scores_list = self.forward(users, pos_items, neg_items)
        loss = - (pos_score - neg_score).sigmoid().log().mean()
        # if torch.isnan(loss):



        # weight_list = [1, 1, 1]
        for i in range(self.client_count):
            # criterion = nn.MSELoss()
            # loss += criterion(pos_scores_list[i], neg_scores_list[i]) * 0.01
            loss += -(pos_scores_list[i] - neg_scores_list[i]).sigmoid().log().mean() * 0.01
        return loss, 0



