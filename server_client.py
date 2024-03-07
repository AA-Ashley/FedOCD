import copy

import torch
import os
import util
from torch import optim
from torch.utils.data import DataLoader
from datetime import datetime

from config_model import config
from util import load_model, rank_result, write_log, draw_client_figure, create_sparse_binary_matrix, normalize_graph_mat, convert_sparse_mat_to_tensor
from Data.data import TrainDataset, TestDataset


class Client:
    def __init__(self, parser):
        self.parser = parser
        self.epochs = parser.epochs
        # 数据集
        self.train_data = TrainDataset(parser)
        self.test_data = TestDataset(parser)
        self.train_loader = DataLoader(self.train_data, batch_size=parser.batch_size, shuffle=True)  #在每个epoch之前对数据进行重新洗牌
        self.test_loader = DataLoader(self.test_data, batch_size=parser.batch_size,shuffle=True)
        # 图数据矩阵
        self.ui_adj = create_sparse_binary_matrix(parser, self.train_data)
        self.norm_adj = normalize_graph_mat(self.ui_adj)
        self.sparse_norm_adj = convert_sparse_mat_to_tensor(self.norm_adj)
        # 模型
        self.model = load_model(parser, self.sparse_norm_adj)
        # 过滤掉不需要梯度更新的参数 filter:过滤 lambda:匿名函数
        self.opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=parser.lr)

        self.step, self.iterator = 0, iter(self.train_loader)
        self.best_result = [0, 0, 0]
        self.best_result_epoch, self.down_epoch = 0,0

    def train(self):
        epoch_list, result_list = [], []
        self.down_epoch = 0
        for epoch in range(1, self.epochs):
            self.train_one_epoch()
            print("epoch = {0}已经完成".format(epoch))
            # 评估
            if epoch % 2 == 0:
                result = self.evaluate(epoch)
                epoch_list.append(epoch)
                result_list.append(result)
            if self.down_epoch >= 10:
                break
        return epoch_list, result_list


    def train_one_epoch(self):
        final_loss = 0
        all_zero = True
        device = self.parser.device
        # TODO:JDBuy数据集只有1个batch
        for i, batch in enumerate(self.train_loader):
            user, pos_item, neg_item, seq = batch
            user, pos_item, neg_item, seq = user.to(device), pos_item.to(device), neg_item.to(device), seq.to(device)
            # TODO：reg_loss不知道用处
            loss, reg_loss = self.model.get_loss(user, pos_item, neg_item, seq)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            final_loss += loss
        final_loss = final_loss / len(self.train_loader)
        print("loss = {0}".format(final_loss))
        return final_loss

    def evaluate(self, epoch=0):
        rank = None
        device = self.parser.device
        for step, data in enumerate(self.test_loader):
            user, pos_item, neg_item, seq = data
            user, pos_item, neg_item = user.to(device), pos_item.to(device), neg_item.to(device)
            epoch_rank = torch.tensor([1] * len(user)).to(device)  # 创建每个用户的排名rank张量 初始值为1
            pos_score = self.model.predict(user, pos_item, seq)  # 通过user的embedding和pos item的embedding计算分数
            for i in range(self.parser.neg_count):
                neg_score = self.model.predict(user, neg_item[:, i], seq)
                res = ((pos_score - neg_score) <= 0).to(device)
                epoch_rank = epoch_rank +res
            if rank is None:
                rank = epoch_rank
            else:
                rank = torch.cat([rank, epoch_rank], dim=0)
        #TODO:这一段还没仔细看
        MRR, HIT5, NDCG5 = rank_result(self.parser, rank, 5)
        MRR, HIT5, NDCG5 = round(MRR, 4), round(HIT5, 4), round(NDCG5, 4)
        text = ("{0}: best epoch = {1} model = {2} down epoch = {3} dataset = {4}"
                "MRR = {5} HIT@5 = {6} NDCG@5 = {7} ").format(
                 self.parser.data_type, self.best_result_epoch, self.parser.model_name, self.down_epoch,
                self.parser.test_path, self.best_result[0], self.best_result[1], self.best_result[2])

        if sum(self.best_result) - MRR - HIT5 - NDCG5 < 0:
            self.best_result = [MRR, HIT5, NDCG5]
            self.best_result_epoch = epoch
            self.down_epoch = 0
            text = ("{0}: best epoch = {1}\t model = {2}\t down epoch = {3}\t "
                    "dataset = {4}\t MRR = {5}\t HIT@5 = {6}\t NDCG@5 = {7} time = {8}\t ").format(
                self.parser.data_type, self.best_result_epoch, self.parser.model_name, self.down_epoch,
                self.parser.test_path, self.best_result[0], self.best_result[1], self.best_result[2], datetime.now())
            write_log(self.parser, text)
        else:
            self.down_epoch += 2
        print(text)
        return MRR + HIT5 + NDCG5


if __name__ == '__main__':
    parser = config()
    client_list = []
    for i in ['Car', 'Buy', 'Like']:
        parser.data_type = i
        parser.train_path = parser.data_path + parser.data_name + parser.data_type + 'SequenceTrain.csv'
        parser.test_path = parser.data_path + parser.data_name + parser.data_type + 'SequenceTest.csv'
        client = Client(copy.deepcopy(parser))
        client_list.append(client)

    # 服务器初始化
    global_model = load_model(parser, 0)
    w_avg = global_model.state_dict()

    for epoch in range(1, 50):
        print("epoch = {0}\n".format(epoch))
        model_weight_list = []
        for i in range(len(client_list)):
            client_list[i].model.load_state_dict(copy.deepcopy(w_avg))
            epoch_list, result_list = client_list[i].train()
            model_weight_list.append(copy.deepcopy(client_list[i].model.state_dict()))

        w_avg = copy.deepcopy(model_weight_list[0])
        for k in w_avg.keys():
            for i in range(1, len(model_weight_list)):
                w_avg[k] += model_weight_list[i][k]
            w_avg[k] = torch.div(w_avg[k], len(model_weight_list))


    print(parser)
