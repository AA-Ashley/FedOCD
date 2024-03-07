import torch
from torch import optim
from torch.utils.data import DataLoader
from datetime import datetime

from config import config
from util import rank_result, write_log, load_embedding
from Data.data import TrainDataset, TestDataset
from Baselines.MyModel import MyModel
from Methods.FedBOD import FedBOD
from Baselines.Oh_FedRec_without_user import OFR
from Methods.FedEnsemble_seq import FES
from Methods.FedEnsemble_BOD import FEB
from Methods.FedEnsemble import FE
from Baselines.FedAvg_One import FAO
from Baselines.P2FCDR import P2FCDR


class Server:
    def __init__(self, parser, model):
        self.parser = parser
        self.model = model
        self.epochs = parser.epochs
        self.device = parser.device
        self.k = parser.k
        # 过滤掉不需要梯度更新的参数 filter:过滤 lambda:匿名函数
        self.opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=parser.lr)
        self.train_data = TrainDataset(parser)
        self.test_data = TestDataset(parser)
        self.train_loader = DataLoader(self.train_data, batch_size=parser.batch_size, shuffle=True)  #在每个epoch之前对数据进行重新洗牌
        self.test_loader = DataLoader(self.test_data, batch_size=parser.batch_size,shuffle=True)
        self.step, self.iterator = 0, iter(self.train_loader)
        self.best_result = [0, 0, 0]
        self.best_result_epoch, self.down_epoch = 0,0

    def train(self):
        epoch_list, result_list = [], []
        for epoch in range(1, self.epochs):
            self.train_one_epoch()
            print("epoch = {0}已经完成".format(epoch))
            # 评估
            if epoch % 2 == 0:
                result = self.evaluate(epoch)
                epoch_list.append(epoch)
                result_list.append(result)
            if self.down_epoch >= self.parser.down_epoch:
                break
            ###############################
            if self.parser.model_name == 'FedBOD' or self.parser.model_name == 'FEB':
                # BOD专属
                # TODO：有问题还是有问题这个batch size
                # # 随机选取batch size个数据
                data_loader_out = DataLoader(self.train_data, batch_size=self.parser.batch_size, shuffle=True)
                data_iter = iter(data_loader_out)
                user, pos_item, neg_item, seq = next(data_iter)
                user, pos_item, neg_item, seq = user.to(self.device), pos_item.to(self.device), neg_item.to(self.device), seq.to(self.device)
                loss = self.model.out_loss(user, pos_item, neg_item)

                self.opt.zero_grad()
                self.model.generator_opt.zero_grad()
                loss.backward()
                self.model.generator_opt.step()
                print("out loss = {0}".format(loss))
                self.model.model_generator.eval()
            ###############################
        return epoch_list, result_list


    def train_one_epoch(self):
        # final_loss = 0
        inner_loss = 0
        all_zero = True
        device = self.parser.device
        # TODO:JDBuy数据集只有1个batch
        for i, batch in enumerate(self.train_loader):
            user, pos_item, neg_item, seq = batch
            user, pos_item, neg_item, seq = user.to(device), pos_item.to(device), neg_item.to(device), seq.to(self.device)
            # TODO：reg_loss不知道用处
            loss, reg_loss = self.model.get_loss(user, pos_item, neg_item, seq)
            ##########################################
            if self.parser.model_name == 'FedBOD' or self.parser.model_name == 'FEB':
                self.model.generator_opt.zero_grad()
                # self.model.model_generator.zero_grad()
            # self.model.zero_grad()
            self.opt.zero_grad()
            ##########################################

            loss.backward()
            self.opt.step()
            # final_loss += loss
            inner_loss += loss
        inner_loss = inner_loss / len(self.train_loader)
        print("loss = {0}".format(inner_loss))
        return inner_loss
        # final_loss = final_loss / len(self.train_loader)
        # print("loss = {0}".format(final_loss))
        # return final_loss

    def evaluate(self, epoch=0):
        rank = None
        device = self.parser.device
        for step, data in enumerate(self.test_loader):
            user, pos_item, neg_item, seq = data
            user, pos_item, neg_item, seq = user.to(device), pos_item.to(device), neg_item.to(device), seq.to(self.device)
            epoch_rank = torch.tensor([1] * len(user)).to(device)  # 创建每个用户的排名rank张量 初始值为1
            pos_score = self.model.predict(user, pos_item, seq)  # 通过user的embedding和pos item的embedding计算分数
            for i in range(self.parser.neg_count):
                neg_score = self.model.predict(user, neg_item[:, i], seq)
                res = ((pos_score - neg_score) <= 0).to(device)
                epoch_rank = epoch_rank + res
            if rank is None:
                rank = epoch_rank
            else:
                rank = torch.cat([rank, epoch_rank], dim=0)
        MRR, HIT5, NDCG5 = rank_result(self.parser, rank, self.k)
        MRR, HIT5, NDCG5 = round(MRR, 4), round(HIT5, 4), round(NDCG5, 4)
        text = ("best epoch = {0} model = {1} down epoch = {2} dataset = {3}"
                "MRR = {4} HIT@{5} = {6} NDCG@{5} = {7} ").format(
                self.best_result_epoch, self.parser.model_name, self.down_epoch,
                self.parser.test_path, self.best_result[0], self.k, self.best_result[1], self.best_result[2])

        if sum(self.best_result) - MRR - HIT5 - NDCG5 < 0:
            self.best_result = [MRR, HIT5, NDCG5]
            self.best_result_epoch = epoch
            self.down_epoch = 0
            text = ("time = {0}\t best epoch = {1}\t model = {2}\t down epoch = {3}\t "
                    "dataset = {4}\t MRR = {5}\t HIT@{6} = {7}\t NDCG@{6} = {8} ").format(
                datetime.now(), self.best_result_epoch, self.parser.model_name, self.down_epoch,
                self.parser.test_path, self.best_result[0], self.k, self.best_result[1], self.best_result[2])
            write_log(self.parser, text)
        else:
            self.down_epoch += 2
        print(text)
        return MRR + HIT5 +NDCG5


if __name__ == '__main__':
    parser = config()
    data = parser.data_name
    if data == 'JD':
        # server_emb_user, server_emb_item = load_embedding('LightGCN', data + 'Car', parser)
        # client_emb_user1, client_emb_item1 = load_embedding('MF', data + 'Buy', parser)
        # client_emb_user2, client_emb_item2 = load_embedding('GRU', data + 'Like', parser)
        # server_emb_user, server_emb_item = load_embedding('LightGCN', data + 'Buy', parser)
        # client_emb_user1, client_emb_item1 = load_embedding('MF', data + 'Car', parser)
        # client_emb_user2, client_emb_item2 = load_embedding('GRU', data + 'Like', parser)
        server_emb_user, server_emb_item = load_embedding('LightGCN', data + 'Like', parser)
        client_emb_user1, client_emb_item1 = load_embedding('MF', data + 'Car', parser)
        client_emb_user2, client_emb_item2 = load_embedding('GRU', data + 'Buy', parser)
    elif data == 'Tmall':
        # server_emb_user, server_emb_item = load_embedding('LightGCN', data + 'Buy', parser)
        # client_emb_user1, client_emb_item1 = load_embedding('MF', data + 'Like', parser)
        # client_emb_user2, client_emb_item2 = load_embedding('GRU', data + 'Car', parser)
        # server_emb_user, server_emb_item = load_embedding('LightGCN', data + 'Like', parser)
        # client_emb_user1, client_emb_item1 = load_embedding('MF', data + 'Buy', parser)
        # client_emb_user2, client_emb_item2 = load_embedding('GRU', data + 'Car', parser)
        server_emb_user, server_emb_item = load_embedding('LightGCN', data + 'Car', parser)
        client_emb_user1, client_emb_item1 = load_embedding('MF', data + 'Buy', parser)
        client_emb_user2, client_emb_item2 = load_embedding('GRU', data + 'Like', parser)
    elif data == 'Ten':
        # server_emb_user, server_emb_item = load_embedding('LightGCN', data + 'Share', parser)
        # client_emb_user1, client_emb_item1 = load_embedding('MF', data + 'Like', parser)
        # client_emb_user2, client_emb_item2 = load_embedding('GRU', data + 'Follow', parser)
        # server_emb_user, server_emb_item = load_embedding('LightGCN', data + 'Like', parser)
        # client_emb_user1, client_emb_item1 = load_embedding('MF', data + 'Share', parser)
        # client_emb_user2, client_emb_item2 = load_embedding('GRU', data + 'Follow', parser)
        server_emb_user, server_emb_item = load_embedding('LightGCN', data + 'Follow', parser)
        client_emb_user1, client_emb_item1 = load_embedding('MF', data + 'Share', parser)
        client_emb_user2, client_emb_item2 = load_embedding('GRU', data + 'Like', parser)
    else:
        print("data name error")
    '''Original'''
    # # 读取embedding
    # server_emb_user, server_emb_item = load_embedding('BPR', data + 'Buy', parser)
    # # server_emb_user, server_emb_item = load_BOD_embedding('LightGCN', data + 'Buy', parser)
    # # server_emb_user, server_emb_item = load_embedding('LightGCN', data + 'Buy', parser)
    # client_emb_user1, client_emb_item1 = load_embedding('MF', data + 'Like', parser)
    # # client_emb_user1, client_emb_item1 = load_BOD_embedding('LightGCN', data + 'Like', parser)
    # client_emb_user2, client_emb_item2 = load_embedding('LightGCN', data + 'Car', parser)
    # # client_emb_user2, client_emb_item2 = load_embedding('BPR', data + 'Car', parser)
    # # client_emb_user2, client_emb_item2 = load_BOD_embedding('LightGCN', data + 'Car', parser)


    client_emb_user_list, client_emb_item_list = [client_emb_user1, client_emb_user2], [client_emb_item1, client_emb_item2]
    # 加载模型
    # train_model = MyModel(parser, server_emb_user, server_emb_item, client_emb_user_list, client_emb_item_list)
    if parser.model_name == 'OFR':
        train_model = OFR(parser, server_emb_user, server_emb_item, client_emb_user_list, client_emb_item_list)
    elif parser.model_name == 'FedBOD':
        train_model = FedBOD(parser, server_emb_user, server_emb_item, client_emb_user_list, client_emb_item_list)
    elif parser.model_name == 'FE':
        train_model = FE(parser, server_emb_user, server_emb_item, client_emb_user_list, client_emb_item_list)
    elif parser.model_name == 'MyModel':
        train_model = MyModel(parser, server_emb_user, server_emb_item, client_emb_user_list, client_emb_item_list)
    elif parser.model_name == 'FEB':
        train_model = FEB(parser, server_emb_user, server_emb_item, client_emb_user_list, client_emb_item_list)
    elif parser.model_name == 'FES':
        train_model = FES(parser, server_emb_user, server_emb_item, client_emb_user_list, client_emb_item_list)
    elif parser.model_name == 'FAO':
        train_model = FAO(parser, server_emb_user, server_emb_item, client_emb_user_list, client_emb_item_list)
    elif parser.model_name == 'P2FCDR':
        train_model = P2FCDR(parser, server_emb_user, server_emb_item, client_emb_user_list, client_emb_item_list)
    else:
        print("model name error")
    server = Server(parser, train_model)
    epoch_list, result_list = server.train()


    # fig_name = "the effect of {0} in {1}{2}".format(parser.model_name, parser.data_name,
    #                                                 parser.data_type)
    # fig_path = "./Figure/Client/" + fig_name
    # draw_client_figure(fig_name,fig_path, epoch_list, result_list)
    # print(parser)
