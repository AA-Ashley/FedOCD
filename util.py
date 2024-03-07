import torch
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch.nn.functional as F

from Model.LightGCN import LightGCN
from Model.MF import MatrixFactorization as MF
from Model.BPR import BPR
from Model.GRU import GRU
from Model.SASRec import SASRec
# from BOD import BOD


'''排名结果'''
def rank_result(parser, rank, k):
    device = parser.device
    """
    rank:用户行为的物品得分在101个物品得分中的排名
    MRR:Mean Reciprocal Rank=(1 / Q) * Sum(1 / rank_i) i属于1到Q Q为用户总数
    HIT:
    '"""
    MRR, HIT, NDCG = 0, 0, 0
    for v in rank:
        MRR += 1.0 / v
        if v <= k:
            HIT += 1
    for v in rank:
        if v > k:
            continue
        else:
            NDCG += torch.log(torch.Tensor([2])) / torch.log(torch.Tensor([v + 1]))
    MRR = MRR / len(rank)
    HIT = HIT / len(rank)
    NDCG = NDCG / len(rank)
    NDCG = NDCG[0].to(device).item() if NDCG != 0 else NDCG
    return MRR.to(device).item(), HIT, NDCG

'''加载embedding'''
def load_embedding(model, dataset, parser):
    user_path = "./Save/{0}/{1}/user_emb.pth".format(model, dataset)
    item_path = "./Save/{0}/{1}/item_emb.pth".format(model, dataset)
    # user_path = "./Save/test/{0}/{1}/userEmbed.pth".format(model, dataset)
    # item_path = "./Save/test/{0}/{1}/itemEmbed.pth".format(model, dataset)
    user_emb = torch.load(user_path, map_location=parser.device)
    item_emb = torch.load(item_path, map_location=parser.device)
    return user_emb, item_emb

def load_BOD_embedding(model, dataset, parser):
    user_path = "./Save/{0}/{1}/{2}user_emb.pth".format('BOD', model, dataset)
    item_path = "./Save/{0}/{1}/{2}item_emb.pth".format('BOD', model, dataset)
    user_emb = torch.load(user_path, map_location=parser.device)
    item_emb = torch.load(item_path, map_location=parser.device)
    return user_emb, item_emb

'''加载所使用的模型'''
def load_model(parser, adj):
    # TODO:不知道layer是不是根据模型改变的
    if parser.model_name == 'LightGCN':
        model = LightGCN(parser, adj)
    elif parser.model_name == 'MF':
        model = MF(parser)
    elif parser.model_name == 'BPR':
        model = BPR(parser)
    elif parser.model_name == 'GRU':
        model = GRU(parser)
    elif parser.model_name == 'SASRec':
        model = SASRec(parser)
    else:
        print("Wrong model name")
    return model

'''将numpy的coo矩阵转化为pytorch可以使用的矩阵'''
def coo_to_torch(parser, matrix: sp.coo_matrix):
    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))  # ndarray:垂直堆叠矩阵的行和列
    i = torch.LongTensor(indices)  # 将行和列转换为tensor
    v = torch.Tensor(values)  # 将数据转换为tensor
    shape = matrix.shape
    torch_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))  # 转换为torch可使用的矩阵
    torch_matrix = torch_matrix.float()
    return torch_matrix.to(parser.device)

'''创建稀疏二部邻接'''
def create_sparse_binary_matrix(parser, data, self_connection=False):
    n_nodes = parser.user_count + parser.item_count
    row_idx = [pair[0] for pair in data]  # 用户索引
    col_idx = [pair[1] for pair in data]  # 物品索引
    user_np = np.array(row_idx)
    item_np = np.array(col_idx)
    ratings = np.ones_like(user_np, dtype=np.float32)
    tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + parser.user_count)), shape=(n_nodes, n_nodes), dtype=np.float32)
    adj_mat = tmp_adj + tmp_adj.T
    if self_connection:
        adj_mat += sp.eye(n_nodes)
    return adj_mat

'''标准化'''
def normalize_graph_mat(adj_mat):
    shape = adj_mat.get_shape()
    row_sum = np.array(adj_mat.sum(1))
    if shape[0] == shape[1]:
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
    else:
        d_inv = np.power(row_sum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat)
    return norm_adj_mat

def convert_sparse_mat_to_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)

def calGraph(adjacency: sp.coo_matrix):
    """
    生成归一化矩阵
    :param adjacency: 邻接矩阵
    :return: 归一化矩阵D-1/2 A D-1/2
    """
    adjacency = adjacency.tolil()
    adjacencyUU = sp.identity(adjacency.shape[0])
    adjacencyII = sp.identity(adjacency.shape[1])
    adjacencyT = adjacency.T
    adjacencyUI = sp.hstack([adjacencyUU, adjacency])
    adjacencyIU = sp.hstack([adjacencyT, adjacencyII])
    adjacencyMatrix = sp.vstack([adjacencyUI, adjacencyIU])
    adjacencyMatrix = adjacencyMatrix.tolil()
    print("——————————————————————————————————————{0}——————————————————————————————————————".format("获得邻接矩阵"))
    degrees = np.array(adjacencyMatrix.sum(1))  # 按行求和得到rowsum, 即每个节点的度
    dInvSqrt = np.power(degrees, -0.5).flatten()  # (行和rowsum)^(-1/2)
    dInvSqrt[np.isinf(dInvSqrt)] = 0.  # isinf部分赋值为0
    d_mat_inv_sqrt = sp.diags(dInvSqrt)  # 对角化; 将d_inv_sqrt 赋值到对角线元素上, 得到度矩阵^-1/2
    return adjacencyMatrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # (度矩阵^-1/2)*邻接矩阵*(度矩阵^-1/2)

'''随着client训练生成图像'''
def draw_client_figure(fig_name, fig_path, fig_index, fig_value):
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    plt.plot(fig_index, fig_value)
    plt.xlabel("epoch")
    plt.ylabel("sum")
    plt.title(fig_name)
    plt.savefig(fig_path + ".png")
    plt.show()
    pass

'''写日志'''
def write_log(parser, text):
    note = open("./Log/log_{0}_{1}{2}.txt".format(
        parser.model_name, parser.data_name, parser.data_type), "a+")
    note.write(text + "\n")
    note.close()


"""以下BOD专用"""
def similarity(item_emb_1, item_emb_2):
    sim = F.cosine_similarity(item_emb_1, item_emb_2)
    return sim.view(-1, 1)


def bpr_loss(pos_score, neg_score, weight_pos, weight_neg):
    # bpr loss
    pos_scores = weight_pos * pos_score
    neg_scores = weight_neg * neg_score
    # TODO:这里的10e-8是什么意思
    bpr_loss = -torch.log(10e-8 + torch.sigmoid(pos_scores - neg_scores))
    return torch.mean(bpr_loss)

def ali_loss(user_emb, item_emb, weight):
    x, y = F.normalize(user_emb, dim=-1), F.normalize(item_emb, dim=-1)
    ali_loss = (x - y).norm(p=2, dim=1).pow(2)
    return (weight * ali_loss).mean()

    # x, y = F.normalize(user_emb, dim=-1), F.normalize(item_emb, dim=-1)
    # ali_loss = torch.empty(x.size(0), y.size(0))
    # for i in range(x.size(0)):
    #     for j in range(y.size(0)):
    #         ali = (x[i] - y[j]).norm(p=2, dim=-1).pow(2)
    #         ali_loss[i, j] = ali
    return (weight * ali_loss).mean()

# 计算uniformity loss方便
def uniformity_loss(x, t=2):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def uni_loss(user_emb, pos_item_emb, neg_item_emb):
    # uniformity loss
    uni_loss = (uniformity_loss(user_emb) + uniformity_loss(pos_item_emb) + uniformity_loss(neg_item_emb)) / 3
    return uni_loss

# 计算两个loss之间距离
def loss_distance(gwr, gws):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(
        1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)
    return emb_loss * reg


# 对数据的user id重新编号
# def DataProcess(data_dir):
#     rawdata = pd.read_csv(data_dir)
#     new_data = pd.DataFrame()
#     userCount, itemCount = 0, 0
#     userSet, itemSet = None, None
#
#     '''对数据进行重新编号'''
#     userMap, itemMap = {}, {}  #存储原始ID到新ID的映射关系
#     newDataList = []
#     userId, itemId = 1, 1
#     for index, row in rawdata.iterrows():
#         uid, iid = row[0], row[1]  #当前user、item id
#         if uid in userMap:
#             row[0] = userMap[uid]
#         else:
#             userMap[uid] = userId
#             row[0] = userId
#             userId += 1
#         if iid in itemMap:
#             row[1] = itemMap[iid]
#         else:
#             itemMap[iid] = itemId
#             row[1] = itemId
#             itemId += 1
#         new_data = new_data.append(row)
#         if len(new_data) == 10000:
#             newDataList.append(new_data)
#             new_data = pd.DataFrame()
#             print("完成第{0}个".format(index + 1))
#     userCount, itemCount = userId, itemId
#
#     userSet = set([x for x in range(1, userCount)])
#     itemSet = set([x for x in range(1, itemCount)])
#
#     '''划分数据集，分成test和train '''
#     # 生成邻接矩阵  协同过滤模型
#     train_data = rawdata.sample(frac=0.8, random_state=16)  #随机选择80%的数据作为训练集（使用sample函数和随机种子）
#     test_data = rawdata.drop(train_data.index).values.tolist()  #剩下的作为测试集
#     train_data = train_data.values.tolist()
#     return train_data, test_data, userCount, itemCount


