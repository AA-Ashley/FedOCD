from config import config
from util import load_embedding
from Methods.BOD import BOD
from Baselines.MyModel import MyModel

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = config()
    data = 'JD'
    # 读取embedding
    server_emb_user, server_emb_item = load_embedding( 'BPR', data+'Buy', parser)
    client_emb_user1, client_emb_item1 = load_embedding('MF', data+'Like', parser)
    client_emb_user2, client_emb_item2 = load_embedding('LightGCN', data+'Car', parser)
    client_emb_user_list, client_emb_item_list = [client_emb_user1, client_emb_user2], [client_emb_item1, client_emb_item2]
    # 加载模型
    train_model = MyModel(parser, server_emb_user, server_emb_item, client_emb_user_list, client_emb_item_list)
    model = BOD(parser, train_model)
    model.train()

    # train_model = load_model(parser)
    # model = BOD(parser, train_model)

    # model = BOD(parser)
    # model.train()
    print("end")

