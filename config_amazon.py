import argparse

def config():
    parser = argparse.ArgumentParser()
    # 数据参数
    parser.add_argument('--data_path', type=str, default='./Data/', help='path of the dataset')
    parser.add_argument('--data_name', type=str, default='Amazon', help='name of the dataset')
    parser.add_argument('--data_type', type=str, default='Clothing', help='type of the dataset')
    parser.add_argument('--user_count', type=int, default=34792, help='Amazon:34792 Douban:2719')
    parser.add_argument('--item_count', type=int, default=0, help='total count of the items')
    parser.add_argument('--train_path', type=str, default='./Data/', help='path of the train dataset')
    parser.add_argument('--test_path', type=str, default='./Data/', help='path of the test dataset')
    parser.add_argument('--lap_matrix', type=str, default='./Data/', help='path of the laplacian matrix')
    parser.add_argument('--seq', type=int, default=5, help='length of sequence, 0 represents graph')
    # 基本参数
    parser.add_argument('--device', type=str, default='cuda:6', help='device')
    parser.add_argument('--batch_size', type=int, default='256', help='number of total iterations in each epoch')
    parser.add_argument('--lr', type=float, default='0.001', help='learning rate')
    parser.add_argument('--epochs', type=int, default='1000', help='number of rounds of training')
    parser.add_argument('--weight_decay', type=float, default='0.0', help='weight decay')
    parser.add_argument('--dropout', type=float, default='0.2', help='generate dropout mask')
    parser.add_argument('--down_epoch', type=int, default='20', help='down epoch')
    parser.add_argument('--k', type=int, default=10, help='top k')

    # 模型参数
    parser.add_argument('--model_name', type=str, default='FEA', help='name of the model')
    parser.add_argument('--server_model', type=str)
    parser.add_argument('--client_model1', type=str)
    parser.add_argument('--client_model2', type=str)
    parser.add_argument('--emb_size', type=int, default=32, help='size of embedding(common is 32 while BOD is 64)')
    parser.add_argument('--layers', type=int, default=2, help='layer of model')
    parser.add_argument('--neg_count', type=int, default='100', help='count of negative samples in the test data set')
    parser.add_argument('--l2_emb', type=float, default='0.0', help='l2_emb')
    parser.add_argument('--num_heads', type=int, default=1, help='SAS')
    parser.add_argument('--num_blocks', type=int, default=2, help='SAS')

    args = parser.parse_args()
    # args.data_path = args.data_path + args.data_name + '/'
    # if args.seq > 0:
    #     args.train_path = args.data_path + args.data_name + args.data_type + 'SequenceTrain.csv'
    #     args.test_path = args.data_path + args.data_name + args.data_type + 'SequenceTest.csv'
    # else:
    #     args.train_path = args.data_path + args.data_name + args.data_type + 'GraphTrain.csv'
    #     args.test_path = args.data_path + args.data_name + args.data_type + 'GraphTest.csv'
    # args.lap_matrix = args.data_path + args.data_name + args.data_type + 'LapMatrix.npz'
    return args