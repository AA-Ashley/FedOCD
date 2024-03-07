import argparse

def config():
    parser = argparse.ArgumentParser()
    # 数据参数
    parser.add_argument('--data_path', type=str, default='./Data/', help='path of the dataset')
    parser.add_argument('--data_name', type=str, default='JD', help='name of the dataset')
    parser.add_argument('--data_type', type=str, default='Buy', help='type of the dataset')
    parser.add_argument('--user_count', type=int, default=0, help='total count of the users')
    parser.add_argument('--item_count', type=int, default=0, help='total count of the items')
    parser.add_argument('--train_path', type=str, default='./Data/', help='path of the train dataset')
    parser.add_argument('--test_path', type=str, default='./Data/', help='path of the test dataset')
    parser.add_argument('--lap_matrix', type=str, default='./Data/', help='path of the laplacian matrix')
    parser.add_argument('--seq', type=int, default=5, help='length of sequence')
    # 基本参数
    parser.add_argument('--device', type=str, default='cuda:1', help='device')
    parser.add_argument('--batch_size', type=int, default='4096', help='number of total iterations in each epoch')
    parser.add_argument('--lr', type=float, default='0.001', help='learning rate')
    parser.add_argument('--epochs', type=int, default='50', help='number of rounds of training')
    parser.add_argument('--weight_decay', type=float, default='0.0', help='weight decay')
    parser.add_argument('--dropout', type=float, default='0.2', help='generate dropout mask')

    # 模型参数
    parser.add_argument('--model_name', type=str, default='MF', help='name of the model')
    parser.add_argument('--emb_size', type=int, default=32, help='size of embedding(common is 32 while BOD is 64)')
    parser.add_argument('--layers', type=int, default=2, help='layer of model')
    parser.add_argument('--neg_count', type=int, default='100', help='count of negative samples in the test data set')
    # TODO：不懂
    parser.add_argument('--l2_emb', type=float, default='0.0', help='l2_emb')
    ####################################################################################################################################
    # BOD参数
    parser.add_argument('--generator_lr', type=float, default='0.001', help='learning rate of generator')
    # generator和model的embedding size按理说应该保持一致
    parser.add_argument('--generator_emb_size', type=int, default='32', help='embedding size of generator')
    parser.add_argument('--generator_reg', type=float, default='0.001', help='reg of generator')
    parser.add_argument('--outer_loop', type=int, default='1', help='outer_loop')
    parser.add_argument('--inner_loop', type=int, default='1', help='inner_loop')
    #TODO:这些参数都可以调
    parser.add_argument('--weight_bpr', type=int, default='1', help='weight of inner bpr loss ')
    parser.add_argument('--weight_alignment', type=int, default='1', help='weight of inner alignment loss')
    parser.add_argument('--weight_uniformity', type=float, default='0.5', help='weight of inner uniformity loss')


    args = parser.parse_args()
    if args.data_name == 'JD':
        args.user_count = 22211
        args.item_count = 10274
    elif args.data_name == 'Tmall':
        args.user_count = 6877
        args.item_count = 237701
    elif args.data_name == 'RR':
        args.user_count = 38300
        args.item_count = 24200
    elif args.data_name == 'Ten':
        args.user_count = 34241
        args.item_count = 130638
    else:
        print("Wrong dateset name")
    args.data_path = args.data_path + args.data_name + '/'
    if args.seq > 0:
        args.train_path = args.data_path + args.data_name + args.data_type + 'SequenceTrain.csv'
        args.test_path = args.data_path + args.data_name + args.data_type + 'SequenceTest.csv'
    else:
        args.train_path = args.data_path + args.data_name + args.data_type + 'GraphTrain.csv'
        args.test_path = args.data_path + args.data_name + args.data_type + 'GraphTest.csv'
    args.lap_matrix = args.data_path + args.data_name + args.data_type + 'LapMatrix.npz'
    return args