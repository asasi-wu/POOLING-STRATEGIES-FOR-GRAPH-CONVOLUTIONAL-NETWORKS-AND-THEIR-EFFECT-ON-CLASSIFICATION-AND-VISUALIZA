import argparse
def get_config():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777,
                    help='seed')
    parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                    help='weight decay')
    parser.add_argument('--hidden_size', type=int, default=128,
                    help='hidden size')
    parser.add_argument('--pooling_ratio', type=float, default=0.8,
                    help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                    help='dropout ratio')
    parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1')
    parser.add_argument('--epochs', type=int, default=50,
                    help='maximum number of epochs')

    parser.add_argument('--pooling_layer_type', type=str, default='GCNConv')
    parser.add_argument('--device', type=str, default='cuda:0',
                    help='cuda:0/cpu')
    
    parser.add_argument('--pool_method', type=str, default='DiffPool',
                    help='SAGPool/gPool/edgepool/DiffPool')
    parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
    parser.add_argument('--add_node', type=bool, default=False,
                    help='add_node feature or not')
    parser.add_argument('--two_conv', type=bool, default=False,
                    help='two convolution layers or not')

    return parser.parse_args(args=[])
