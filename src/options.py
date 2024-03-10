import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_search', type=bool, default=True, help="do a grid search validation")
    
    parser.add_argument('--num_clients', type=int, default=4, help="number of clients")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs")
    
    parser.add_argument('--weight_decay', type=float, default=0, help="weight_decay")
    parser.add_argument('--epochs', type=int, default=30, help="number of rounds of training")
    parser.add_argument('--batch_size',type=int,default=512,help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    parser.add_argument('--model', type=str, default='MTGNN', help='model name')
    parser.add_argument('--ohio_directory', type=str, default='../data/Ohio Data/', help='Ohio data directory')
    parser.add_argument('--validation', type=bool, default=False, help='do validation split, set this to False to use test data')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='fraction of train data to use as validation')
    parser.add_argument('--scale', type=float, default=0.01, help='scaling the y values')
    parser.add_argument('--imputation_method', type=str, default="KNN", help='imputation method, Options: KNN, Linear, Cubic')
    parser.add_argument('--imputation_KNN_K', type=int, default=5, help='K parameter in KNN')
    
    parser.add_argument('--seq_in_len',type=int,default=25, help='input sequence length')
    parser.add_argument('--horizon', type=int, default=6)
    parser.add_argument('--clamp_output', type=bool, default=True, help='Clamping the output of model to be between min and max value')
    parser.add_argument('--clamp_output_min', type=float, default=40.0, help='Clamping the output of model to be between min and max value')
    parser.add_argument('--clamp_output_max', type=float, default=400.0, help='Clamping the output of model to be between min and max value')
    
    ### BiLSTM
    parser.add_argument('--input_size', type=float, default=7, help='input dim of LSTM')
    parser.add_argument('--hidden_size', type=float, default=7, help='hidden size of LSTM')
    parser.add_argument('--num_layers', type=float, default=1, help='number of LSTM layers')
    parser.add_argument('--lstm_dropout', type=float, default=0.2, help='dropout fraction of LSTM layer')

    
    ###MTGNN
    parser.add_argument('--num_nodes',type=int,default=7, help='Number of Nodes')
    parser.add_argument('--node_dim',type=int,default=4, help='dim of nodes')
    parser.add_argument('--gcn-depth',type=int,default=2,help='depth of GCN')
    parser.add_argument('--mtgnn_dropout',type=int,default=0.2, help='dropout fraction in MTGNN')
    parser.add_argument('--subgraph_size',type=int,default=7,help='max number of edges each node can have in the graph')
    parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
    
    parser.add_argument('--conv_channels',type=int,default=4,help='convolution channels')
    parser.add_argument('--residual_channels',type=int,default=4,help='residual channels') 
    parser.add_argument('--skip_channels',type=int,default=6,help='skip channels')
    parser.add_argument('--end_channels',type=int,default=16,help='end channels') 
    
    parser.add_argument('--layers',type=int,default=2,help='number of layers')
    parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
    parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')
    
    

    args = parser.parse_args()
    return args
