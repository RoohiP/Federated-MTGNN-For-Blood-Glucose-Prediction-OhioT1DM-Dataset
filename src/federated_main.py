import os
import copy
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from options import args_parser
from update import LocalUpdate
from models import BiLSTM, gtnet
from utils import get_dataset, average_weights, exp_details, plotting

def grid_search(args):
    args.validation = True
    param_grid_MTGNN = {
        'learning_rate': [0.001, 0.0001],
        'optimizer': [optim.Adam], #optim.SGD, 
        'local_ep':[5, 10],
        'epochs': [30],
        'weight_decay': [0],
        'batch_size': [512],
        'mtgnn_dropout':[0.2],
        'channels_mul': [1],
        'layers': [2],
        'num_clients': [1, 4]
    }
    param_grid_BiLSTM = {
        'learning_rate': [0.001, 0.01],
        'optimizer': [optim.Adam],
        'local_ep':[5, 10],
        'epochs': [30],
        'weight_decay': [0],
        'batch_size': [512],
        'lstm_dropout': [0.2],
        'num_clients': [1, 4]
    }

    all_params = None
    if args.model == 'BiLSTM':
        all_params = ParameterGrid(param_grid_BiLSTM)
    elif args.model == 'MTGNN':
        all_params = ParameterGrid(param_grid_MTGNN)
    else:
        exit('Error: unrecognized model')
    results = []
    for params in all_params:
        result = []
        optimizer = params['optimizer']
        args.local_ep = params['local_ep']
        args.epochs = params['epochs']
        args.lr = params['learning_rate']
        args.weight_decay = params['weight_decay']
        args.batch_size = params['batch_size']
        if args.model == 'BiLSTM':
            args.lstm_dropout = params['lstm_dropout']
        elif args.model == 'MTGNN':
            args.mtgnn_dropout = params['mtgnn_dropout']
            args.conv_channels = params['channels_mul'] * args.conv_channels
            args.residual_channels = params['channels_mul'] * args.residual_channels
            args.skip_channels = params['channels_mul'] * args.skip_channels
            args.end_channels = params['channels_mul'] * args.end_channels
            args.layers = params['layers']
            args.num_clients = params['num_clients']
        
        print(args)
        for _, arg_value in vars(args).items():
            result.append(arg_value)
        client_loader, client_mapping = get_dataset(args)
        try:
            client_losses, global_losses = train(args, client_loader, optimizer)
        except Exception as e:
            print("An error occurred:", e)
            continue
            
        
        result.append(global_losses["train_mae"][-1])
        result.append(global_losses["train_mse"][-1])
        result.append(global_losses["test_mae"][-1])
        result.append(global_losses["test_mse"][-1])
        results.append(result)
        
    results_column = []
    for arg_name, _ in vars(args).items():
        results_column.append(arg_name) 
    results_column += ["MAE_Train_Global", "MAE_Test_Global", "MSE_Train_Global", "MSE_Test_Global"]
    results = pd.DataFrame(results, columns=results_column)
    results.to_csv("../save/Param_search_results.csv")
        
        
    
def train(args, client_loader, optimizer=None):
    exp_details(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client_losses = defaultdict(lambda: defaultdict(list))
    global_losses = defaultdict(list)
    if args.model == 'BiLSTM':
        global_model = BiLSTM(args, device)
    elif args.model == 'MTGNN':
        global_model = gtnet(True, True, args.gcn_depth, args.num_nodes, device, dropout=args.mtgnn_dropout, subgraph_size=args.subgraph_size,
              node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
              conv_channels=args.conv_channels, residual_channels=args.residual_channels, skip_channels=args.skip_channels, end_channels=args.end_channels,
              seq_length=args.seq_in_len, in_dim=1, out_dim=1, layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False)
    else:
        exit('Error: unrecognized model')
    
    global_model.to(device)
    criterion = nn.L1Loss()
    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    
    for epoch in tqdm(range(args.epochs)):
        local_weights = []
        train_mae_avg, train_mse_avg, test_mae_avg, test_mse_avg = 0, 0, 0, 0
        print(f'\n | Global Training Round : {epoch+1} |\n')
        global_model.train()
        for client_idx in range(args.num_clients):
            print(f'| Client Index : {client_idx} |')
            local_model = LocalUpdate(args, client_loader[client_idx], device, criterion)
            w, train_mae_loss, test_mae_loss, train_mse_loss, test_mse_loss = local_model.update_weights(args, copy.deepcopy(global_model), epoch+1, optimizer)
            local_weights.append(copy.deepcopy(w))
            client_losses[client_idx]["train_mae"].append(train_mae_loss)
            client_losses[client_idx]["train_mse"].append(train_mse_loss)
            client_losses[client_idx]["test_mae"].append(test_mae_loss)
            client_losses[client_idx]["test_mse"].append(test_mse_loss)
       
        
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        for client_idx in range(args.num_clients):
            train_mae_avg += client_losses[client_idx]["train_mae"][-1].mean() / args.num_clients
            train_mse_avg += client_losses[client_idx]["train_mse"][-1].mean() / args.num_clients
            test_mae_avg += client_losses[client_idx]["test_mae"][-1].mean() / args.num_clients
            test_mse_avg += client_losses[client_idx]["test_mse"][-1].mean() / args.num_clients
        
        
        train_global_mae, train_global_mse, test_global_mae, test_global_mse = 0, 0, 0, 0
        train_global_samples, test_global_samples = 0, 0
        global_model.eval()
        with torch.no_grad():
            for client_idx in range(args.num_clients):
                local_model = LocalUpdate(args, client_loader[client_idx], device, criterion)
                running_mae, running_mse, running_num_samples = local_model.inference(args, global_model, "train")
                train_global_mae += running_mae
                train_global_mse += running_mse
                train_global_samples += running_num_samples
                running_mae, running_mse, running_num_samples = local_model.inference(args, global_model, "test")
                test_global_mae += running_mae
                test_global_mse += running_mse
                test_global_samples += running_num_samples
        global_losses["train_mae"].append(train_global_mae/train_global_samples)
        global_losses["train_mse"].append(train_global_mse/train_global_samples)
        global_losses["test_mae"].append(test_global_mae/test_global_samples)
        global_losses["test_mse"].append(test_global_mse/test_global_samples)
    # print global training loss after every 'i' rounds
        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        val_test_str = "Val" if args.validation else "Test"
        print('Avg Loss on each client  model | Train MAE: {:.6f} | {} MAE: {:.6f} | Train MSE: {:.6f} | {} MSE: {:.6f}'.format(train_mae_avg, val_test_str, test_mae_avg, train_mse_avg, val_test_str, test_mse_avg))
        print('Loss for Global model          | Train MAE: {:.6f} | {} MAE: {:.6f} | Train MSE: {:.6f} | {} MSE: {:.6f}'.format(global_losses["train_mae"][-1], val_test_str, global_losses["test_mae"][-1], global_losses["train_mse"][-1], val_test_str, global_losses["test_mse"][-1]))
        
    return client_losses, global_losses

def main():
    start_time = time.time()

    args = args_parser()
    if args.grid_search:
        grid_search(args)
    else:
        client_loader, client_mapping = get_dataset(args)
        client_losses, global_losses = train(args, client_loader)
        plotting(args, client_losses, global_losses)
if __name__ == '__main__':
    main()
    