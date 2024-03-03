import os
import copy
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from my_options import args_parser
from my_update import LocalUpdate#, test_inference
from my_models import BiLSTM, gtnet
from my_utils import get_dataset, average_weights, exp_details

def grid_search(args, client_loader):
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'optimizer': [optim.SGD, optim.Adam],
        'weight_decay': [0, 0.01, 0.1],
        'batch_size': [32, 512]
    }
    all_params = ParameterGrid(param_grid)
    results = []
    for params in all_params:
        result = []
        optimizer = params['optimizer']
        args.lr = params['learning_rate']
        args.weight_decay = params['weight_decay']
        args.batch_size = params['batch_size']
        result.append(params['optimizer'])
        result.append(params['learning_rate'])
        result.append(params['weight_decay'])
        result.append(params['batch_size'])
        
        train_res = train(args, client_loader, optimizer)
        for item in train_res:
            result.append(item)
        results.append(result)
        
        
    results = pd.DataFrame(results, columns=["optimizer" , "learning_rate", "weight_decay", "batch_size", "Avg_MAE_Train_locals", "Avg_MAE_Val_locals", "Avg_MSE_Train_locals", "Avg_MSE_Val_locals", "MAE_Train_Global", "MAE_Val_Global", "MSE_Train_Global", "MSE_Val_Global"])
    results.to_csv("../save/Param_search_results.csv")
        
        
    
def train(args, client_loader, optimizer=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        local_weights, train_local_mae_losses, train_local_mse_losses, val_local_mae_losses, val_local_mse_losses = [], [], [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        for client_idx in range(args.num_clients):
            print(f'| Client Index : {client_idx} |')
            local_model = LocalUpdate(args, client_loader[client_idx], device, criterion)
            w, train_mae_loss, val_mae_loss, train_mse_loss, val_mse_loss = local_model.update_weights(args, copy.deepcopy(global_model), epoch+1, optimizer)
            local_weights.append(copy.deepcopy(w))
            train_local_mae_losses.append(copy.deepcopy(train_mae_loss))
            train_local_mse_losses.append(copy.deepcopy(train_mse_loss))
            val_local_mae_losses.append(copy.deepcopy(val_mae_loss))
            val_local_mse_losses.append(copy.deepcopy(val_mse_loss))
       
        
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        
        train_mae_avg = sum(train_local_mae_losses) / len(train_local_mae_losses)
        train_mse_avg = sum(train_local_mse_losses) / len(train_local_mse_losses)
        val_mae_avg = sum(val_local_mae_losses) / len(val_local_mae_losses)
        val_mse_avg = sum(val_local_mse_losses) / len(val_local_mse_losses)
        
        
        train_global_mae, train_global_mse, val_global_mae, val_global_mse = 0, 0, 0, 0
        train_global_samples, val_global_samples = 0, 0
        global_model.eval()
        with torch.no_grad():
            for client_idx in range(args.num_clients):
                local_model = LocalUpdate(args, client_loader[client_idx], device, criterion)
                running_mae, running_mse, running_num_samples = local_model.inference(args, global_model, "train")
                train_global_mae += running_mae
                train_global_mse += running_mse
                train_global_samples += running_num_samples
                running_mae, running_mse, running_num_samples = local_model.inference(args, global_model, "val")
                val_global_mae += running_mae
                val_global_mse += running_mse
                val_global_samples += running_num_samples

    # print global training loss after every 'i' rounds
        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print('Avg Loss on each client  model | Train MAE: {:.6f} | Val MAE: {:.6f} | Train MSE: {:.6f} | Val MSE: {:.6f}'.format(train_mae_avg, val_mae_avg, train_mse_avg, val_mse_avg))
        print('Loss for Global model          | Train MAE: {:.6f} | Val MAE: {:.6f} | Train MSE: {:.6f} | Val MSE: {:.6f}'.format(train_global_mae/train_global_samples, val_global_mae/val_global_samples, train_global_mse/train_global_samples, val_global_mse/val_global_samples))
        
    return train_mae_avg, val_mae_avg, train_mse_avg, val_mse_avg,train_global_mae/train_global_samples, val_global_mae/val_global_samples, train_global_mse/train_global_samples, val_global_mse/val_global_samples

def main():
    start_time = time.time()

    args = args_parser()
    
    exp_details(args)
    client_loader, client_mapping = get_dataset(args)

    
    if args.grid_search:
        grid_search(args, client_loader)
    else:
        train(args, client_loader)
    

# 
#             
            
            
            
    # Test inference after completion of training
#     test_acc, test_loss = test_inference(args, global_model, test_dataset)

#     print(f' \n Results after {args.epochs} global rounds of training:')
#     print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
#     print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

#     # Saving the objects train_loss and train_accuracy:
#     file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
#         format(args.dataset, args.model, args.epochs, args.frac, args.iid,
#                args.local_ep, args.local_bs)

#     with open(file_name, 'wb') as f:
#         pickle.dump([train_loss, train_accuracy], f)

#     print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))





if __name__ == '__main__':
    main()
    