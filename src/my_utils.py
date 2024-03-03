import copy
import torch
import os
from collections import defaultdict
from fast_dataloader import FastTensorDataLoader
import numpy as np 
import pandas as pd
from scipy.interpolate import CubicSpline

def get_dataset(args):
    ohio_directory = '../../Code/Ohio Data/' #hyper
    ohio_folders = []
    data_dict = defaultdict(dict)
    for folder in os.listdir(ohio_directory):
        if folder.startswith("Ohio"):
            ohio_folders.append(os.path.join(ohio_directory, folder))
    for folder in ohio_folders:
        train_dir = os.path.join(folder, "train")
        test_dir = os.path.join(folder, "test")
        for file in os.listdir(train_dir):
            data_dict[file.split("-")[0]]["train"] = pd.read_csv(os.path.join(train_dir, file), index_col=0)
        for file in os.listdir(test_dir):
            data_dict[file.split("-")[0]]["test"] = pd.read_csv(os.path.join(test_dir, file), index_col=0)
  
    ### Cubic Interpolation
    for patient in data_dict.keys():
        train_df = data_dict[patient]["train"]
        test_df = data_dict[patient]["test"]
        last_train_idx = int((1-args.val_ratio) * train_df.shape[0])
        assert pd.isna(train_df["missing_cbg"]).sum() == 0
        assert pd.isna(test_df["missing_cbg"]).sum() == 0
        train_impute_df, train_impute_mask = cubic_interpolate(train_df.iloc[:last_train_idx])
        val_impute_df, val_impute_mask = cubic_interpolate(train_df.iloc[last_train_idx:])
        test_impute_df, test_impute_mask = cubic_interpolate(test_df)

        data_dict[patient]["train_impute"] = train_impute_df
        data_dict[patient]["train_impute_mask"] = train_impute_mask
        
        data_dict[patient]["val_impute"] = val_impute_df
        data_dict[patient]["val_impute_mask"] = val_impute_mask
        
        data_dict[patient]["test_impute"] = test_impute_df
        data_dict[patient]["test_impute_mask"] = test_impute_mask  
        
  
    ### Creating Client data
    patient_to_client_mapping = {}
    for idx, patient in enumerate(data_dict.keys()):
        client_idx = int(idx/(len(data_dict)/args.num_clients))
        patient_to_client_mapping[patient] = client_idx


    ### Normalizing the data and add them to corresponding client
    client_data = defaultdict(lambda: defaultdict(list))
    for patient in data_dict.keys():
        
        train_df = data_dict[patient]["train_impute"]
        val_df = data_dict[patient]["val_impute"]
        test_df = data_dict[patient]["test_impute"]
        
        # min-max normalization
        min_values = train_df.min()
        max_values = train_df.max()

        X_normalized_train_df = train_df.apply(lambda x: (x - min_values[x.name]) / (max_values[x.name] - min_values[x.name])).values
        X_normalized_val_df = val_df.apply(lambda x: (x - min_values[x.name]) / (max_values[x.name] - min_values[x.name])).values
        X_normalized_test_df = test_df.apply(lambda x: (x - min_values[x.name]) / (max_values[x.name] - min_values[x.name])).values
        
        np.nan_to_num(X_normalized_train_df, copy=False, nan=0.0)
        np.nan_to_num(X_normalized_val_df, copy=False, nan=0.0)
        np.nan_to_num(X_normalized_test_df, copy=False, nan=0.0)
        ############### change to cbg latter
        y_train = train_df["cbg"].values * args.scale
        y_train_mask = data_dict[patient]["train_impute_mask"]["cbg"].values

        y_val = val_df["cbg"].values * args.scale 
        y_val_mask = data_dict[patient]["val_impute_mask"]["cbg"].values

        y_test = test_df["cbg"].values * args.scale 
        y_test_mask = data_dict[patient]["test_impute_mask"]["cbg"].values
        ############### change to cbg latter
        client_idx = patient_to_client_mapping[patient]

        for index in range(args.seq_in_len, X_normalized_train_df.shape[0]-args.horizon+1):
            if y_train_mask[index+args.horizon-1] == 0:
                client_data[client_idx]["X_train"].append(X_normalized_train_df[index-args.seq_in_len:index, 1:]) 
                client_data[client_idx]["y_train"].append(y_train[index+args.horizon-1])

        for index in range(args.seq_in_len, X_normalized_val_df.shape[0]-args.horizon+1):
            if y_val_mask[index+args.horizon-1] == 0:
                client_data[client_idx]["X_val"].append(X_normalized_val_df[index-args.seq_in_len:index, 1:]) 
                client_data[client_idx]["y_val"].append(y_val[index+args.horizon-1])
                
        for index in range(args.seq_in_len, X_normalized_test_df.shape[0]-args.horizon+1):        
            if y_test_mask[index+args.horizon-1] == 0:
                client_data[client_idx]["X_test"].append(X_normalized_test_df[index-args.seq_in_len:index, 1:]) 
                client_data[client_idx]["y_test"].append(y_test[index+args.horizon-1])
              

    ## create dataloaders
    client_loaders = defaultdict(dict)
    for client_idx in client_data.keys():
        client_loaders[client_idx]["train"] = FastTensorDataLoader(torch.tensor(np.array(client_data[client_idx]["X_train"])).float(),
                                                                   torch.tensor(np.array(client_data[client_idx]["y_train"])).float().unsqueeze(1),
                                                                   batch_size=args.batch_size, shuffle=False)
        client_loaders[client_idx]["val"] = FastTensorDataLoader(torch.tensor(np.array(client_data[client_idx]["X_val"])).float(),
                                                                 torch.tensor(np.array(client_data[client_idx]["y_val"])).float().unsqueeze(1),
                                                                 batch_size=args.batch_size, shuffle=False)
        client_loaders[client_idx]["test"] = FastTensorDataLoader(torch.tensor(np.array(client_data[client_idx]["X_test"])).float(),
                                                                  torch.tensor(np.array(client_data[client_idx]["y_test"])).float().unsqueeze(1),
                                                                  batch_size=args.batch_size, shuffle=False)
    
    print(patient_to_client_mapping)

    for client_idx in client_loaders.keys():
        print('| client index : {} | Train size : {} | Val size : {} | Test size : {} |'.format(client_idx, 
              len(client_loaders[client_idx]["train"]), len(client_loaders[client_idx]["val"]), len(client_loaders[client_idx]["test"])))
    return client_loaders, patient_to_client_mapping



def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')

    print(f'    Fraction of clients  : {args.num_clients}')
    print(f'    Local Batch size   : {args.batch_size}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def cubic_interpolate(df):
    df_np = df.values
    missing_mask = np.isnan(df_np) * 1
    df_impute_np = np.zeros(df_np.shape)
    x_range = np.arange(df_np.shape[0])
    for col_index in range(df_np.shape[1]):
        not_missing_indexes = np.where(missing_mask[:,col_index] == 0)[0]
        if len(not_missing_indexes) == 0:
            continue
        cs_impute = CubicSpline(not_missing_indexes, df_np[not_missing_indexes, col_index])
        df_impute_np[:,col_index] = cs_impute(x_range)
    return pd.DataFrame(df_impute_np, columns=df.columns, index=df.index), pd.DataFrame(missing_mask, columns=df.columns, index=df.index)


def create_clients(data, window_length, horizon, num_clients):
    X_list = []
    y_list = []
    for X, y, mask_y in data:
        for index in range(window_length, X.shape[0]-horizon+1):
            if mask_y[window_length+horizon-1, 0]:
                X_list.append(X[index-window_length:index])
                y_list.append(y[index+horizon-1, :])
    return torch.tensor(np.array(X_list)).float(), torch.tensor(np.array(y_list)).unsqueeze(-1).float()