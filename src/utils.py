import copy
import torch
import os
from collections import defaultdict
from fast_dataloader import FastTensorDataLoader
import numpy as np 
import pandas as pd
from scipy import interpolate
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

def get_dataset(args):
    ohio_directory = args.ohio_directory
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
  
    ### Imputation
    for patient in data_dict.keys():
        train_df, test_df = None, None
        if args.validation:
            last_train_idx = int((1-args.val_ratio) * data_dict[patient]["train"].shape[0])
            train_df = data_dict[patient]["train"].iloc[:last_train_idx]
            test_df = data_dict[patient]["train"].iloc[last_train_idx:]
        else:
            train_df = data_dict[patient]["train"]
            test_df = data_dict[patient]["test"]

        assert pd.isna(train_df["missing_cbg"]).sum() == 0
        assert pd.isna(test_df["missing_cbg"]).sum() == 0
        
        train_impute_df, train_impute_mask, test_impute_df, test_impute_mask = None, None, None, None
        if args.imputation_method == 'KNN':
            train_impute_df, train_impute_mask = KNN_interpolate(args, patient, train_df, "train", None)
            test_impute_df, test_impute_mask = KNN_interpolate(args, patient, test_df, "test", train_impute_df)
        elif args.imputation_method == 'Linear':
            train_impute_df, train_impute_mask = linear_interpolate(train_df)
            test_impute_df, test_impute_mask = linear_interpolate(test_df)
        elif args.imputation_method == 'Cubic':
            train_impute_df, train_impute_mask = cubic_interpolate(train_df)
            test_impute_df, test_impute_mask = cubic_interpolate(test_df)
        else:
            exit('Error: unrecognized imputation method')
        
        data_dict[patient]["train_impute"] = train_impute_df
        data_dict[patient]["train_impute_mask"] = train_impute_mask
        
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
        test_df = data_dict[patient]["test_impute"]
        
        # min-max normalization
        min_values = train_df.min()
        max_values = train_df.max()
        X_normalized_train_df = train_df.copy()
        X_normalized_test_df = test_df.copy()
        
        X_normalized_train_df.iloc[:, 2:] = train_df.iloc[:, 2:].apply(lambda x: (x - min_values[x.name]) / (max_values[x.name] - min_values[x.name]))
        X_normalized_test_df.iloc[:, 2:] = test_df.iloc[:, 2:].apply(lambda x: (x - min_values[x.name]) / (max_values[x.name] - min_values[x.name]))
        
        X_normalized_train_df.iloc[:, 1] = X_normalized_train_df.iloc[:, 1] * args.scale
        X_normalized_test_df.iloc[:, 1] = X_normalized_test_df.iloc[:, 1] * args.scale
        
        X_normalized_train_df = X_normalized_train_df.values
        X_normalized_test_df = X_normalized_test_df.values
        
        np.nan_to_num(X_normalized_train_df, copy=False, nan=0.0)
        np.nan_to_num(X_normalized_test_df, copy=False, nan=0.0)
        
        y_train = train_df["cbg"].values * args.scale
        y_train_mask = data_dict[patient]["train_impute_mask"]["cbg"].values

        y_test = test_df["cbg"].values * args.scale 
        y_test_mask = data_dict[patient]["test_impute_mask"]["cbg"].values
        
        client_idx = patient_to_client_mapping[patient]

        for index in range(args.seq_in_len, X_normalized_train_df.shape[0]-args.horizon+1):
            if y_train_mask[index+args.horizon-1] == 0:
                client_data[client_idx]["X_train"].append(X_normalized_train_df[index-args.seq_in_len:index, 1:]) 
                client_data[client_idx]["y_train"].append(y_train[index+args.horizon-1])
                
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
        client_loaders[client_idx]["test"] = FastTensorDataLoader(torch.tensor(np.array(client_data[client_idx]["X_test"])).float(),
                                                                  torch.tensor(np.array(client_data[client_idx]["y_test"])).float().unsqueeze(1),
                                                                  batch_size=args.batch_size, shuffle=False)
    
    print(patient_to_client_mapping)
    
    for client_idx in client_loaders.keys():
        val_test_str = "Val" if args.validation else "Test"
        print('| client index : {} | Train size : {} | {} size : {}'.format(client_idx, 
              len(client_loaders[client_idx]["train"]), val_test_str, len(client_loaders[client_idx]["test"])))
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
        cs_impute = interpolate.CubicSpline(not_missing_indexes, df_np[not_missing_indexes, col_index])
        df_impute_np[:,col_index] = cs_impute(x_range)
    return pd.DataFrame(df_impute_np, columns=df.columns, index=df.index), pd.DataFrame(missing_mask, columns=df.columns, index=df.index)


def linear_interpolate(df):
    df_np = df.values
    missing_mask = np.isnan(df_np) * 1
    df_impute_np = np.zeros(df_np.shape)
    x_range = np.arange(df_np.shape[0])
    for col_index in range(df_np.shape[1]):
        not_missing_indexes = np.where(missing_mask[:,col_index] == 0)[0]
        if len(not_missing_indexes) == 0:
            continue
        for i in range(df.shape[0]):
            if np.isnan(df_np[i, col_index] ):
                df_np[i, col_index] = 0
            else:
                break
        for j in range(df.shape[0]):
            if np.isnan(df_np[df.shape[0] - j - 1, col_index]):
                df_np[df.shape[0] - j - 1, col_index] = 0
            else:
                break
        cs_impute = interpolate.interp1d(not_missing_indexes, df_np[not_missing_indexes, col_index])
        df_impute_np[:,col_index] = cs_impute(x_range)
    return pd.DataFrame(df_impute_np, columns=df.columns, index=df.index), pd.DataFrame(missing_mask, columns=df.columns, index=df.index)


def KNN_interpolate(args, patient, df, which, train_df=None):
    folder_name = "tmp"
    if args.validation:
        folder_name += "_val"
    else:
        folder_name += "_test"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    name = folder_name + "/" + patient + "_" + which + "_" + str(df.shape[0]) +".csv"
    imputed_df = None
    try:
        return pd.read_csv(name, index_col=0), pd.isna(df) * 1
    except OSError as e:
        k = args.imputation_KNN_K
        if train_df is None:
            imputer = KNNImputer(n_neighbors=k, keep_empty_features=True)
            imputed_df = imputer.fit_transform(df)
            imputed_df = pd.DataFrame(imputed_df, columns=df.columns, index=df.index)
            
        else:
            imputer = KNNImputer(n_neighbors=k, keep_empty_features=True)
            imputer.fit(train_df)
            imputed_df = imputer.transform(df)
            imputed_df = pd.DataFrame(imputed_df, columns=df.columns, index=df.index)

        imputed_df.to_csv(name)
        return imputed_df, pd.isna(df) * 1
    
    
def plotting(args, client_losses, global_losses):
    x = np.arange(1, args.local_ep*args.epochs+1)

    for clinet_idx in client_losses.keys():
        plt.figure(figsize=(14, 8))

        plt.subplot(2, 1, 1) 
        plt.plot(x, np.concatenate(client_losses[clinet_idx]["train_mae"]), label='Train MAE')
        plt.plot(x, np.concatenate(client_losses[clinet_idx]["test_mae"]), label='Test MAE')
        for i in range(args.epochs):
            plt.axvline((i+1)*args.local_ep, color='r', linestyle='--', label="Global Epochs" if i == 0 else None, alpha=0.3)
        plt.title('Mean Absolute Error')
        plt.legend()

        plt.subplot(2, 1, 2)  
        plt.plot(x, np.concatenate(client_losses[clinet_idx]["train_mse"]), label='Train MSE')
        plt.plot(x, np.concatenate(client_losses[clinet_idx]["test_mse"]), label='Test MSE')
        for i in range(args.epochs):
            plt.axvline((i+1)*args.local_ep, color='r', linestyle='--', label="Global Epochs" if i == 0 else None, alpha=0.3)
        plt.title('Mean Square Error')
        plt.legend()

        plt.tight_layout()
        plt.savefig('../save/'+ args.model +'_Client_' + str(clinet_idx+1) + '.png')
        plt.show()


    x = np.arange(args.epochs)
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)  
    plt.plot(x, global_losses["train_mae"], label='Global Train MAE')
    plt.plot(x, global_losses["test_mae"], label='Global Test MAE')
    plt.title('Mean Absolute Error')
    plt.legend()

    plt.subplot(2, 1, 2) 
    plt.plot(x, global_losses["train_mse"], label='Global Train MSE')
    plt.plot(x, global_losses["test_mse"], label='Global Test MSE')
    plt.title('Mean Square Error')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../save/'+ args.model +'_Global.png')
    plt.show()
    