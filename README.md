# Federated MTGNN For Blood Glucose Prediction OhioT1DM Dataset (PyTorch)

This repository is an implementation of Federated Learning for Blood Glucose prediction using the OhioT1DM dataset. The code contains two base models: a bidirectional LSTM model and a graph neural network for multivariate time series called MTGNN.

The papers and GitHub repositories referenced in this implementation are as follows:

- Bi-LSTM paper: [Predicting Blood Glucose with an LSTM and Bi-LSTM Based Deep Neural Network](https://arxiv.org/abs/1809.03817).
- MTGNN paper: [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/abs/2005.11650), [Github](https://github.com/nnzhan/MTGNN).
- Federated Learning GitHub Repository: [Github](https://github.com/AshwinRJ/Federated-Learning-PyTorch/tree/master).
- FastTensorDataLoader GitHub Repository: [Github](https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py).

## Requirments

Install all the packages

- Python3
- Pytorch
- Numpy
- Pandas
- Scipy

## Dataset

- You can get the Whole Dataset form http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html.
- The dataset that has been provided to me has this folder structure:

```
ohiot1dm
|-- Ohio2018-processed
    |-- train
        |-- 559-ws-training_processed.csv
        ...
    |-- test
        |-- 559-ws-testing_processed.csv
        ...
|-- Ohio2020-processed
    |-- train
        |-- 540-ws-training_processed.csv
        ...
    |-- test
        |-- 540-ws-testing_processed.csv
        ...
```

- Each CSV file follows this format:

  | 5minute_intervals_timestamp | missing_cbg | cbg | finger | basal | hr  | gsr | bolus |
  | --------------------------- | ----------- | --- | ------ | ----- | --- | --- | ----- |
  | 6024291                     | 0           | 142 | NaN    | NaN   | NaN | NaN | NaN   |
  | 6024292                     | 0           | 143 | NaN    | NaN   | NaN | NaN | NaN   |

## Running the Code

## Options

The default values for various paramters parsed to the experiment are given in `options.py`. Details are given some of those parameters:

### General Options

- `--grid_search:` Do a grid search validation, Default is True.
- `--validation:` Do validation split, set this to False to use test data.
- `--val_ration:` Fraction of train data to use as validation.
- `--num_clients:` Number of clients for Federated Learning, Default is 4.
- `--epochs:` Number of global epochs, Default is 10.
- `--local_ep:` Number of local epochs, Default is 5.
- `--ohio_directory:` Ohio data directory, Default is ../data/Ohio Data/.
- `--batch_size:` Batch size, Default is 256.
- `--lr:` Learning rate in Optimizer, Default is 0.01.
- `--weight_decay:` Weight Decay in Optimizer, Default is 0.01.
- `--model:` Model Type, Default is MTGNN, Options: 'MTGNN or 'BiLSTM'.
- `--scale:` Scaling the output values, Default is 0.01.
- `--seq_in_len:` Window size of input, Default is 25, meaning 2 hours of input data.
- `--horizon:` Prediction horizon \* 5 min interval , Default is 6, meaning 30 min in the future.
- `--imputation_method:` imputation method, Options: KNN, Linear, Cubic.
- `--imputation_KNN_K:` K parameter in KNN future, Default is 5.
- `--clamp_output:` Clamping the output of model to be between min and max value, Default is True.
- `--clamp_output_min:` Default is 40.
- `--clamp_output_max:` Default is 400.

### Bi-LSTM Options

- `--input_size:` Input dim of Bi-LSTM, Default is 7.
- `--hidden_size:` Hidden size of Bi-LSTM, Default is 7.
- `--num_layers:` Number of LSTM layers, Default is 1.
- `--lstm_dropout:` Dropout fraction of Linear layers after LSTM, Default is 0.2.

### MTGNN Options

- `--num_nodes:` Number of nodes in the graph, Default is 7.
- `--node_dim:` Node Embedding dimension, Default is 4.
- `--gcn-depth:` Graph convolution depth, Default is 2.
- `--mtgnn_dropout:` Dropout fraction, Default is 0.2.
- `--subgraph_size:` How many top nodes to use in the graph, Default is 7.
- `--dilation_exponential:` Dilation exponential, Default is 2.
- `--conv_channels:` Convolution channels, Default is 4.
- `--residual_channels:` Residual channels, Default is 4.
- `--skip_channels:` Skip channels, Default is 6.
- `--end_channels:` End channels, Default is 16.
- `--layers:` Number of layers, Default is 5.
- `--propalpha:` Prop alpha, Default is 0.05.
- `--tanhalpha:` Adjacency alpha, Default is 3.

## Results
