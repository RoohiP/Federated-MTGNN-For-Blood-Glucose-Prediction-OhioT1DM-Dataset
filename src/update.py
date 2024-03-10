import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np


class LocalUpdate(object):
    def __init__(self, args, clientloader, device, criterion):
        self.args = args
        self.trainloader = clientloader["train"]
        self.testloader = clientloader["test"]
        self.device = device
        self.criterion = criterion
        


    def update_weights(self, args, model, global_round, optimizer=None):
        model.train()
        train_epoch_mae_loss = []
        train_epoch_mse_loss = []
        test_epoch_mae_loss = []
        test_epoch_mse_loss = []
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            optimizer = optimizer(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            
        for iter in range(self.args.local_ep):
            model.train()
            
            train_running_loss = 0
            train_running_num_samples = 0
            train_mae_real = 0
            train_mse_real = 0
            for batch_idx, (inputs, y_true) in enumerate(self.trainloader):
                inputs, y_true = inputs.to(self.device), y_true.to(self.device)
                if args.model == "MTGNN":
                    inputs = torch.unsqueeze(inputs,dim=1)
                    inputs = inputs.transpose(2,3)
                model.zero_grad()
                outputs = model(inputs)
                
                loss = self.criterion(outputs, y_true)
                loss.backward()
                optimizer.step()
                train_running_loss += loss.item() * inputs.size(0)
                if args.clamp_output:
                    outputs = torch.clamp(outputs, min=args.clamp_output_min*args.scale, max=args.clamp_output_max*args.scale)
                train_mse_real += nn.functional.mse_loss(outputs*(1/args.scale), y_true*(1/args.scale)).item() * inputs.size(0)
                train_mae_real += nn.functional.l1_loss(outputs*(1/args.scale), y_true*(1/args.scale)).item() * inputs.size(0)
                train_running_num_samples += inputs.size(0)
                  
                
            model.eval()    
            test_mae_real = 0
            test_mse_real = 0
            test_num_samples = 0
            with torch.no_grad():
                for batch_idx, (inputs, y_true) in enumerate(self.testloader):
                    inputs, y_true = inputs.to(self.device), y_true.to(self.device)
                    if args.model == "MTGNN":
                        inputs = torch.unsqueeze(inputs,dim=1)
                        inputs = inputs.transpose(2,3)
                    outputs = model(inputs)
                    if args.clamp_output:
                        outputs = torch.clamp(outputs, min=args.clamp_output_min*args.scale, max=args.clamp_output_max*args.scale)
                    test_mse_real += nn.functional.mse_loss(outputs*(1/args.scale), y_true*(1/args.scale)).item() * inputs.size(0)
                    test_mae_real += nn.functional.l1_loss(outputs*(1/args.scale), y_true*(1/args.scale)).item() * inputs.size(0)
                    test_num_samples += inputs.size(0)
            val_test_str = "Val" if args.validation else "Test"
            print('| Global Round : {} | Local Epoch : {} | Train Loss: {:.6f} | Train MAE: {:.6f} | {} MAE: {:.6f} | Train MSE: {:.6f} | {} MSE: {:.6f} '.format(
                  global_round, iter, train_running_loss/train_running_num_samples, 
                  train_mae_real/train_running_num_samples, val_test_str, test_mae_real/test_num_samples,
                  train_mse_real/train_running_num_samples, val_test_str, test_mse_real/test_num_samples))
                
            train_epoch_mae_loss.append(train_mae_real/train_running_num_samples)
            train_epoch_mse_loss.append(train_mse_real/train_running_num_samples)
            test_epoch_mae_loss.append(test_mae_real/test_num_samples)
            test_epoch_mse_loss.append(test_mse_real/test_num_samples)
            
        return model.state_dict(), np.array(train_epoch_mae_loss), np.array(test_epoch_mae_loss), np.array(train_epoch_mse_loss), np.array(test_epoch_mse_loss)
    
    
    
    
    def inference(self, args, model, dataset):
        loader = None
        if dataset == "test":
            loader = self.testloader
        else:
            loader = self.trainloader
            
        model.eval()
        running_mae = 0
        running_mse = 0
        running_num_samples = 0
        with torch.no_grad():
            for batch_idx, (inputs, y_true) in enumerate(loader):
                inputs, y_true = inputs.to(self.device), y_true.to(self.device)
                if args.model == "MTGNN":
                    inputs = torch.unsqueeze(inputs,dim=1)
                    inputs = inputs.transpose(2,3)
                outputs = model(inputs)
                if args.clamp_output:
                    outputs = torch.clamp(outputs, min=args.clamp_output_min*args.scale, max=args.clamp_output_max*args.scale)
                running_mse += nn.functional.mse_loss(outputs*(1/args.scale), y_true*(1/args.scale)).item() * inputs.size(0)
                running_mae += nn.functional.l1_loss(outputs*(1/args.scale), y_true*(1/args.scale)).item() * inputs.size(0)
                running_num_samples += inputs.size(0)
        
        return running_mae, running_mse, running_num_samples

