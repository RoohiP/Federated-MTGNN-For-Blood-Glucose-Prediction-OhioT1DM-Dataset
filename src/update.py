import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np


class LocalUpdate(object):
    def __init__(self, args, clientloader, device, criterion):
        self.args = args
        self.trainloader = clientloader["train"]
        self.validloader = clientloader["val"]
        self.testloader = clientloader["test"]
        self.device = device
        # Default criterion set to NLL loss function
        self.criterion = criterion


    def update_weights(self, args, model, global_round, optimizer=None):
        # Set mode to train model
        model.train()
        train_epoch_mae_loss = []
        train_epoch_mse_loss = []
        val_epoch_mae_loss = []
        val_epoch_mse_loss = []
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
                train_mse_real += nn.functional.mse_loss(outputs*(1/args.scale), y_true*(1/args.scale)).item() * inputs.size(0)
                train_mae_real += nn.functional.l1_loss(outputs*(1/args.scale), y_true*(1/args.scale)).item() * inputs.size(0)
                train_running_num_samples += inputs.size(0)
                  
                
            model.eval()    
            val_mae_real = 0
            val_mse_real = 0
            val_num_samples = 0
            with torch.no_grad():
                for batch_idx, (inputs, y_true) in enumerate(self.validloader):
                    inputs, y_true = inputs.to(self.device), y_true.to(self.device)
                    if args.model == "MTGNN":
                        inputs = torch.unsqueeze(inputs,dim=1)
                        inputs = inputs.transpose(2,3)
                    outputs = model(inputs)
                    val_mse_real += nn.functional.mse_loss(outputs*(1/args.scale), y_true*(1/args.scale)).item() * inputs.size(0)
                    val_mae_real += nn.functional.l1_loss(outputs*(1/args.scale), y_true*(1/args.scale)).item() * inputs.size(0)
                    val_num_samples += inputs.size(0)
            print('| Global Round : {} | Local Epoch : {} | Train Loss: {:.6f} | Train MAE: {:.6f} | Val MAE: {:.6f} | Train MSE: {:.6f} | Val MSE: {:.6f} '.format(
                  global_round, iter, train_running_loss/train_running_num_samples, 
                  train_mae_real/train_running_num_samples, val_mae_real/val_num_samples,
                  train_mse_real/train_running_num_samples, val_mse_real/val_num_samples))
                
            train_epoch_mae_loss.append(train_mae_real/train_running_num_samples)
            train_epoch_mse_loss.append(train_mse_real/train_running_num_samples)
            val_epoch_mae_loss.append(val_mae_real/val_num_samples)
            val_epoch_mse_loss.append(val_mse_real/val_num_samples)

        return model.state_dict(), np.array(train_epoch_mae_loss).mean(), np.array(val_epoch_mae_loss).mean(), np.array(train_epoch_mse_loss).mean(), np.array(val_epoch_mse_loss).mean()
    
    
    
    
    def inference(self, args, model, dataset):
        """ Returns the inference accuracy and loss.
        """
        loader = None
        if dataset == "test":
            loader = self.testloader
        elif dataset == "val":
            loader = self.validloader
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
                running_mse += nn.functional.mse_loss(outputs*(1/args.scale), y_true*(1/args.scale)).item() * inputs.size(0)
                running_mae += nn.functional.l1_loss(outputs*(1/args.scale), y_true*(1/args.scale)).item() * inputs.size(0)
                running_num_samples += inputs.size(0)
        
        return running_mae, running_mse, running_num_samples



# def test_inference(args, model, test_loader, criterion, device):
#     """ Returns the test accuracy and loss.
#     """

#     model.eval()
#     loss= 0.0 
    
#     criterion = nn.NLLLoss().to(device)

#     for batch_idx, (images, labels) in enumerate(testloader):
#         images, labels = images.to(device), labels.to(device)

#         # Inference
#         outputs = model(images)
#         batch_loss = criterion(outputs, labels)
#         loss += batch_loss.item()

#         # Prediction
#         _, pred_labels = torch.max(outputs, 1)
#         pred_labels = pred_labels.view(-1)
#         correct += torch.sum(torch.eq(pred_labels, labels)).item()
#         total += len(labels)

#     accuracy = correct/total
#     return accuracy, loss
