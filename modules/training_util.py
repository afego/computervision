import time
import torch
import os
import numpy as np
import pandas as pd

from copy import deepcopy
from torch import utils
from torchvision import datasets
from torch.nn import BCELoss
from sklearn.metrics import confusion_matrix
from modules.dataset_util import PytorchDataset

def makeDataloader(dataset_dir, data_transforms, batch_size=64):
    dataset_splits = ['train','test','val']
    dataset = {}
    dataloader = {}
    for split in dataset_splits:
        curr_dir = os.path.join(dataset_dir, split)
        if os.path.exists(curr_dir):
            dataset[split] = datasets.ImageFolder(curr_dir, data_transforms[split])
            dataloader[split] = utils.data.DataLoader(dataset[split], batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader 

class EarlyStopping:
    # https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    # https://www.youtube.com/watch?v=lS0vvIWiahU
    def __init__(self, patience, delta=0, min_epoch=10, starting_patience_counter=0,restore_best_weights=True):
        self.patience = patience
        self.delta = delta
        self.epoch_counter = starting_patience_counter
        self.min_val_loss = float('inf')
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.message = ''
        self.min_epoch = min_epoch  # Number of epochs before initializing patience counter
        
    def __call__(self, model, val_loss, epoch):
        if epoch <= self.min_epoch:
            return False
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.best_model = deepcopy(model.state_dict())
            self.epoch_counter = 0
            self.message = f'Lower loss found, resetting patience counter'
        elif val_loss > (self.min_val_loss + self.delta):
            self.epoch_counter += 1
            self.message = 'Loss didnt decrease from {:.4f}. Increasing patience counter to {}'.format(self.min_val_loss, self.epoch_counter)
            if self.epoch_counter >= self.patience:
                self.message = f'Early stopping after {self.patience} epochs'
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                    self.message += f'\nLower loss model restored'
                return True
        return False
    
class PytorchTraining:
    '''
    Generic training class for any pytorch classification model
    Requires a PytorchDataset object already instantiated
    '''
    def __init__(self, device, pytorch_dataset:PytorchDataset, output_directory):
        self.device = device                        # 
        self.output_directory = output_directory    # 
        self.dataset = pytorch_dataset
           
    def train_pytorch_model(self, model, criterion, optimizer, arch, scheduler=None, early_stopper=None, start_epoch=1, num_epochs:int=25, epoch_save_interval:int=2, save_weights=True):
        
        if not epoch_save_interval:
            epoch_save_interval = 2
            
        best_model_wts = deepcopy(model.state_dict())
        best_acc = 0.0
        lower_loss = 0.0
        
        fully_trainable = True
        for param in model.parameters():
            if param.requires_grad == False:
                fully_trainable = False
        
        #=====================
        # Initializing Log
        #=====================
        
        log_path = f"{self.output_directory}/log.txt"
        stats_dir = f"{self.output_directory}/stats"
        
        run_info =  'Model {} {}\nFully Trained = {}\nDataset {}\nLearning Rate Epoch Schedule = {}\
                    \nLearning Rate Gamma = {}\nOptimizer = {}\nBatch Size = {}\nTransforms = {}\n'.format(
                    model._get_name(), arch.arch, fully_trainable, self.dataset.directory, scheduler.step_size, 
                    scheduler.gamma, type (optimizer).__name__, self.dataset.batch_size, arch.data_transforms['train']
            )
        
        if not os.path.exists(log_path): 
            log = open(log_path,'x')
            log.writelines('=' * 10+'\n')
            log.writelines('\n'+run_info+'\n')
            log.writelines('=' * 10+'\n')
            log.close() 
        
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)
            
        conf_mat_file = os.path.join(stats_dir,'conf_mat.csv')  
        if not os.path.exists(conf_mat_file):
            log = open(conf_mat_file,'x')
            log.writelines(f"epoch,TP,FN,FP,TN\n")
            log.close()
        
        camera_acc_file = os.path.join(stats_dir,'cam_acc.csv')
        if not os.path.exists(camera_acc_file):
            log = open(camera_acc_file,'x')
            log.writelines(f"epoch,code,cls_0_corr,cls_0_total,cls_1_corr,cls_1_total\n")
            log.close()
            
        acc_loss_file = os.path.join(stats_dir,'acc_loss.csv')
        if not os.path.exists(acc_loss_file):
            log = open(acc_loss_file,'x')
            header = "epoch"
            for phase in self.dataset.training_phases:
                header += f",{phase}_acc,{phase}_loss"
            log.writelines(header+"\n")
            log.close()
        
        daynight_acc_file = os.path.join(stats_dir,'day_night_acc.csv')
        if not os.path.exists(daynight_acc_file):
            log = open(daynight_acc_file, 'x')
            log.writelines(f"epoch,time,cls_0_corr,cls_0_total,cls_1_corr,cls_1_total\n")
            log.close()
        
        print(f'Saving model to folder {self.output_directory}')
        print('\n'+run_info+'\n')
        
        #=====================
        # Starting training
        #=====================
        
        since = time.time()
        early_stop = False
        epoch = start_epoch
    
        while epoch < num_epochs+1 and not early_stop:
        # for epoch in range(start_epoch, num_epochs+1):

            ## https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial
            # Initialize the prediction and label lists(tensors)
            predlist=torch.zeros(0,dtype=torch.long, device='cpu')
            lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
            #=====================
            # Epoch Metrics
            #=====================
            cam_acc = {
                x:[
                    [0,0],  # Class 0 correct classification, Class 0 total labels
                    [0,0]   # Class 1 correct classification, Class 1 total labels
                    ] for x in self.dataset.codes['val']
                }
            
            acc_loss = {
                x:[0,0] for x in self.dataset.training_phases
            }
            
            daynight_acc = {
                x:[
                    [0,0],
                    [0,0]
                ] for x in ['day','night']
            }
            
            epoch_info = '\nEpoch {}/{}\n----------\n'.format(epoch, num_epochs)
            
            with open(log_path,'a') as log:
                log.writelines(epoch_info)
            print(epoch_info)
            
            #=====================
            # Starting Epoch
            #=====================
            
            epoch_start = time.time()
            
            # Each epoch has a training and validation phase
            for phase in self.dataset.training_phases:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels, paths in self.dataset.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                            
                        if phase == 'train':  
                            loss.backward()
                            torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), clip_value=1)
                            optimizer.step()
                            
                        else:
                            # Calculating overall accuracy by camera and time of day per epoch
                            for j, path in enumerate(paths):
                                code = os.path.basename(path).split(' ')[0]
                                
                                timeofday = os.path.basename(path).split(' ')[2]
                                timeofday = os.path.splitext(timeofday)[0]
                                timeofday = int(timeofday.split('-')[0])
                                
                                if timeofday >= 5 and timeofday <= 18:
                                    daynight = 'day'
                                else:
                                    daynight = 'night'
                                
                                # Accuracy per class                
                                for idx in self.dataset.dataloaders[phase].dataset.class_to_idx.values():  
                                    #if preds[j].item() == idx:
                                    if (preds[j] == labels[j]).item():
                                        # cam_acc[code][idx][0] += 1
                                        cam_acc[code][preds[j].item()][0] += 1 # increasing counter for correct classification in 'preds[j].item()' class
                                        daynight_acc[daynight][preds[j].item()][0] += 1
                                    
                                    # cam_acc[code][idx][1] += 1
                                    cam_acc[code][preds[j].item()][1] += 1 # increasing counter for total 'preds[j].item()' class objects evaluated
                                    daynight_acc[daynight][preds[j].item()][1] += 1   
                                
                            # Append batch prediction results
                            predlist=torch.cat([predlist,preds.view(-1).cpu()])
                            lbllist=torch.cat([lbllist,labels.view(-1).cpu()])
                                            
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train' and scheduler is not None:
                    scheduler.step()

                epoch_loss = running_loss / self.dataset.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset.dataset_sizes[phase]

                acc_loss[phase][0] = epoch_acc
                acc_loss[phase][1] = epoch_loss
                
                # Logging Loss
                loss_acc_info = '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)
                with open(log_path,'a') as log:
                    log.writelines(loss_acc_info+'\n')      
                print(loss_acc_info)
                
                #=====================
                # Saving model
                #=====================
                
                if save_weights:
                    if epoch % epoch_save_interval == 0 and epoch > 10:
                        epoch_weight_path = f'{self.output_directory}/epoch_{epoch}.pth'
                        torch.save(model.state_dict(), epoch_weight_path)
                        
                        text = f"Model saved at {epoch_weight_path}"
                        with open(log_path,'a') as log:
                            log.writelines(text+'\n')      
                        print(text)
                
                if phase == 'val' and early_stopper(model, epoch_loss, epoch):
                    early_stop = True
                
                # if phase == 'val' and epoch_acc > best_acc and epoch >= early_stopper.min_epoch:
                if phase == 'val' and epoch_loss < lower_loss and epoch >= early_stopper.min_epoch:
                    # best_acc = epoch_acc
                    lower_loss = epoch_loss
                    best_model_wts = deepcopy(model.state_dict())
            
            #=====================
            # Logging Epoch info
            #=====================
            
            epoch_end = time.time() - epoch_start
            
            # cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
            cm = confusion_matrix(lbllist.numpy(), predlist.numpy())
                 
            tp = cm[0][0]
            fn = cm[0][1]
            fp = cm[1][0]
            tn = cm[1][1]
                
            epoch_duration_info = 'Epoch duration: {:.0f} m {:.0f}s'.format(epoch_end//60, epoch_end%60)
            lr_info = 'Learning Rate = {}'.format(scheduler.get_last_lr()[0])
            with open(log_path,'a') as log:
                log.writelines(epoch_duration_info+'\n')
                log.writelines(lr_info+'\n')
                log.writelines(early_stopper.message+'\n')
                log.writelines('-' * 15+'\n')
            print(epoch_duration_info)
            print(lr_info)
            print(early_stopper.message)
            print()
            
            with open(conf_mat_file, 'a') as fl:
                fl.writelines(f'{epoch},{tp},{fn},{fp},{tn}\n')
            
            with open(camera_acc_file, 'a') as fl:
                for key,value in cam_acc.items():
                    fl.writelines(f"{epoch},{key},{value[0][0]},{value[0][1]},{value[1][0]},{value[1][1]}\n")
            
            with open(daynight_acc_file, 'a') as fl:
                for key,value in daynight_acc.items():
                    fl.writelines(f"{epoch},{key},{value[0][0]},{value[0][1]},{value[1][0]},{value[1][1]}\n")
                    
            with open(acc_loss_file, 'a') as fl:
                line = f"{epoch}"
                for phase in self.dataset.training_phases:
                    line += f",{acc_loss[phase][0]},{acc_loss[phase][1]}"
                fl.writelines(line+"\n")
            
            epoch += 1
        
        # Training finished
        time_elapsed = time.time() - since
        if save_weights:
            torch.save(model.state_dict(), f'{self.output_directory}/last.pth') # With early stopping, this should be the last weight before the counter started
            model.load_state_dict(best_model_wts)
            torch.save(model.state_dict(), f'{self.output_directory}/best.pth')
        
        #=====================
        # Logging complete 
        # statistics info
        #=====================
        
        complete_text = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
        print(complete_text)
        # best_acc_text = 'Best val Acc: {:4f}'.format(best_acc)
        lower_loss_text = 'Lower val Loss: {:4f}'.format(lower_loss)
        print(lower_loss_text)
        
        conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
        class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
        
        with open(log_path,'a') as log:
            log.writelines(complete_text+'\n')
            log.writelines(lower_loss_text+'\n')
            
        print("\nConfusion Matrix")
        print(conf_mat)
        print("\nAccuracy by class")
        print(class_accuracy)
        
        return model