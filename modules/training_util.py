import time
import torch
import os
import numpy as np

from copy import deepcopy
from torch import utils
from torchvision import datasets, models, transforms
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
    def __init__(self, patience, delta=0, min_epoch=10, restore_best_weights=True):
        self.patience = patience
        self.delta = delta
        self.epoch_counter = 0
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
            self.message = 'Loss didnt decrease from {:.4f}. Increasing patience counter'.format(self.min_val_loss)
            if self.epoch_counter >= self.patience:
                self.message = f'Early stopping after {self.patience} epochs'
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False
    
class PytorchTraining:
    '''
    Generic training class for any pytorch model
    Requires a PytorchDataset object already instantiated
    '''
    def __init__(self, device, pytorch_dataset:PytorchDataset, output_directory):
        self.device = device                        # 
        self.output_directory = output_directory    # 
        self.dataset = pytorch_dataset
           
    def train_pytorch_model(self, model, criterion, optimizer, scheduler, early_stopper, start_epoch=1, num_epochs=25, epoch_save_interval=2):

        best_model_wts = deepcopy(model.state_dict())
        best_acc = 0.0

        fully_trainable = True
        for param in model.parameters():
            if param.requires_grad == False:
                fully_trainable = False
        log_path = f"{self.output_directory}/log.txt"
        
        run_info = 'Model {} Fully Trained = {}\nDataset {}\nLearning Rate Epoch Schedule = {}\nLearning Rate Gamma = {}\nOptimizer = {}\nBatch Size = {}'.format(
            model._get_name(), fully_trainable, self.dataset.directory, scheduler.step_size, scheduler.gamma, type (optimizer).__name__, self.dataset.batch_size
            )
        
        if not os.path.exists(log_path): 
            log = open(log_path,'x')
            log.writelines('=' * 10+'\n')
            log.writelines('\n'+run_info+'\n')
            log.writelines('=' * 10+'\n')
            log.close() 
        
        print('\n'+run_info+'\n')
        
        # Overall accuracy per camera
        codelist = {x:[0,0] for x in self.dataset.codes['val']}
        # Accuracy per epoch
        code_epoch = []
        # Overall accuracy per class per camera
        code_class = {
            x:[
            [0,0],  # Class 0 correct classification, Class 0 total labels
            [0,0]   # Class 1 correct classification, Class 1 total labels
            ] for x in self.dataset.codes['val']}
        
        since = time.time()
        early_stop = False
        epoch = start_epoch
        
        ## https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial
        # Initialize the prediction and label lists(tensors)
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
    
        while epoch < num_epochs+1 and not early_stop:
        # for epoch in range(start_epoch, num_epochs+1):
        
            epoch_start = time.time()
            
            epoch_info = '\nEpoch {}/{}\n----------\n'.format(epoch, num_epochs)
            with open(log_path,'a') as log:
                log.writelines(epoch_info)
            print(epoch_info)
            
            temp_dict = {}
            
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
                            torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), clip_value=1) # gradient clipping
                            optimizer.step()
                            
                        else:
                            # Calculating overall accuracy by camera per epoch
                            for j, path in enumerate(paths):
                                code = os.path.basename(path).split(' ')[0]
                                correct_class = (preds[j] == labels[j]).sum().item()
                                
                                # Accuracy per camera per class                
                                for idx in self.dataset.dataloaders[phase].dataset.class_to_idx.values():  
                                    if preds[j].item() == idx:
                                        if (preds[j] == labels[j]).item():
                                            code_class[code][idx][0] += 1 
                                        code_class[code][idx][1] += 1
                                
                                codelist[code][0] += correct_class 
                                codelist[code][1] += 1 # Total images evaluated
                                temp_dict[code] = [codelist[code][0], codelist[code][1]] # maybe rewrite this line, so it doesnt do this for every file with each code
                            code_epoch.append(temp_dict)
                                
                            # Append batch prediction results
                            predlist=torch.cat([predlist, preds.view(-1).cpu()])
                            lbllist=torch.cat([lbllist, labels.view(-1).cpu()])
                                            
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset.dataset_sizes[phase]

                # Logging Loss
                loss_acc_info = '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)
                with open(log_path,'a') as log:
                    log.writelines(loss_acc_info+'\n')      
                print(loss_acc_info)
                
                if epoch % epoch_save_interval == 0:
                    torch.save(model.state_dict(), f'{self.output_directory}/epoch_{epoch}.pth')
                
                if phase == 'val' and early_stopper(model, epoch_loss, epoch):
                    early_stop = True
                
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = deepcopy(model.state_dict())
            
            epoch_end = time.time() - epoch_start
            
            # Logging Epoch info
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

            epoch += 1
            
        torch.save(model.state_dict(), f'{self.output_directory}/last.pth')
        
        time_elapsed = time.time() - since
        
        # Logging complete statistics info
        complete_text = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
        print(complete_text)
        best_acc_text = 'Best val Acc: {:4f}'.format(best_acc)
        print(best_acc_text)
        
        conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
        class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
        
        with open(log_path,'a') as log:
            log.writelines(complete_text+'\n')
            log.writelines(best_acc_text+'\n')
            
        print("\nConfusion Matrix")
        print(conf_mat)
        print("\nAccuracy by class")
        print(class_accuracy)
        
        statistics_path = f"{self.output_directory}/stats.txt"
        
        conf_mat = conf_mat.tolist()
        class_accuracy = class_accuracy.tolist()
        
        with open(statistics_path,'x') as stats:
            stats.writelines("Confusion matrix")
            for row in conf_mat:
                stats.writelines("\n{} {}".format(row[0],row[1]))
            stats.writelines('\n\nAccuracy per class')
            cls_acc = '\n'
            for i in class_accuracy:
                cls_acc += "{:4f} ".format(i)
            stats.writelines(cls_acc)
            
            stats.write(f"\n\nOverall accuracy by camera during validation")
            for key, value in codelist.items():
                stats.write(f"\n{key}: {(value[0]/value[1]):.2f}")
            
            stats.write(f"\n\nOverall accuracy by camera per class during validation")
            for key,value in code_class.items():
                stats.write(f"\n{key}: Class 0: {(value[0][0]/value[0][1]):.2f} Class 1: {(value[1][0]/value[1][1]):.2f}")
                
            stats.write(f"\n\nCamera accuracy per epoch")
            for i in range(len(code_epoch)):
                stats.write(f"\nEpoch {i+1}")
                for code in self.dataset.codes['val']:
                    values = code_epoch[i][code]
                    stats.write(f"\n{code}: {(values[0]/values[1]):.2f}, correctly classifying {values[0]} out of {values[1]} images")

        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), f'{self.output_directory}/best.pth')
        
        return model