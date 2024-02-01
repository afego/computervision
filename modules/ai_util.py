import time
import torch

from os import path
from copy import deepcopy
from torch import utils
from torchvision import datasets, models, transforms

class PytorchDataset:
    '''
    directory: path to dataset directory with 'train', 'val' folders by default
    training_phases: folders to be used during training: both 'train' and 'val' or only 'train'
    data_transforms: dictionary with training phases as key and torchvision.transforms.Compose() as values
    '''
    def __init__(self, directory, data_transforms, batch_size = 64, training_phases = ['train','val']):
        self.directory = directory  # 'E:\Datasets\cor-splits\sgkf-8-1-1-4000'
        self.training_phases = training_phases      # ['train','val']
        self.data_transforms = data_transforms
        
        image_datasets = {x: datasets.ImageFolder(path.join(self.directory, x), self.data_transforms[x])
                        for x in self.training_phases}
        
        self.dataloaders = {x: utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
                        for x in self.training_phases}
           
        self.dataset_sizes = {x: len(image_datasets[x]) for x in self.training_phases}
    
    def change_dataloaders(self, data_transforms, batch_size):
        '''
        if at any point there is a need to change the dataloaders for the same dataset
        '''
        self.data_transforms = data_transforms
        
        image_datasets = {x: datasets.ImageFolder(path.join(self.directory, x), self.data_transforms[x])
                        for x in self.training_phases}
        
        self.dataloaders = {x: utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
                        for x in self.training_phases}
           
        self.dataset_sizes = {x: len(image_datasets[x]) for x in self.training_phases}
        
class PytorchTraining:
    '''
    Generic training class for any pytorch model
    Requires a PytorchDataset object already instantiated
    '''
    def __init__(self, device, pytorch_dataset, output_directory):
        self.device = device                        # 
        self.output_directory = output_directory    # 
        self.dataset = pytorch_dataset
       
    def train_pytorch_model(self, model, criterion, optimizer, scheduler, start_epoch=1, num_epochs=25, epoch_save_interval=2):

        best_model_wts = deepcopy(model.state_dict())
        best_acc = 0.0

        log_path = f"{self.output_directory}/log.txt"
        
        run_info = 'Dataset {}\nLearning Rate Epoch Schedule = {}\nLearning Rate Gamma = {}'.format(
            self.dataset.directory, scheduler.step_size, scheduler.gamma
            )
        
        if not path.exists(log_path): 
            log = open(log_path,'x')
            log.writelines('=' * 10+'\n')
            log.writelines('\n'+run_info+'\n')
            log.writelines('=' * 10+'\n')
            log.close() 
        
        since = time.time()
        
        for epoch in range(start_epoch, num_epochs+1):
            epoch_start = time.time()
            
            epoch_info = 'Epoch {}/{}'.format(epoch, num_epochs)
            
            with open(log_path,'a') as log:
                log.writelines('\n'+epoch_info+'\n')
                log.writelines('-' * 10+'\n')
            
            print(epoch_info)
            print('-' * 10)
            

            # Each epoch has a training and validation phase
            for phase in self.dataset.training_phases:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataset.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset.dataset_sizes[phase]

                loss_acc_info = '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)
                with open(log_path,'a') as log:
                    log.writelines(loss_acc_info+'\n')
                    
                print(loss_acc_info)
                
                
                if epoch % epoch_save_interval == 0:
                    torch.save(model.state_dict(), f'{self.output_directory}/epoch_{epoch}.pth')
                    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = deepcopy(model.state_dict())
            
            epoch_end = time.time() - epoch_start
            epoch_duration_info = 'Epoch duration: {:.0f} m {:.0f}s'.format(epoch_end//60, epoch_end%60)
            lr_info = 'Learning Rate = {}'.format(scheduler.get_last_lr()[0])
            with open(log_path,'a') as log:
                log.writelines(epoch_duration_info+'\n')
                log.writelines(lr_info+'\n')
                log.writelines('-' * 15+'\n')
            print(epoch_duration_info)
            print(lr_info)
            print()
        
        torch.save(model.state_dict(), f'{self.output_directory}/last.pth')
        
        time_elapsed = time.time() - since
        
        complete_text = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
        print(complete_text)
        best_acc_text = 'Best val Acc: {:4f}'.format(best_acc)
        print(best_acc_text)
        
        with open(log_path,'a') as log:
            log.writelines(complete_text+'\n')
            log.writelines(best_acc_text+'\n')
            
        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), f'{self.output_directory}/best.pth')
        
        return model