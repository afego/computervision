import os

from torch import utils
from torchvision import datasets

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
        self.batch_size = batch_size
        
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.directory, x), self.data_transforms[x])
                        for x in self.training_phases}
        
        self.dataloaders = {x: utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
                        for x in self.training_phases}
           
        self.dataset_sizes = {x: len(image_datasets[x]) for x in self.training_phases}
    
    def __len__(self):
        return sum(self.dataset_sizes.values())
    
    def get_n_classes(self):
        path = os.path.join(self.directory, self.training_phases[0])
        return sum(os.path.isdir(os.path.join(path,i)) for i in os.listdir(path))
    