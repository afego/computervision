import os

from typing import Tuple, Any
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

class FloodDataset(datasets.ImageFolder):    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, path) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target, path
    
class PytorchDataset:
    '''
    directory: path to dataset directory with 'train', 'val' folders by default
    training_phases: folders to be used during training: both 'train' and 'val' or only 'train'
    data_transforms: dictionary with training phases as key and torchvision.transforms.Compose() as values
    '''
    def __init__(self, directory, data_transforms, batch_size = 64, training_phases = ['train','val'], num_workers=2):
        self.directory = directory  # 'E:\Datasets\cor-splits\sgkf-8-1-1-4000'
        self.training_phases = training_phases
        self.data_transforms = data_transforms
        self.batch_size = batch_size
        
        image_datasets = {x: FloodDataset(os.path.join(self.directory, x), self.data_transforms[x])
                        for x in self.training_phases}
        
        self.dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                        for x in self.training_phases}
           
        self.dataset_sizes = {x: len(image_datasets[x]) for x in self.training_phases}
        
        self.codes = {x:set() for x in self.training_phases}
        for phase in self.training_phases:
            for path, cls in self.dataloaders[phase].dataset.samples:
                self.codes[phase].add(os.path.basename(path).split(' ')[0])
        
    
    def __len__(self):
        return sum(self.dataset_sizes.values())
    
    def get_n_classes(self):
        path = os.path.join(self.directory, self.training_phases[0])
        return sum(os.path.isdir(os.path.join(path,i)) for i in os.listdir(path))
    