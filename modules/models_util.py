
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class Erase(object):
    '''
    '''
    def __init__(self, i=39,j=0,h=90,w=265,v=0):
        self.i = i
        self.j = j
        self.h = h
        self.w = w
        self.v = v
    
    def __call__(self, img):
        img_array = np.array(img)
        img_array[self.i:self.i+self.h, self.j:self.j+self.w, :] = 0
        return Image.fromarray(img_array)
            
        
class PytorchModel():
    
    def __init__(self, mean, std, data_transforms, model_func, model_weigths):
        self.mean = mean
        self.std = std
        self.data_transforms = data_transforms
        self.model_function = model_func
        self.model_args = model_weigths
        
    def _grad_and_load_weights(self, model, weights_path:str=None, fully_trainable=True):
        if fully_trainable:
            for param in model.parameters():
                param.requires_grad = True
            print(f"All parameters for model {model._get_name()} requires grad.")         
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path))
            print(f"Weights for model {model._get_name()} loaded from {weights_path}")
        return model
    
    def _change_fc_layer(self, model):
        pass
    
    def load(self, weights_path:str=None, fully_trainable=True):
        model = self.model_function(weights=self.model_args)
        model = self._change_fc_layer(model)
        return self._grad_and_load_weights(model, weights_path, fully_trainable)

class EfficientNet(PytorchModel):
    def __init__(self):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        data_transforms = {
        'train': transforms.Compose([
            Erase(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(256, transforms.InterpolationMode('bilinear')),
            transforms.CenterCrop(224),
            transforms.ToTensor(), # When converting to tensor, pytorch automatically rescales to [0,1]
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            Erase(),
            transforms.Resize(256, transforms.InterpolationMode('bilinear')),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            Erase(),
            transforms.Resize(256, transforms.InterpolationMode('bilinear')),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        }
        super().__init__(mean, std, data_transforms, models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT)
    
    def _change_fc_layer(self, model):
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=2)
            )
        return model

class ViT():
    def __init__(self):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
        super().__init__(mean, std, data_transforms, models.vit_b_16, models.ViT_B_16_Weights.DEFAULT)
    def _change_fc_layer(self, model):
        model.heads = nn.Sequential(
            nn.Linear(in_features=768, out_features=2)
            )
        return model
    
class VGG19():
    def __init__(self):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(256, transforms.InterpolationMode('bilinear')),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256, transforms.InterpolationMode('bilinear')),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(256, transforms.InterpolationMode('bilinear')),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
        super().__init__(mean, std, data_transforms, models.vgg19, models.VGG19_Weights.DEFAULT)
    
    def _change_fc_layer(self, model):
        model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1000, bias=True)
            )
        return model
    
class ResNet():
    # resnet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    
    def load(self, weights_path:str=None):
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        
        model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=2)
        )
        
        for param in model.parameters():
            param.requires_grad = True
            
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path))
        return model