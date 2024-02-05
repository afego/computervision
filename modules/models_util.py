import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

class EfficientNet():
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(), # When converting to tensor, pytorch automatically rescales to [0,1]
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    # probably shouldnt do this
    'test': transforms.Compose([
        #transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    }
        
    def load(weights_path:str=None):
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # A newly defined layer is created with requires_grad=True by default
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=2)
            )
        # if weights_path is not None:
        #     model.load_state_dict(torch.load(weights_path))
        return model

class ViT():
    # vit_b_16
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
            transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    
    def load(weights_path=None):
        # model = models.vit_b_16()
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        model.heads = nn.Sequential(
            nn.Linear(in_features=768, out_features=2)
            )
        # if len(weights_path):
        #     model.load_state_dict(torch.load(weights_path))
            
        return model

class VGG19():
    # vgg19
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
            transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    
    def load(self, weights_path:str=None):
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        
        model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=2)
        )
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path))
        return model