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
        transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    }
        
    def load(weights_path):
        model = models.efficientnet_b0()

        # A newly defined layer is created with requires_grad=True by default
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=2)
            )

        model.load_state_dict(torch.load(weights_path))
        return model