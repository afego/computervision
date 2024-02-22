
import numpy as np
import torch
import torch.nn as nn

from modules.image_util import Erase, LowerBrightness, Normalize
from torchvision import models, transforms
    
class PytorchModel():
    
    def __init__(self, mean, std, data_transforms, model_func, model_weigths, device='cuda'):
        self.mean = mean
        self.std = std
        self.data_transforms = data_transforms
        self.model_function = model_func
        self.model_args = model_weigths
        self.device = device
        
    def _grad_and_load_weights(self, model, weights_path:str=None, fully_trainable=True):
        if fully_trainable:
            for param in model.parameters():
                param.requires_grad = True
            print(f"All parameters for model {model._get_name()} requires grad.")         
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device(self.device)))
            print(f"Weights for model {model._get_name()} loaded from {weights_path}")
        return model
    
    def _change_fc_layer(self, model):
        pass
    
    def load(self, weights_path:str=None, fully_trainable=True):
        model = self.model_function(weights=self.model_args)
        model = self._change_fc_layer(model)
        return self._grad_and_load_weights(model, weights_path, fully_trainable)

class EfficientNet(PytorchModel):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arch_opts = {
        'b0': [models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 256, 224],
        'b1': [models.efficientnet_b1, models.EfficientNet_B1_Weights.IMAGENET1K_V1, 256, 240],
        'b2': [models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1, 288, 288],
        'b4': [models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1, 384, 380]
    }
    def __init__(self, architecture:str='b0', device='cuda'):
        
        self.arch = architecture.lower()
        
        if self.arch not in EfficientNet.arch_opts.keys():
            self.arch = 'b0'
            print(f"\nChosen architecture {architecture} not in allowed options: {EfficientNet.arch_opts.keys()}.\nUsing base architecture.\n")
        
        model, weights, self.resize, self.crop_size = self.arch_opts[self.arch]
        
        self.data_transforms = {
            'train': transforms.Compose([
                Erase(),
                LowerBrightness(),
                # Normalize(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(self.resize, transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(), # When converting to tensor, pytorch automatically rescales to [0,1]
                transforms.Normalize(self.mean, self.std)
                # weights.transforms
            ]),
            'val': transforms.Compose([
                Erase(),
                LowerBrightness(),
                # Normalize(),
                transforms.Resize(self.resize, transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'test': transforms.Compose([
                Erase(),
                transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        }
        
        super().__init__(EfficientNet.mean, EfficientNet.std, self.data_transforms, model, weights, device)
    
    def _change_fc_layer(self, model):
        model.classifier = nn.Sequential(
            nn.Dropout(p=model.classifier[0].p, inplace=True),
            nn.Linear(in_features=model.classifier[1].in_features, out_features=2)
            )
        return model

class ViT(PytorchModel):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    arch_opts = {
        'base': [models.vit_b_16, models.ViT_B_16_Weights.DEFAULT],
        'b32': [models.vit_b_32, models.ViT_B_32_Weights.DEFAULT],
        'large': [models.vit_l_16, models.ViT_L_16_Weights.IMAGENET1K_V1],
        'huge': [models.vit_h_14, models.ViT_H_14_Weights.DEFAULT]
        }    
    
    def __init__(self, architecture:str='base', device='cuda'):
        self.arch = architecture.lower()
        
        if self.arch not in ViT.arch_opts.keys():
            self.arch = 'base'
            print(f"\nChosen architecture {architecture} not in allowed options: {ViT.arch_opts.keys()}.\nUsing base architecture.\n")
        
        model, weights = ViT.arch_opts[self.arch]
        
        self.data_transforms = {
            'train': transforms.Compose([
                Erase(),
                LowerBrightness(),
                # Normalize(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'val': transforms.Compose([
                Erase(),
                LowerBrightness(),
                # Normalize(),
                transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'test': transforms.Compose([
                Erase(),
                transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        }
        super().__init__(ViT.mean, ViT.std, self.data_transforms, model, weights, device)
        
    def _change_fc_layer(self, model):
        model.heads = nn.Sequential(
            nn.Linear(in_features=model.heads.head.in_features, out_features=2)
            )
        return model

class DenseNet(PytorchModel):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arch_opts = {
        '121': [models.densenet121, models.DenseNet121_Weights],
        '161': [models.densenet161, models.DenseNet161_Weights],
        '169': [models.densenet169, models.DenseNet169_Weights]
        }    
    
    def __init__(self, architecture:str='121', device='cuda'):
        self.arch = architecture.lower()
        
        if self.arch not in DenseNet.arch_opts.keys():
            self.arch = '121'
            print(f"\nChosen architecture {architecture} not in allowed options: {DenseNet.arch_opts.keys()}.\nUsing base architecture.\n")
        
        model, weights = DenseNet.arch_opts[self.arch]
        
        self.data_transforms = {
            'train': transforms.Compose([
                Erase(),
                LowerBrightness(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'val': transforms.Compose([
                Erase(),
                LowerBrightness(),
                transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'test': transforms.Compose([
                Erase(),
                transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        }
        
        super().__init__(DenseNet.mean, DenseNet.std, self.data_transforms, model, weights, device)

    def _change_fc_layer(self, model):
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=2)
        return model

class GoogLeNet(PytorchModel):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self, device='cuda'):
        
        self.data_transforms = {
            'train': transforms.Compose([
                Erase(),
                LowerBrightness(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'val': transforms.Compose([
                Erase(),
                LowerBrightness(),
                transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'test': transforms.Compose([
                Erase(),
                transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        }
        super().__init__(GoogLeNet.mean, GoogLeNet.std, self.data_transforms, models.googlenet, models.GoogLeNet_Weights, device)
    
    def _change_fc_layer(self, model):
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)
        return model
    
class VGG19(PytorchModel):
    
    def __init__(self):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        data_transforms = {
            'train': transforms.Compose([
                Erase(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(256, transforms.InterpolationMode('bilinear')),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
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
    
class ResNet(PytorchModel):
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