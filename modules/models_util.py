
import numpy as np
import torch
import torch.nn as nn

from modules.image_util import Erase, LowerBrightness, Equalize, UnsharpMask
from torchvision import models, transforms
    
class PytorchModel():
    
    def __init__(self, mean, std, data_transforms, model_func, model_weigths, out_features, device='cuda'):
        self.mean = mean
        self.std = std
        self.data_transforms = data_transforms
        self.model_function = model_func
        self.model_args = model_weigths
        self.device = device
        self.out_features = out_features
        
    def _grad_and_load_weights(self, model, weights_path:str=None, fully_trainable=True):
        # if fully_trainable:
        #     for param in model.parameters():
        #         param.requires_grad = True
        #     print(f"All parameters for model {model._get_name()} requires grad.")
        # else:
        #     print(f"Only fully connected layer requires grad.")         
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device(self.device)))
            print(f"Weights for model {model._get_name()} loaded from {weights_path}")
        return model
    
    def _change_fc_layer(self, model):
        pass
    
    def load(self, weights_path:str=None, fully_trainable=True):
        model = self.model_function(weights=self.model_args)
        if not fully_trainable:
            for param in model.parameters():
                param.requires_grad = False
            print("Only fully connected layer requires grad.")
        else:
            print(f"All parameters for model {model._get_name()} requires grad.")
        model = self._change_fc_layer(model)
        return self._grad_and_load_weights(model, weights_path, fully_trainable)

class EfficientNet(PytorchModel):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arch_opts = {
        'b0': [models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 256, 224],
        'b1': [models.efficientnet_b1, models.EfficientNet_B1_Weights.IMAGENET1K_V1, 256, 240],
        'b2': [models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1, 288, 288],
        'b3': [models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1, 320, 300],
        'b4': [models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1, 384, 380]
    }
    def __init__(self, out_features, architecture:str='b0', device='cuda'):
        
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
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
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
        
        super().__init__(EfficientNet.mean, EfficientNet.std, self.data_transforms, model, weights, device, out_features)
    
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
    
    def __init__(self, architecture:str='base', device='cuda', bp=0.5):
        self.arch = architecture.lower()
        
        if self.arch not in ViT.arch_opts.keys():
            self.arch = 'base'
            print(f"\nChosen architecture {architecture} not in allowed options: {ViT.arch_opts.keys()}.\nUsing base architecture.\n")
        
        model, weights = ViT.arch_opts[self.arch]
        
        self.data_transforms = {
            'train': transforms.Compose([
                Erase(),
                # Equalize(),
                LowerBrightness(bp),
                # UnsharpMask(),    
                transforms.RandomHorizontalFlip(),
                transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'val': transforms.Compose([
                Erase(),
                # Equalize(),
                LowerBrightness(bp),
                # UnsharpMask(),
                transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'test': transforms.Compose([
                # Erase(),
                # Equalize(),
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

class VGG19(PytorchModel):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    def __init__(self, device='cuda', bp=0.5):
        
        model = models.vgg19
        weights = models.VGG19_Weights.DEFAULT
        
        self.data_transforms = {
            'train': transforms.Compose([
                Erase(),
                LowerBrightness(bp),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(256, transforms.InterpolationMode('bilinear')),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'val': transforms.Compose([
                Erase(),
                LowerBrightness(bp),
                transforms.Resize(256, transforms.InterpolationMode('bilinear')),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'test': transforms.Compose([
                Erase(),
                transforms.Resize(256, transforms.InterpolationMode('bilinear')),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        }
        super().__init__(VGG19.mean, VGG19.std, self.data_transforms, model, weights, device)
    
    def _change_fc_layer(self, model):
        model.classifier = nn.Sequential(
            nn.Linear(in_features=model.classifier[0].in_features, out_features=model.classifier[0].out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=model.classifier[2].p, inplace=False),
            nn.Linear(in_features=model.classifier[3].in_features, out_features=model.classifier[3].out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=model.classifier[5].p, inplace=False),
            nn.Linear(in_features=model.classifier[6].in_features, out_features=2)
            )
        # model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=2)
        return model

class DenseNet(PytorchModel):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arch_opts = {
        '121': [models.densenet121, models.DenseNet121_Weights],
        '161': [models.densenet161, models.DenseNet161_Weights],
        '169': [models.densenet169, models.DenseNet169_Weights]
        }    
    
    def __init__(self, architecture:str='121', device='cuda', bp=0.5):
        self.arch = architecture.lower()
        
        if self.arch not in DenseNet.arch_opts.keys():
            self.arch = '121'
            print(f"\nChosen architecture {architecture} not in allowed options: {DenseNet.arch_opts.keys()}.\nUsing base architecture.\n")
        
        model, weights = DenseNet.arch_opts[self.arch]
        
        self.data_transforms = {
            'train': transforms.Compose([
                Erase(),
                LowerBrightness(bp),
                # transforms.RandomPerspective(),
                # transforms.ElasticTransform(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'val': transforms.Compose([
                Erase(),
                LowerBrightness(bp),
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

class ResNeXt(PytorchModel):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    arch_opts = {
        '50': [models.resnext50_32x4d, models.ResNeXt50_32X4D_Weights.DEFAULT, 256],
        '101_32': [models.resnext101_32x8d, models.ResNeXt101_32X8D_Weights.DEFAULT, 256],
        '101_64': [models.resnext101_64x4d, models.ResNeXt101_64X4D_Weights.DEFAULT, 232]
        }    
    
    def __init__(self, architecture:str='50', device='cuda'):
        self.arch = architecture.lower()
        
        if self.arch not in ResNeXt.arch_opts.keys():
            self.arch = '50'
            print(f"\nChosen architecture {architecture} not in allowed options: {ResNeXt.arch_opts.keys()}.\nUsing default {self.arch} architecture.\n")
        
        model, weights, resize_size = ResNeXt.arch_opts[self.arch]
        
        self.data_transforms = {
            'train': transforms.Compose([
                Erase(),
                LowerBrightness(),
                # CLAHE(),
                # Normalize(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(resize_size, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'val': transforms.Compose([
                Erase(),
                LowerBrightness(),
                # CLAHE(),
                # Normalize(),
                transforms.Resize(resize_size, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'test': transforms.Compose([
                Erase(),
                transforms.Resize(resize_size, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        }
        super().__init__(ResNeXt.mean, ResNeXt.std, self.data_transforms, model, weights, device)
        
    def _change_fc_layer(self, model):
        # model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)
        
        model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=10),
            nn.Linear(10,2)
        )
        return model

class MobileNet(PytorchModel):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    arch_opts = {
        'v2': [models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT],
        'v3_small': [models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT],
        'v3_large': [models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT]
        } 
    
    def __init__(self, architecture='v2',device='cuda',bp=0.5):
        self.arch = architecture.lower()
        
        if self.arch not in MobileNet.arch_opts.keys():
            self.arch = 'v2'
            print(f"\nChosen architecture {architecture} not in allowed options: {MobileNet.arch_opts.keys()}.\nUsing default {self.arch} architecture.\n")
            
        model, weights = self.arch_opts[self.arch]
        resize_size = 256
        crop_size = 224
        
        self.data_transforms = {
            'train': transforms.Compose([
                Erase(),
                LowerBrightness(bp),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(resize_size, transforms.InterpolationMode('bilinear')),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'val': transforms.Compose([
                Erase(),
                LowerBrightness(bp),
                transforms.Resize(resize_size, transforms.InterpolationMode('bilinear')),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'test': transforms.Compose([
                Erase(),
                transforms.Resize(resize_size, transforms.InterpolationMode('bilinear')),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        }
        super().__init__(MobileNet.mean, MobileNet.std, self.data_transforms, model, weights, device)
    
    def _change_fc_layer(self, model):
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=2)
        return model

class InceptionV3(PytorchModel):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    def __init__(self, device='cuda',bp=0.5):
        
        model = models.inception_v3
        weights = models.Inception_V3_Weights.DEFAULT
        resize_size = 342
        crop_size = 299
        
        self.data_transforms = {
            'train': transforms.Compose([
                Erase(),
                LowerBrightness(bp),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(resize_size, transforms.InterpolationMode('bilinear')),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'val': transforms.Compose([
                Erase(),
                LowerBrightness(bp),
                transforms.Resize(resize_size, transforms.InterpolationMode('bilinear')),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'test': transforms.Compose([
                Erase(),
                transforms.Resize(resize_size, transforms.InterpolationMode('bilinear')),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        }
        super().__init__(InceptionV3.mean, InceptionV3.std, self.data_transforms, model, weights, device)
    
    def _change_fc_layer(self, model):
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)
        
        # model.fc = nn.Sequential(
        #     nn.Linear(in_features=model.fc.in_features, out_features=10),
        #     nn.Linear(10,2)
        # )
        
        model.aux_logits=False
        return model