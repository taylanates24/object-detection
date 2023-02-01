from statistics import mode
from turtle import forward
import torch
import torch.nn as nn
import timm



class TyNet(nn.Module):
    def __init__(self, backbone='cspresnet50') -> None:
        super(TyNet, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True).cuda()
        self.neck = nn.Identity()
        self.head = nn.Identity()
        
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

class TyNeck(nn.Module):
    def __init__(self) -> None:
        super(TyNeck, self).__init__()
    
        pass
    
    def forward(self, x):
        
        pass


class TyHead(nn.Module):
    def __init__(self) -> None:
        super(TyHead, self).__init__()
    
        pass
    
    def forward(self, x):
        
        pass