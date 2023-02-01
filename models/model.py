from statistics import mode
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
