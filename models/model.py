import torch
import torch.nn as nn
import timm



class TyNet(nn.Module):
    def __init__(self, backbone='cspdarknet53') -> None:
        super(TyNet, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True, out_indices=[3,4,5]).cuda()
        self.neck = TyNeck()
        self.head = nn.Identity()
        
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

class TyNeck(nn.Module):
    def __init__(self) -> None:
        super(TyNeck, self).__init__()
        self.conv_out = nn.Sequential(Conv(1024, 512, k_size=3, s=1, p=1), nn.Mish())
        
        self.conv_p5 = nn.Sequential(Conv(512,1024,1,1,0), nn.Mish())
        self.conv_p4 = nn.Sequential(Conv(256,1024,1,1,0), nn.Mish())
        self.basic_block = ScalableCSPResBlock(1024)
        self.upsample = nn.Upsample(scale_factor=2.0)
        self.act = nn.Mish()
        

    
    def forward(self, x):
        p4, p5, p6 = x
        
        out_6 = self.conv_out(p6)
        f6 = self.upsample(p6)
        p5 = self.conv_p5(p5)
        
        f5 = p5 + f6
        
        out_5 = self.basic_block(self.basic_block(f5))
        
        f5 = self.upsample(f5)
        p4 = self.conv_p4(p4)
        
        f4 = f5 + p4
        out_4 = self.basic_block(self.basic_block(self.basic_block(f4)))
        
        out_5 = self.conv_out(out_5)
        
        out_4 = self.conv_out(out_4)
        
        
        return out_4, out_5, out_6
        


class TyHead(nn.Module):
    def __init__(self) -> None:
        super(TyHead, self).__init__()
    
        pass
    
    def forward(self, x):
        
        pass
    

class ScalableCSPResBlock(nn.Module):

    def __init__(self, in_ch=512, num_basic_layers=1) -> None:
        super(ScalableCSPResBlock, self).__init__()

        
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch*2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=in_ch*2)
        self.conv2 = nn.Conv2d(in_channels=in_ch*2, out_channels=in_ch*2, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=in_ch*2)
        basic_layers = []
        
        for _ in range(num_basic_layers):
            basic_layers.append(BasicBlock(in_ch, in_ch))
            
        self.basic_layers = nn.Sequential(*basic_layers)
        self.transition = nn.Conv2d(in_ch*2, in_ch, kernel_size=1, stride=1, padding=0)
        self.act = nn.Mish()

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        
        xs, xb = x.split(x.shape[1] // 2, dim=1)
        xb = self.basic_layers(xb)
        
        out = self.transition(torch.cat([xs, xb], dim=1))

        return out



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.Mish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1,1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2), bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
        else:
            self.downsample = None
        self.out_channels = out_channels
    
    
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            x = self.downsample(x)
            
        out = x + out
        out = self.act(out)
        
        return out


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, s=1, p=0, upsample=False, act=nn.Mish()) -> None:
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2.0)
            self.act = act
        self.is_upsample = upsample
        
    def forward(self, x):
        if self.is_upsample:
            return self.upsample(self.act(self.bn(self.conv(x))))
        return self.bn(self.conv(x))

