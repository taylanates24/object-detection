import torch
import torch.nn as nn
import timm


class TyNet(nn.Module):
    def __init__(self, backbone='cspdarknet53', nc=80) -> None:
        super(TyNet, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
        x=torch.randn(1,3,224,224)
        out=self.backbone(x)
        layers = [x.shape[1] for x in out][-3:]
        
        #avail_pretrained_models = timm.list_models(pretrained=True)
        self.neck = TyNeck(layers=layers)
        self.head = TyHead(nc=nc+5, num_outs=len(layers))
        
    def forward(self, x):
        x = self.backbone(x)[-3:]
        x = self.neck(x)
        x = self.head(x)
        return x


class TyNeck(nn.Module):
    def __init__(self, layers=[256, 512, 1024], out_size=256, procedure=[2, 3]) -> None:
        super(TyNeck, self).__init__()

        self.out_layer = Conv(in_ch=layers[-1], out_ch=out_size, k_size=3, s=1, p=1)

        module_dict = {}
        for i in layers[:-1]:
            module_dict[str(i)] = Conv(in_ch=i, out_ch=layers[-1], k_size=1, s=1, p=0)

        self.upconv_dict = nn.ModuleDict(module_dict)

        self.layer1 = nn.Sequential(*[ScalableCSPResBlock(layers[-1])] * procedure[0])
        self.layer2 = nn.Sequential(*[ScalableCSPResBlock(layers[-1])] * procedure[1])
        self.layers = nn.ModuleList([self.layer1, self.layer2])
        self.upsample = nn.Upsample(scale_factor=2.0, mode='bilinear')

        self.act = nn.SiLU()

    def upsample_add(self, x, y):

        y = self.act(self.upconv_dict[str(y.shape[1])](y))

        return self.upsample(x) + y

    def forward(self, inputs):

        x = inputs.pop(-1)

        mids = [x]
        for layer in self.layers:
            x = self.upsample_add(x, inputs.pop(-1))
            x = layer(x)
            mids.append(x)

        outs = []
        for mid in mids:
            outs.append(self.out_layer(mid))

        return outs


class TyHead(nn.Module):
    def __init__(self, in_ch=256, out_ch=255, nc=85) -> None:
        super(TyHead, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)


        self.nc = nc
        
    def forward(self, inputs):
        for i in range(len(inputs)):
            inputs[i] = self.conv(inputs[i])
            bs, _, ny, nx = inputs[i].shape
            inputs[i] = inputs[i].view(bs, self.na, self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()



class ScalableCSPResBlock(nn.Module):

    def __init__(self, in_ch=512, num_basic_layers=1) -> None:
        super(ScalableCSPResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch * 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=in_ch * 2)
        self.conv2 = nn.Conv2d(in_channels=in_ch * 2, out_channels=in_ch * 2, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=in_ch * 2)
        basic_layers = []

        for _ in range(num_basic_layers):
            basic_layers.append(BasicBlock(in_ch, in_ch))

        self.basic_layers = nn.Sequential(*basic_layers)
        self.transition = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, stride=1, padding=0)
        self.act = nn.Mish()

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))

        xs, xb = x.split(x.shape[1] // 2, dim=1)
        xb = self.basic_layers(xb)

        out = self.transition(torch.cat([xs, xb], dim=1))

        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.Mish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
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
