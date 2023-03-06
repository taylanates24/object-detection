import torch
import torch.nn as nn
import timm
import numpy as np
import itertools
from typing import List, Callable, Tuple

class TyNet(nn.Module):
    def __init__(self, num_classes: int=80, backbone: str='tf_efficientnet_lite0.in1k', out_size: int=64, **kwargs) -> None:
        """The object detector which has a changable backbone structure.

        Args:
            num_classes (int, optional): Number of classes in the dataset. Defaults to 80.
            backbone (str, optional): The backbone name from timm module. Defaults to 'tf_efficientnet_lite0.in1k'.
            out_size (int, optional): The size of the output of head layer. Defaults to 64.
        """
        
        super(TyNet, self).__init__()

        #avail_pretrained_models= timm.list_models(pretrained=True)

        self.anchor_scale = 4.0
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))

        self.num_classes = num_classes
        self.pyramid_level = 3
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
        x=torch.randn(1,3,224,224)
        out=self.backbone(x)
        layers = [x.shape[1] for x in out]
        

        self.neck = TyNeck(layers=layers, out_size=out_size)
        
        self.head = TyHead(out_size=out_size, num_anchors=9, num_classes=num_classes, num_layers=3, anchor_scale=self.anchor_scale,
                           pyramid_levels=(torch.arange(self.pyramid_level) + 3).tolist(), **kwargs)
        
    def forward(self, inputs: torch.tensor) -> Tuple:

        x = self.backbone(inputs)[-3:]
        x = self.neck(x)
        x, regression, classification, anchors = self.head(x, inputs, inputs.dtype)
        
        return x, regression, classification, anchors


class TyNeck(nn.Module):
    def __init__(self, layers: List[int]=[512, 1024, 2048], out_size: int=64, procedure: List[int]=[3, 4]) -> None:
        """The FPN neck function which uses addition as fusion function.

        Args:
            layers (List[int], optional): The list of number of channels in FPN outputs. Defaults to [512, 1024, 2048].
            out_size (int, optional): The size of the output of head layer Defaults to 64.
            procedure (List[int], optional): The number of layers of ScalableCSPResBlocks that applied to out of FPN 2 and 3 layers. Defaults to [3, 4].
        """
        super(TyNeck, self).__init__()

        self.out_layer1 = Conv(in_ch=layers[-1], out_ch=out_size * 2, k_size=3, s=1, p=1)
        self.out_layer2 = Conv(in_ch=out_size * 2, out_ch=out_size, k_size=3, s=1, p=1)
        self.out_layer = nn.Sequential(self.out_layer1, self.out_layer2)
        module_dict = {}
        for i in layers[:-1]:
            module_dict[str(i)] = Conv(in_ch=i, out_ch=layers[-1], k_size=1, s=1, p=0)

        self.upconv_dict = nn.ModuleDict(module_dict)

        self.layer1 = nn.Sequential(*[ScalableCSPResBlock(layers[-1])] * procedure[0])
        self.layer2 = nn.Sequential(*[ScalableCSPResBlock(layers[-1])] * procedure[1])
        self.layers = nn.ModuleList([self.layer1, self.layer2])
        self.upsample = nn.Upsample(scale_factor=2.0, mode='bilinear')

        self.act = nn.SiLU()

    def upsample_add(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        """Fusion function of FPN layer. It upsamples the nth FPN output and add it to the (n+1)th FPN output.

        Args:
            x (torch.tensor): Input layer comas from nth layer of FPN.
            y (torch.tensor): Input layer comas from (n+1)th layer of FPN.

        Returns:
            torch.tensor: Output
        """

        y = self.act(self.upconv_dict[str(y.shape[1])](y))

        return self.upsample(x) + y

    def forward(self, inputs: List[torch.tensor]) -> List[torch.tensor]:

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
    def __init__(self, 
                 out_size: int, 
                 anchor_scale: float, 
                 pyramid_levels: List[int], 
                 num_anchors: int=9, 
                 num_classes: int=80, 
                 num_layers: int=3, **kwargs) -> None:
        """The head class of the model. It applies classifier and bounding box regression to the neck output.

        Args:
            out_size (int): The output size of regression and classification.
            anchor_scale (float): The scale of anchors, depends on the target bounding box size of the dataset.
            pyramid_levels (List[int]): The pyramid levels comes from FPN.
            num_anchors (int, optional): Number of anchors. Defaults to 9.
            num_classes (int, optional): Number of classes in the dataset. Defaults to 80.
            num_layers (int, optional): Number of layers in classifier and regressor. Defaults to 3.
        """
        super(TyHead, self).__init__()

        self.regressor = Regressor(in_channels=out_size, num_anchors=num_anchors, num_layers=num_layers)
        self.classifier = Classifier(in_channels=out_size, num_anchors=9, num_classes=num_classes, num_layers=3)

        self.anchors = Anchors(anchor_scale=anchor_scale,
                               pyramid_levels=pyramid_levels,
                               **kwargs)
        
    def forward(self, x: List[torch.tensor], inputs: torch.tensor, dtype: torch.dtype) -> Tuple[List[torch.tensor], torch.tensor, torch.tensor, torch.tensor]:
        
        regression = self.regressor(x)
        classification = self.classifier(x)
        anchors = self.anchors(inputs, dtype)
        
        return x, regression, classification, anchors


class Regressor(nn.Module):

    def __init__(self, in_channels: int, num_anchors: int=9, num_layers: int=3) -> None:
        """The bounding box regression of the model. It gives a 1xNx4 shape output. The 4 corresponds to the coordinates of the bounding box.

        Args:
            in_channels (int): Input channel size.
            num_anchors (int, optional): Number of anchors. Defaults to 9.
            num_layers (int, optional): Number of layer in regressor. Defaults to 3.
        """
        super(Regressor, self).__init__()

        self.conv_list = nn.Sequential(
            *[ConvBlock(in_channels, in_channels, norm_2=True, norm_1=False, activation=True) for i in range(num_layers)])

        self.head_conv = ConvBlock(in_channels, num_anchors * 4, norm_1=False, norm_2=False, activation=False)

        self.act = nn.SiLU()


    def forward(self, x: List[torch.tensor]) -> torch.tensor:
        feats = []
        for feat in x: #    APPLY CSPNET

            feat = self.conv_list(feat)
            feat = self.head_conv(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)
        feats = torch.cat(feats, dim=1)
        
        return feats



class Classifier(nn.Module):

    def __init__(self, in_channels: int, num_anchors: int, num_classes:int, num_layers:int=3) -> None:
        """The bounding box regression of the model. It gives a 1xNxnum_classes shape output, N class probabilities for each class.
        
        Args:
            in_channels (int): Input channel size.
            num_anchors (int, optional): Number of anchors. Defaults to 9.
            num_classes (int): Number of classes in the dataset.
            num_layers (int, optional): Number of layer in regressor. Defaults to 3.
        """
        super(Classifier, self).__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.conv_list = nn.Sequential(
            *[ConvBlock(in_channels, in_channels, norm_2=True, norm_1=False, activation=True) for i in range(num_layers)])

        self.header = ConvBlock(in_channels, num_anchors * num_classes, norm_1=False, norm_2=False, activation=False)

        self.act = nn.SiLU()



    def forward(self, x: List[torch.tensor]) -> torch.tensor:


        feats = []
        for feat in x: #    APPLY CSPNET

            feat = self.conv_list(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)
        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()
        return feats


class ScalableCSPResBlock(nn.Module):

    def __init__(self, in_ch: int=512, num_basic_layers: int=1) -> None:
        """The scalable residual block with amplemented by using CSPNet. It is more accurate and faster than traditional resblock thanks to cross partial block.

        Args:
            in_ch (int, optional): The input size of the block. Defaults to 512.
            num_basic_layers (int, optional): Number of basic layer applied to the half of the frame. Defaults to 1.
        """
        super(ScalableCSPResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch * 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.LayerNorm(in_ch * 2)
        self.conv2 = nn.Conv2d(in_channels=in_ch * 2, out_channels=in_ch * 2, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.LayerNorm(in_ch * 2)
        basic_layers = []

        for _ in range(num_basic_layers):
            basic_layers.append(BasicBlock(in_ch, in_ch))

        self.basic_layers = nn.Sequential(*basic_layers)
        self.transition = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, stride=1, padding=0)
        self.act = nn.SiLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        
        x = self.conv1(x).permute(0, 2, 3, 1)
        x = self.bn1(x).permute(0, 3, 1, 2)
        x = self.act(x)
        
        x = self.conv2(x).permute(0, 2, 3, 1)
        x = self.bn2(x).permute(0, 3, 1, 2)
        x = self.act(x)


        xs, xb = x.split(x.shape[1] // 2, dim=1)
        xb = self.basic_layers(xb)

        out = self.transition(torch.cat([xs, xb], dim=1))

        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int=1, downsample: bool=False) ->None:
        """The basic block of residual networks.

        Args:
            in_channels (int): Input size.
            out_channels (int): Output size.
            stride (int, optional): The stride of the first convolution. Defaults to 1.
            downsample (bool, optional): Decides of downsample is applied. Defaults to False.
        """
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn2 = nn.LayerNorm(out_channels)

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2), bias=False),
                nn.LayerNorm(out_channels)
            )
        else:
            self.downsample = None
        self.out_channels = out_channels

    def forward(self, x: torch.tensor) -> torch.tensor:

        out = self.conv1(x).permute(0, 2, 3, 1)
        out = self.bn1(out).permute(0, 3, 1, 2)
        out = self.act(out)
        
        out = self.conv2(out).permute(0, 2, 3, 1)
        out = self.bn2(out).permute(0, 3, 1, 2)

        if self.downsample is not None:
            x = self.downsample(x)

        out = x + out
        out = self.act(out)

        return out


class Conv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k_size: int=3, s: int=1, p: int=0, act: Callable=nn.Mish(), norm: bool=True, bias: bool=True) -> None:
        """The conv2d with activation and normalization.

        Args:
            in_ch (int): Input channel size
            out_ch (int): Output channel size
            k_size (int, optional): Kernel size of Conv2d. Defaults to 3.
            s (int, optional): Stride of Conv2d. Defaults to 1.
            p (int, optional): Padding of Conv2d. Defaults to 0.
            act (Callable, optional): Activation function. Defaults to nn.Mish().
            norm (bool, optional): Decides if normalization applied. Defaults to True.
            bias (bool, optional): Bias boolean of Conv2d. Defaults to True.
        """
        super(Conv, self).__init__()

        self.norm = norm
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=s, padding=p, bias=bias)

        if norm:
            self.bn = nn.LayerNorm(out_ch)


    def forward(self, x: torch.tensor) -> torch.tensor:

        if self.norm:
            x = self.conv(x).permute(0, 2, 3, 1)
            x = self.bn(x).permute(0, 3, 1, 2)
            return x

        else:

            return self.conv(x)


class ConvBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, norm_1: bool, norm_2: bool, activation: bool) -> None:
        """The 3x3 and 1x1 (actually any size) convolution block in regressor and classifier.

        Args:
            in_channel (int): Input channel size.
            out_channel (int): Output channel size.
            norm_1 (bool): Decides if normalization is applied to the first conv block.
            norm_2 (bool): Decides if normalization is applied to the Second conv block.
            activation (bool): Decides if activation is applied to the end of the block.
        """
        super().__init__()

        self.activation = activation
        self.conv1 = Conv(in_ch=in_channel, out_ch=in_channel, k_size=3, s=1, p=1, norm=norm_1, bias=False)

        self.conv2 = Conv(in_ch=in_channel, out_ch=out_channel, k_size=1, s=1, norm=norm_2)

        if self.activation:
            self.act = nn.SiLU()
        #self.bn = nn.LayerNorm(out_channel)

    def forward(self, x: torch.tensor) -> torch.tensor:

        x = self.conv1(x)
        x = self.conv2(x)

        if self.activation:
            return self.act(x)

        return x


class Anchors(nn.Module):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117
    """

    def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
        self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])

        self.last_anchors = {}
        self.last_shape = None

    def forward(self, image, dtype=torch.float32):
        """Generates multiscale anchor boxes.

        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

        Returns:
          anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """
        image_shape = image.shape[2:]

        if image_shape == self.last_shape and image.device in self.last_anchors:
            return self.last_anchors[image.device]

        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape

        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)

        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)

        # save it for later use to reduce overhead
        self.last_anchors[image.device] = anchor_boxes
        return anchor_boxes