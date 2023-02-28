import torch
import torch.nn as nn
import timm
import numpy as np
import itertools
from models.utils import Swish
#timm.list_models(pretrained=True)
#'hrnet_w64'
#tf_efficientnet_b3.ns_jft_in1k
#'tf_efficientnet_b2.ns_jft_in1k'
#'tf_efficientnet_b0.aa_in1k'   CHECKED OK SLOW
#'tf_efficientnet_lite0.in1k'  CHECKED OK BETTER
#'tf_efficientnetv2_b0.in1k'   LIGHTWEIGHT
#'tf_efficientnet_lite4.in1k'   CHECKED SLOW
#'tf_inception_v3'
#'tf_mixnet_m.in1k' CHECKED
#'tf_mobilenetv3_small_minimal_100.in1k'  # CHECKED SLOW
#'efficientnetv2_rw_t.ra2_in1k' # CHECKED
#'efficientnet_lite0.ra_in1k' # CHECKED SLOWER
#'mobilenetv3_small_100.lamb_in1k' #CHECKED SLOW BAD
#'tf_efficientnet_b0.ap_in1k' # CHECKED SLOWER GOOD
#'tf_efficientnetv2_s.in1k'  # CHECKED SLOWER GOOD
# 'efficientnet_l2' TRY
# 'tf_efficientnet_l2' TRY
class TyNet(nn.Module):
    def __init__(self, num_classes=80, backbone='tf_efficientnetv2_b0.in1k', out_size=64, nc=80, compound_coef=0, **kwargs) -> None:
        super(TyNet, self).__init__()

        #avail_pretrained_models= timm.list_models()[600:]
        self.compound_coef = compound_coef
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        num_anchors = len(self.aspect_ratios) * self.num_scales
        self.num_classes = num_classes
        self.pyramid_levels = [3]
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
        x=torch.randn(1,3,224,224)
        out=self.backbone(x)
        layers = [x.shape[1] for x in out]
        
        #avail_pretrained_models = timm.list_models(pretrained=True)

        self.neck = TyNeck(layers=layers, out_size=out_size)
        #self.regressor = Regressor(in_channels=out_size, num_anchors=num_anchors, num_layers=3, pyramid_levels=3)
        #self.classifier = Classifier(in_channels=out_size, num_anchors=num_anchors, num_classes=self.num_classes, num_layers=3)

        #self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
        #                       pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
        #                       **kwargs)
        
        self.head = TyHead(out_size=out_size, num_anchors=9, num_classes=num_classes, num_layers=3, anchor_scale=self.anchor_scale[compound_coef],
                           pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(), **kwargs)
        
    def forward(self, inputs):

        x = self.backbone(inputs)[-3:]
        x = self.neck(x)
        x, regression, classification, anchors = self.head(x, inputs, inputs.dtype)
        
        return x, regression, classification, anchors


class TyNeck(nn.Module):
    def __init__(self, layers=[512, 1024, 2048], out_size=64, procedure=[2, 3]) -> None:
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
    def __init__(self, out_size, anchor_scale, pyramid_levels, num_anchors=9, num_classes=80, num_layers=3, **kwargs):
        super(TyHead, self).__init__()

        self.regressor = Regressor(in_channels=out_size, num_anchors=num_anchors, num_layers=num_layers)
        self.classifier = Classifier(in_channels=out_size, num_anchors=9, num_classes=num_classes, num_layers=3)

        self.anchors = Anchors(anchor_scale=anchor_scale,
                               pyramid_levels=pyramid_levels,
                               **kwargs)
        
    def forward(self, x, inputs, dtype):
        
        regression = self.regressor(x)
        classification = self.classifier(x)
        anchors = self.anchors(inputs, dtype)
        
        return x, regression, classification, anchors



class Regressor(nn.Module):

    def __init__(self, in_channels, num_anchors=9, num_layers=3, pyramid_levels=3) -> None:
        super(Regressor, self).__init__()

        self.conv_list = nn.Sequential(
            *[ConvBlock(in_channels, in_channels, norm_2=True, norm_1=False, activation=True) for i in range(num_layers)])

        self.header = ConvBlock(in_channels, num_anchors * 4, norm_1=False, norm_2=False, activation=False)

        self.act = nn.SiLU()


    def forward(self, x):
        feats = []
        for feat in x: #    APPLY CSPNET

            feat = self.conv_list(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)
        feats = torch.cat(feats, dim=1)
        
        return feats



class Classifier(nn.Module):

    def __init__(self, in_channels, num_anchors, num_classes, num_layers=3, pyramid_levels=3) -> None:
        super(Classifier, self).__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.conv_list = nn.Sequential(
            *[ConvBlock(in_channels, in_channels, norm_2=True, norm_1=False, activation=True) for i in range(num_layers)])

        self.header = ConvBlock(in_channels, num_anchors * num_classes, norm_1=False, norm_2=False, activation=False)

        self.act = nn.SiLU()



    def forward(self, x):


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
        self.act = nn.SiLU()

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
        self.act = nn.SiLU()
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
    def __init__(self, in_ch, out_ch, k_size=3, s=1, p=0, upsample=False, act=nn.Mish(), norm=True, bias=True) -> None:
        super(Conv, self).__init__()

        self.norm = norm
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=s, padding=p, bias=bias)

        if norm:
            self.bn = nn.BatchNorm2d(num_features=out_ch)


    def forward(self, x):

        if self.norm:

            return self.bn(self.conv(x))

        else:

            return self.conv(x)


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, norm_1, norm_2, activation) -> None:
        super().__init__()

        self.activation = activation
        self.conv1 = Conv(in_ch=in_channel, out_ch=in_channel, k_size=3, s=1, p=1, norm=norm_1, bias=False)

        self.conv2 = Conv(in_ch=in_channel, out_ch=out_channel, k_size=1, s=1, norm=norm_2)

        if self.activation:
            self.act = nn.SiLU()
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):

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