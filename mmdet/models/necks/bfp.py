# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import NonLocal2d
from mmcv.runner import BaseModule

from ..builder import NECKS


#####################################################################################################
# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS
############################################

import torch
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        # self.branch3 = nn.Sequential(
        #     BasicConv2d(in_channel, out_channel, 1),
        #     BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
        #     BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
        #     BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        # )

        self.branch3 = CoordAtt(in_channel, out_channel)
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



class MSCA(nn.Module):
    def __init__(self, channels=256, r=4):
        super(MSCA, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):

        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)

        return wei



def cus_sample(feat, **kwargs):
    """
    :param feat: 输入特征
    :param kwargs: size或者scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)


def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False)
    return y

class DGCM(nn.Module):
    def __init__(self, channel=256):
        super(DGCM, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)

        self.h2l = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.h2h = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

        self.mscah = MSCA()
        self.mscal = MSCA()

        self.upsample_add = upsample_add
        self.conv = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

    def forward(self, x):

        # first conv
        x_h = self.h2h(x)
        x_l = self.h2l(self.h2l_pool(x))
        x_h = x_h * self.mscah(x_h)
        x_l = x_l * self.mscal(x_l)
        out = self.upsample_add(x_l, x_h)
        out = self.conv(out)

        return out


#####################################################################################################



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

######################################################

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2)*dilation, bias=bias, dilation=(dilation,dilation))

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class HRAB(nn.Module):
    def __init__(self, conv=default_conv, n_feats=256):
        super(HRAB, self).__init__()


        kernel_size_1 = 3

        reduction = 4

        self.conv_du_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv(n_feats, n_feats // reduction, 1),
            nn.LeakyReLU(inplace=True),
            conv(n_feats // reduction, n_feats, 1),
            nn.Sigmoid()
        )

        self.conv_3 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats, n_feats, kernel_size_1, dilation=2)

        self.conv_3_1 = conv(n_feats*2, n_feats, kernel_size_1)
        self.conv_3_2_1 = conv(n_feats*2, n_feats, kernel_size_1, dilation=2)

        self.LR = nn.LeakyReLU(inplace=True)

        self.conv_11 = conv(n_feats*2, n_feats, 1)



    def forward(self, x):

        res_x = x

        a =  self.conv_du_1(x)
        b1 = self.LR(self.conv_3(x))
        b2 = self.LR(self.conv_3_2(x)) + b1
        B = torch.cat([ b1, b2 ],1)

        b1 = self.conv_3_1(B)
        b2 = self.LR(self.conv_3_2_1(B)) + b1

        B = torch.cat([b1,b2], 1)

        B = self.conv_11(B)


        output = a*B

        output = output + res_x

        return output


###################################################################################



class 1*1_layer(nn.Module):
 

    def __init__(self, channel, gamma=2, b=1):
        super(1*1_layer, self).__init__()

        t = int(abs((np.log2(channel) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv = nn.Conv1D(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  
        y = self.avg_pool(x)  

        y = y.squeeze(-1)  
        y = paddle.transpose(y, [0, 2, 1])  
        y = self.conv(y)  # shape=1,1,16
        y = paddle.transpose(y, [0, 2, 1])  
        y = y.unsqueeze(-1) 

        y = self.sigmoid(y)

        return x * y.expand_as(x)

####################################################################################
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
                BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision+1, dilation=vision+1, relu=False, groups=groups)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1), groups=groups),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1, groups=groups),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

#################################################################################
class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1)
        self.cv3 = nn.Conv2d(c_, c_, 3, 1)
        self.cv4 = nn.Conv2d(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = nn.Conv2d(4 * c_, c_, 1, 1)
        self.cv6 = nn.Conv2d(c_, c_, 3, 1)
        self.cv7 = nn.Conv2d(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        #return self.cv7(torch.cat((y1, y2), dim=1))

        return self.cv7(y1)


@NECKS.register_module()
class BFP(BaseModule):
    """BFP (Balanced Feature Pyramids)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    the paper `Libra R-CNN: Towards Balanced Learning for Object Detection
    <https://arxiv.org/abs/1904.02701>`_ for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=2,
                 refine_type=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(BFP, self).__init__(init_cfg)
        assert refine_type in [None, 'conv', 'non_local']

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels

        if self.refine_type == 'conv':
            self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            # self.refine = NonLocal2d(
            #     self.in_channels,
            #     reduction=1,
            #     use_scale=False,
            #     conv_cfg=self.conv_cfg,
            #     norm_cfg=self.norm_cfg)

            self.refine = nn.Sequential(
                # RFB_modified(self.in_channels, self.in_channels),
                # BasicRFB(self.in_channels, self.in_channels),
                CoordAtt(self.in_channels, self.in_channels),
                # HRAB()
                # SPPCSPC(512, self.in_channels)

            )
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_levels #Python len() 方法返回对象（字符、列表、元组等）长度或项目个数。

        # step 1: gather multi-level features by resize and average
        # 步骤1：通过调整大小和平均值收集多级特征
        # feats = []
        # gather_size = inputs[self.refine_level].size()[2:]

        #################
        feats = []
        gather_size = inputs[0].size()[2:]
        #################

        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)


        #############################

        #############################

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        # 步骤2：细化收集的特征
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        # 步骤3：通过残差路径将细化特征分散到多个层次
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])


        #####################

        #####################


        return tuple(outs)












# # Copyright (c) OpenMMLab. All rights reserved.
# import torch.nn.functional as F
# from mmcv.cnn import ConvModule
# from mmcv.cnn.bricks import NonLocal2d
# from mmcv.runner import BaseModule
#
# from ..builder import NECKS
#
# import math
# import torch
# import torch.nn as nn
#
# import torch.nn as nn
# import torch
# class SEWeightModule(nn.Module):
#
#     def __init__(self, channels, reduction=16):
#         super(SEWeightModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         out = self.avg_pool(x)
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.fc2(out)
#         weight = self.sigmoid(out)
#
#         return weight
# def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
#     """standard convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
#                      padding=padding, dilation=dilation, groups=groups, bias=False)
# class PSAModule(nn.Module):
#
#     def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
#         super(PSAModule, self).__init__()
#         self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
#                             stride=stride, groups=conv_groups[0])
#         self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
#                             stride=stride, groups=conv_groups[1])
#         self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
#                             stride=stride, groups=conv_groups[2])
#         self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
#                             stride=stride, groups=conv_groups[3])
#         self.se = SEWeightModule(planes // 4)
#         self.split_channel = planes // 4
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#         x1 = self.conv_1(x)
#         x2 = self.conv_2(x)
#         x3 = self.conv_3(x)
#         x4 = self.conv_4(x)
#
#         feats = torch.cat((x1, x2, x3, x4), dim=1)
#         feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])
#
#         x1_se = self.se(x1)
#         x2_se = self.se(x2)
#         x3_se = self.se(x3)
#         x4_se = self.se(x4)
#
#         x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
#         attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
#         attention_vectors = self.softmax(attention_vectors)
#         feats_weight = feats * attention_vectors
#         for i in range(4):
#             x_se_weight_fp = feats_weight[:, i, :, :]
#             if i == 0:
#                 out = x_se_weight_fp
#             else:
#                 out = torch.cat((x_se_weight_fp, out), 1)
#
#         return out
#
# ###############################################################################
#
#
# ###############################################################################
#
# @NECKS.register_module()
# class BFP(BaseModule):
#
#     def __init__(self,
#                  in_channels,
#                  num_levels,
#                  refine_level=2,
#                  refine_type=None,
#                  conv_cfg=None,
#                  norm_cfg=None,
#                  aspp_dilations=(1, 3, 6, 1),
#                  init_cfg=dict(
#                      type='Xavier', layer='Conv2d', distribution='uniform')):
#         super(BFP, self).__init__(init_cfg)
#         assert refine_type in [None, 'conv', 'non_local']
#
#         self.in_channels = in_channels
#         self.num_levels = num_levels
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#
#         self.refine_level = refine_level
#         self.refine_type = refine_type
#         assert 0 <= self.refine_level < self.num_levels
#
#
#
#         if self.refine_type == 'conv':
#             self.refine = ConvModule(
#                 self.in_channels,
#                 self.in_channels,
#                 3,
#                 padding=1,
#                 conv_cfg=self.conv_cfg,
#                 norm_cfg=self.norm_cfg)
#         elif self.refine_type == 'non_local':
#
#             # self.refine = NonLocal2d(
#             #     self.in_channels,
#             #     reduction=1,
#             #     use_scale=False,
#             #     conv_cfg=self.conv_cfg,
#             #     norm_cfg=self.norm_cfg)
#             # self.refine = PSAModule(self.in_channels, self.in_channels, stride=3, conv_kernels=[3, 5, 7, 9],
#             #                         conv_groups=[1, 4, 8, 16])
#             # self.refine = PSAModule(self.in_channels, self.in_channels, stride=3, conv_kernels=[1, 3, 5, 7],
#             #                         conv_groups=[1, 2, 4, 8])
#
#     def forward(self, inputs):
#         """Forward function."""
#         assert len(inputs) == self.num_levels #Python len() 方法返回对象（字符、列表、元组等）长度或项目个数。
#
#
# #######################################################################
#         feats1 = []
#         feats2 = []
#         gathered1 = 0
#         gathered2 = 0
#         gather_size1 = inputs[self.refine_level-1].size()[2:]
#         gather_size2 = inputs[self.refine_level+1].size()[2:]
#         for i in range(self.num_levels):
#             if i < self.refine_level:
#                 gathered1 = F.adaptive_max_pool2d(
#                     inputs[i], output_size=gather_size1)
#             elif i > self.refine_level:
#                 gathered2 = F.interpolate(
#                     inputs[i], size=gather_size2, mode='nearest')
#
#             feats1.append(gathered1)
#             feats2.append(gathered2)
# #######################################################################
#
#         bsf1 = sum(feats1)/len(feats1)
#         bsf2 = sum(feats2)/len(feats2)
#
#         if self.refine_type is not None:
#             bsf1 = self.refine(bsf1)
#             bsf2 = self.refine(bsf2)
# ######################################################################
#
#         outs = []
#
#         for i in range(self.num_levels):
#             out_size = inputs[i].size()[2:]
#             if i < self.refine_level:
#                 residual = F.interpolate(bsf1, size=out_size, mode='nearest')
#             else:
#                 residual = F.adaptive_max_pool2d(bsf2, output_size=out_size)
#             outs.append(residual + inputs[i])
#
#         return tuple(outs)
