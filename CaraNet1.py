# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:58:14 2021

@author: angelou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pretrain.Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
import math
import torchvision.models as models
from lib.conv_layer import Conv, BNPReLU
from lib.axial_atten import AA_kernel
from lib.context_module import CFPModule
from lib.partial_decoder import aggregation
import os
from lib.pvtv2 import pvt_v2_b2
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d
import warnings

 
    

    
class caranet(nn.Module):
    def __init__(self, channel=32):
        super().__init__()
        
         # ---- ResNet Backbone ----
        self.resnet = res2net101_v1b_26w_4s(pretrained=True)
        # self.rfb2_1 = Conv(512, 32,3,1,padding=1,bn_acti=True)
        # self.rfb3_1 = Conv(1024, 32,3,1,padding=1,bn_acti=True)
        # self.rfb4_1 = Conv(2048, 32,3,1,padding=1,bn_acti=True)
        # Receptive Field Block
        # self.rfb2_1 = Conv(512, 32,3,1,padding=1,bn_acti=True)
        # self.rfb3_1 = Conv(1024, 32,3,1,padding=1,bn_acti=True)
        # self.rfb4_1 = Conv(2048, 32,3,1,padding=1,bn_acti=True)
        self.rfb2_1 = Conv(128, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb3_1 = Conv(320, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb4_1 = Conv(512, 32, 3, 1, padding=1, bn_acti=True)

        # Partial Decoder
        self.agg1 = aggregation(channel)
        self.agg2 = aggregation(channel)
        
        self.CFP_1 = CFPModule(32, d=8)
        self.CFP_2 = CFPModule(32, d=8)
        self.CFP_3 = CFPModule(32, d=8)
        # self.CFP_1 = CFPModule(256, d=8)
        # self.CFP_2 = CFPModule(256, d=8)
        # self.CFP_3 = CFPModule(256, d=8)
        ###### dilation rate 4, 62.8


        self.ra1_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra2_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra3_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
        
        # self.aa_kernel_1 = AA_kernel(32,32)
        # self.aa_kernel_2 = AA_kernel(32,32)
        # self.aa_kernel_3 = AA_kernel(32,32)

        self.aa_kernel_1 = PSA_p(32, 32)
        self.aa_kernel_2 = PSA_p(32, 32)
        self.aa_kernel_3 = PSA_p(32, 32)

        # self.aa_kernel_1 = PSA_p(32, 32)
        # self.aa_kernel_2 = PSA_p(32, 32)
        # self.aa_kernel_3 = PSA_p(32, 32)

        # self.attention2 = UACA(256, 256)
        # self.attention3 = UACA(256, 256)
        # self.attention4 = UACA(256, 256)
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

    #
    #     self.num_classes = 1
    #
    #     c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = 64, 128, 320, 512
    #     embedding_dim = 256
    #
    #     self.linear_c4 = conv1(input_dim=c4_in_channels, embed_dim=embedding_dim)
    #     self.linear_c3 = conv1(input_dim=c3_in_channels, embed_dim=embedding_dim)
    #     self.linear_c2 = conv1(input_dim=c2_in_channels, embed_dim=embedding_dim)
    #     self.linear_c1 = conv1(input_dim=c1_in_channels, embed_dim=embedding_dim)
    #
    #     self.linear_fuse = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,
    #                                   norm_cfg=dict(type='BN', requires_grad=True))
    #     self.linear_fuse34 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim,
    #                                     kernel_size=1, norm_cfg=dict(type='BN', requires_grad=True))
    #     self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
    #                                    norm_cfg=dict(type='BN', requires_grad=True))
    #     self.linear_fuse1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
    #                                    norm_cfg=dict(type='BN', requires_grad=True))
    #
    #     self.linear_pred = Conv2d(embedding_dim, self.num_classes, kernel_size=1)
    #     self.dropout = nn.Dropout(0.1)
    #
    # def forward(self, x):
    #     pvt = self.backbone(x)
    #     c1 = pvt[0]  # b,64,88,88
    #     c2 = pvt[1]
    #     c3 = pvt[2]
    #     c4 = pvt[3]
    #     # c1, c2, c3, c4 = inputs
    #     ############## MLP decoder on C1-C4 ###########
    #     n, _, h, w = c4.shape
    #
    #     _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
    #     _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
    #     _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
    #     _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
    #     _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
    #     _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
    #     _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
    #
    #     L34 = self.linear_fuse34(torch.cat([_c4, _c3], dim=1))
    #     L2 = self.linear_fuse2(torch.cat([L34, _c2], dim=1))
    #     _c = self.linear_fuse1(torch.cat([L2, _c1], dim=1))
    #
    #     x = self.dropout(_c)
    #     x = self.linear_pred(x)
    #     up = UpsamplingBilinear2d(scale_factor=4)
    #     x = up(x)
    #     # return features
    #
    #     return x



    def forward(self, x):

        # x = self.resnet.conv1(x)
        # x = self.resnet.bn1(x)
        # x = self.resnet.relu(x)
        # x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        #
        # # ----------- low-level features -------------
        #
        # x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        # x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        #
        # x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        # x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11
        pvt = self.backbone(x)
        x1 = pvt[0]  # b,64,88,88
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x2_rfb = self.rfb2_1(x2) # 512 - 32
        x3_rfb = self.rfb3_1(x3) # 1024 - 32
        x4_rfb = self.rfb4_1(x4) # 2048 - 32

        decoder_1 = self.agg1(x4_rfb, x3_rfb, x2_rfb) # 1,44,44
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=8, mode='bilinear', align_corners=False)

        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.25, mode='bilinear', align_corners=False)
        cfp_out_1 = self.CFP_3(x4_rfb) # 32 - 32
        # decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1, decoder_2)
        # aa_atten_3_o = decoder_2_ra.expand(-1, 32, -1, -1).mul(aa_atten_3)

        ra_3 = self.ra3_conv1(aa_atten_3) # 32 - 32
        ra_31 = self.ra3_conv2(ra_3) # 32 - 32
        ra_3 = self.ra3_conv3(ra_31) # 32 - 1
        ra_3 = F.interpolate(ra_3, scale_factor=4, mode='bilinear', align_corners=False)
        # x_3 = ra_3 + decoder_2
        x_3 = ra_3 + decoder_1
        lateral_map_2 = F.interpolate(x_3, scale_factor=8, mode='bilinear', align_corners=False)

        # ------------------- atten-two -----------------------
        decoder_3 = F.interpolate(x_3, scale_factor=0.5, mode='bilinear', align_corners=False)
        cfp_out_2 = self.CFP_2(x3_rfb) # 32 - 32
        # decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        # decoder_3_ra = torch.sigmoid(F.avg_pool2d(decoder_3, kernel_size=31, stride=1, padding=15) - decoder_3)
        # decoder_3_ra = (torch.sigmoid(decoder_3))
        aa_atten_2 = self.aa_kernel_2(cfp_out_2, decoder_3)
        # aa_atten_2_o = decoder_3_ra.expand(-1, 32, -1, -1).mul(aa_atten_2)

        ra_2 = self.ra2_conv1(aa_atten_2) # 32 - 32
        ra_21 = self.ra2_conv2(ra_2) # 32 - 32
        ra_2 = self.ra2_conv3(ra_21) # 32 - 1
        ra_2 = F.interpolate(ra_2, scale_factor=2, mode='bilinear', align_corners=False)
        x_2 = ra_2 + x_3
        # x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2, scale_factor=8, mode='bilinear', align_corners=False)

        # ------------------- atten-three -----------------------
        # decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear', align_corners=True)
        cfp_out_3 = self.CFP_1(x2_rfb) # 32 - 32
        # decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        # decoder_4_ra = -1 * (torch.sigmoid(x_2)) + 1
        # aa_atten_1 = self.aa_kernel_1(cfp_out_3, decoder_4)
        aa_atten_1 = self.aa_kernel_1(cfp_out_3, x_2)
        # aa_atten_1_o = decoder_4_ra.expand(-1, 32, -1, -1).mul(aa_atten_1)

        ra_1 = self.ra1_conv1(aa_atten_1) # 32 - 32
        ra_11 = self.ra1_conv2(ra_1) # 32 - 32
        ra_1 = self.ra1_conv3(ra_11) # 32 - 1

        x_1 = ra_1 + x_2
        # x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1, scale_factor=8, mode='bilinear', align_corners=False)
        x_0 = self.agg2(ra_31, ra_21, ra_11)  # 1,44,44
        lateral_map_6 = F.interpolate(x_0, scale_factor=8, mode='bilinear', align_corners=False)
        # lateral_map_6 = F.interpolate(decoder_, scale_factor=8, mode='bilinear')



        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1,lateral_map_6
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class UACA(nn.Module):
    def __init__(self, in_channel, channel):
        super(UACA, self).__init__()
        self.channel = channel

        self.conv_query = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        self.conv_key = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                      conv(channel, channel, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                        conv(channel, channel, 1, relu=True))

        self.conv_out1 = conv(channel, channel, 3, relu=True)
        self.conv_out2 = conv(in_channel + channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        self.conv_out4 = conv(channel, 1, 1)

    def forward(self, x, map):
        b, c, h, w = x.shape

        # compute class probability
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        fg = torch.sigmoid(map)

        p = fg - .5

        fg = torch.clip(p, 0, 1)  # foreground
        bg = torch.clip(-p, 0, 1)  # background
        cg = .5 - torch.abs(p)  # confusion area

        prob = torch.cat([fg, bg, cg], dim=1)

        # reshape feature & prob
        f = x.view(b, h * w, -1)
        prob = prob.view(b, 3, h * w)

        # compute context vector
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3)  # b, 3, c

        # k q v compute
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.channel, -1)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key)  # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)
        out = out + map
        return x, out

class conv1(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same',
                 bias=False, bn=True, relu=False):
        super(conv, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class PSA_p(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_p, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()


        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)
        context_mask_self = context_mask
        # context_mask = -1*(torch.sigmoid(map)) + 1
        # context_mask = map
        # context_mask = context_mask.view(batch, height * width, 1)
        # map = map.view(batch, 1, height * width)
        # sim = torch.matmul(context_mask, map)
        # sim = F.softmax(sim, dim=-1)
        # context_mask = torch.matmul(sim, context_mask)
        # context_mask = context_mask.view(batch, 1, height, width)
        # context_mask = context_mask + context_mask_self
        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1,2))
        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x, map):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x)
        # map = -1 * (torch.sigmoid(map)) + 1
        theta_x = theta_x * map
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        theta_x = theta_x.view(batch, self.inter_planes, height * width)
        # print(theta_x.shape)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)
        # [N, 1, H*W]
        context = self.softmax_left(context)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)
        mask_sp_self = mask_sp

        mask_sp = mask_sp.view(batch, height * width, 1)
        map = map.view(batch, 1, height * width)
        sim = torch.matmul(mask_sp, map)
        sim = F.softmax(sim, dim=-1)
        mask_sp = torch.matmul(sim, mask_sp)
        mask_sp = mask_sp.view(batch, 1, height, width)
        mask_sp = mask_sp + mask_sp_self
        mask_sp = self.sigmoid(mask_sp)

        out = x * mask_sp

        return out

    def forward(self, x, map):
        # [N, C, H, W]
        context_channel = self.spatial_pool(x)
        # [N, C, H, W]
        context_spatial = self.channel_pool(x, map)
        # [N, C, H, W]
        out = context_spatial + context_channel
        # out = context_channel
        return out

class PSA_s(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_s, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 4

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        # self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))

        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        out = self.spatial_pool(x)

        # [N, C, H, W]
        out = self.channel_pool(out)

        # [N, C, H, W]
        # out = context_spatial + context_channel

        return out

# class PSA_p(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size=1, stride=1):
#         super(PSA_p, self).__init__()
#
#         self.inplanes = inplanes
#         self.inter_planes = planes // 2
#         self.planes = planes
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = (kernel_size-1)//2
#
#         self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
#         self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
#         self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
#         self.softmax_right = nn.Softmax(dim=2)
#         self.sigmoid = nn.Sigmoid()
#
#         self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
#         self.softmax_left = nn.Softmax(dim=2)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         kaiming_init(self.conv_q_right, mode='fan_in')
#         kaiming_init(self.conv_v_right, mode='fan_in')
#         kaiming_init(self.conv_q_left, mode='fan_in')
#         kaiming_init(self.conv_v_left, mode='fan_in')
#
#         self.conv_q_right.inited = True
#         self.conv_v_right.inited = True
#         self.conv_q_left.inited = True
#         self.conv_v_left.inited = True
#
#     def spatial_pool(self, x):
#         input_x = self.conv_v_right(x)
#
#         batch, channel, height, width = input_x.size()
#
#         # [N, IC, H*W]
#         input_x = input_x.view(batch, channel, height * width)
#
#         # [N, 1, H, W]
#         context_mask = self.conv_q_right(x)
#
#         # [N, 1, H*W]
#         context_mask = context_mask.view(batch, 1, height * width)
#
#         # [N, 1, H*W]
#         context_mask = self.softmax_right(context_mask)
#
#         # [N, IC, 1]
#         # context = torch.einsum('ndw,new->nde', input_x, context_mask)
#         context = torch.matmul(input_x, context_mask.transpose(1,2))
#         # [N, IC, 1, 1]
#         context = context.unsqueeze(-1)
#
#         # [N, OC, 1, 1]
#         context = self.conv_up(context)
#
#         # [N, OC, 1, 1]
#         mask_ch = self.sigmoid(context)
#
#         out = x * mask_ch
#
#         return out
#
#     def channel_pool(self, x):
#         # [N, IC, H, W]
#         g_x = self.conv_q_left(x)
#
#         batch, channel, height, width = g_x.size()
#
#         # [N, IC, 1, 1]
#         avg_x = self.avg_pool(g_x)
#
#         batch, channel, avg_x_h, avg_x_w = avg_x.size()
#
#         # [N, 1, IC]
#         avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)
#
#         # [N, IC, H*W]
#         theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
#
#         # [N, 1, H*W]
#         # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
#         context = torch.matmul(avg_x, theta_x)
#         # [N, 1, H*W]
#         context = self.softmax_left(context)
#
#         # [N, 1, H, W]
#         context = context.view(batch, 1, height, width)
#
#         # [N, 1, H, W]
#         mask_sp = self.sigmoid(context)
#
#         out = x * mask_sp
#
#         return out
#
#     def forward(self, x):
#         # [N, C, H, W]
#         context_channel = self.spatial_pool(x)
#         # [N, C, H, W]
#         context_spatial = self.channel_pool(x)
#         # [N, C, H, W]
#         out = context_spatial + context_channel
#         return out

if __name__ == '__main__':
    ras = caranet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor)