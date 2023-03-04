

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
from lib.pvtv2 import pvt_v2_b2
import os


class glgformer(nn.Module):
    def __init__(self, channel=32):
        super().__init__()

        # ---- ResNet Backbone ----
        # self.resnet = res2net101_v1b_26w_4s(pretrained=True)

        # Receptive Field Block
        self.rfb2_1 = Conv(128, channel, 3, 1, padding=1, bn_acti=True)
        self.rfb3_1 = Conv(320, channel, 3, 1, padding=1, bn_acti=True)
        self.rfb4_1 = Conv(512, channel, 3, 1, padding=1, bn_acti=True)

        self.rfb2_11 = Conv(128, channel, 3, 1, padding=1, bn_acti=True)
        self.rfb3_11 = Conv(320, channel, 3, 1, padding=1, bn_acti=True)
        self.rfb4_11 = Conv(512, channel, 3, 1, padding=1, bn_acti=True)

        # self.rfb2_1 = Conv(512, 256, 3, 1, padding=1, bn_acti=True)
        # self.rfb3_1 = Conv(1024, 256, 3, 1, padding=1, bn_acti=True)
        # self.rfb4_1 = Conv(2048, 256, 3, 1, padding=1, bn_acti=True)

        # Partial Decoder
        self.agg1 = aggregation(channel)
        self.agg0 = aggregation(channel)

        # self.CFP_1 = CFPModule(channel, d=8)
        # self.CFP_2 = CFPModule(channel, d=8)
        # self.CFP_3 = CFPModule(channel, d=8)
        # self.CFP_1 = CFPModule(256, d=8)
        # self.CFP_2 = CFPModule(256, d=8)
        # self.CFP_3 = CFPModule(256, d=8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(channel, channel, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(channel, channel, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(channel, 1, 3, 1, padding=1, bn_acti=True)

        self.ra2_conv1 = Conv(channel, channel, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(channel, channel, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(channel, 1, 3, 1, padding=1, bn_acti=True)

        self.ra3_conv1 = Conv(channel, channel, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv(channel, channel, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv(channel, 1, 3, 1, padding=1, bn_acti=True)

        # self.aa_kernel_1 = AA_kernel(32,32)
        # self.aa_kernel_2 = AA_kernel(32,32)
        # self.aa_kernel_3 = AA_kernel(32,32)
        # self.aa_1 = AA_()
        # self.aa_2 = AA_()
        # self.aa_3 = AA_()

        self.aa_kernel_1 = PSA_p(32, 32)
        self.aa_kernel_2 = PSA_p(32, 32)
        self.aa_kernel_3 = PSA_p(32, 32)
        # self.aa_kernel_1 = AA_kernel(256, 256)
        # self.aa_kernel_2 = AA_kernel(256, 256)
        # self.aa_kernel_3 = AA_kernel(256, 256)
        self.lp_4 = LP(input_dim=channel, embed_dim=channel)
        self.lp_3 = LP(input_dim=channel, embed_dim=channel)
        self.lp_2 = LP(input_dim=channel, embed_dim=channel)
        self.lp_1 = LP(input_dim=channel, embed_dim=channel)
        # self.attention2 = UACA(32, 32)
        # self.attention3 = UACA(32, 32)
        # self.attention4 = UACA(32, 32)
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

    def forward(self, x):

        pvt = self.backbone(x)
        x1 = pvt[0]  # b,64,88,88
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x2_rfb = self.rfb2_1(x2)  # 512 - 32
        x3_rfb = self.rfb3_1(x3)  # 1024 - 32
        x4_rfb = self.rfb4_1(x4)  # 2048 - 32

        decoder_1 = self.agg1(x4_rfb, x3_rfb, x2_rfb)  # 1,44,44
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=8, mode='bilinear')

        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.25, mode='bilinear')  # b,c,11,11
        # cfp_out_1 = self.CFP_3(x4_rfb) # 32 - 32 b,c,11,11
        # decoder_2_ra = torch.sigmoid(F.avg_pool2d(decoder_2, kernel_size=31, stride=1, padding=15) - decoder_2)
        # decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        # decoder_2_ra = (torch.sigmoid(decoder_2))
        # decoder_2_ra = decoder_2_ra * self.aa_1(decoder_2_ra)
        # aa_atten_3 = self.aa_kernel_3(x4_rfb, decoder_2)
        aa_atten_3 = self.aa_kernel_3(x4_rfb, decoder_2)
        # aa_atten_3 = self.aa_kernel_3(x4_rfb)

        # _, ra_3 = self.attention4(aa_atten_3, decoder_2)
        # aa_atten_3 = decoder_2_ra.expand(-1, 32, -1, -1).mul(aa_atten_3)
        # x2 = self.lp_2(x2)
        # ra_31 = self.lp_3(aa_atten_3)
        # x4 = self.lp_4(x4)
        # aa_atten_3 = self.rfb4_11(aa_atten_3)
        ra_3 = self.ra3_conv1(aa_atten_3)  # 32 - 32
        ra_31 = self.ra3_conv2(ra_3)  # 32 - 32
        ra_31 = self.lp_3(ra_31)
        # ra_31 = ra_31 + aa_atten_3
        ra_3 = self.ra3_conv3(ra_31)  # 32 -
        # ra_3 = self.ra3_conv3(aa_atten_3)

        # ra_3 = F.interpolate(ra_3, scale_factor=4, mode='bilinear')
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3, scale_factor=32, mode='bilinear')

        # ------------------- atten-two -----------------------
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        # cfp_out_2 = self.CFP_2(x3_rfb) # 32 - 32
        # decoder_3_ra = torch.sigmoid(F.avg_pool2d(decoder_3, kernel_size=31, stride=1, padding=15) - decoder_3)
        # decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(x3_rfb, decoder_3)
        # aa_atten_2 = self.aa_kernel_2(x3_rfb)
        # _, ra_2 = self.attention4(aa_atten_2, decoder_3)
        # aa_atten_2 = decoder_3_ra.expand(-1, 32, -1, -1).mul(aa_atten_2)
        # ra_21 = self.lp_2(aa_atten_2)
        # aa_atten_2 = self.rfb3_11(aa_atten_2)
        ra_2 = self.ra2_conv1(aa_atten_2)  # 32 - 32
        ra_21 = self.ra2_conv2(ra_2)  # 32 - 32
        ra_21 = self.lp_2(ra_21)
        # ra_21 = ra_21 + aa_atten_2
        ra_2 = self.ra2_conv3(ra_21)  # 32 -1
        # ra_2 = self.ra2_conv3(aa_atten_2)

        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2, scale_factor=16, mode='bilinear')

        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        # cfp_out_3 = self.CFP_1(x2_rfb) # 32 - 32

        # decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        # decoder_4_ra = torch.sigmoid(F.avg_pool2d(decoder_4, kernel_size=31, stride=1, padding=15) - decoder_4)
        aa_atten_1 = self.aa_kernel_1(x2_rfb, decoder_4)
        # aa_atten_1 = self.aa_kernel_1(x2_rfb)
        # _, ra_1 = self.attention4(aa_atten_1, decoder_4)
        # aa_atten_1 = decoder_4_ra.expand(-1, 32, -1, -1).mul(aa_atten_1)
        # ra_11 = self.lp_1(aa_atten_1)
        # aa_atten_1= self.rfb2_11(aa_atten_1)
        ra_1 = self.ra1_conv1(aa_atten_1)  # 32 - 32
        ra_11 = self.ra1_conv2(ra_1)  # 32 - 32
        ra_11 = self.lp_1(ra_11)
        # ra_11 = ra_11 + aa_atten_1
        ra_1 = self.ra1_conv3(ra_11)  # 32 - 1
        # ra_1 = self.ra1_conv3(aa_atten_1)

        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1, scale_factor=8, mode='bilinear')
        decoder_ = self.agg0(ra_31, ra_21, ra_11)  # 1,44,44
        # decoder_ = self.agg0(aa_atten_3, aa_atten_2, aa_atten_1)
        lateral_map_6 = F.interpolate(decoder_, scale_factor=8, mode='bilinear')

        return lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1, lateral_map_6


class LP(nn.Module):
    """
    Linear Prediction
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


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
        self.padding = (kernel_size - 1) // 2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
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
        # context_mask_self = context_mask
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
        context = torch.matmul(input_x, context_mask.transpose(1, 2))
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
        map = (torch.sigmoid(map))
        theta_x = theta_x * map
        # theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        theta_x = theta_x.view(batch, self.inter_planes, height * width)
        # print(theta_x.shape)
        theta_x = self.softmax_left(theta_x)
        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)
        # [N, 1, H*W]
        # context = self.softmax_left(context)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)
        # mask_sp =mask_sp * map
        # mask_sp_self = mask_sp
        #
        # mask_sp = mask_sp.view(batch, height * width, 1)
        # map = map.view(batch, 1, height * width)
        # sim = torch.matmul(mask_sp, map)
        # sim = F.softmax(sim, dim=-1)
        # mask_sp = torch.matmul(sim, mask_sp)
        # mask_sp = mask_sp.view(batch, 1, height, width)
        # mask_sp = mask_sp + mask_sp_self
        # mask_sp = self.sigmoid(mask_sp)

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


class AA_(nn.Module):
    def __init__(self):
        super(AA_, self).__init__()
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_x = x

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = x

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

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return mask_ch


if __name__ == '__main__':
    ras = glgformer().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor)