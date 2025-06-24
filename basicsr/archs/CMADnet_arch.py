import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
# from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
# from .edvr_arch import PCDAlignment, TSAFusion
# from .spynet_arch import SpyNet
from .channel_diversity import *

from basicsr.archs.arch_util import ResidualBlockNoBN, flow_warp, make_layer
from basicsr.archs.edvr_arch import PCDAlignment, TSAFusion
from basicsr.archs.spynet_arch import SpyNet
from thop import profile
from basicsr.utils import get_root_logger

#CMADNet 添加C.T和C.A模块
class SimpleGate(nn.Module):
    """简单门控机制：将输入通道分成两半并逐元素相乘"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class KBAFunction(torch.autograd.Function):
    '''
    mainly borrows from Kernel bias attention
    核偏置注意力(Kernel Bias Attention)的自定义函数，支持自动微分
    '''

    @staticmethod
    def forward(ctx, x, att, selfk, selfg, selfb, selfw):
        B, nset, H, W = att.shape
        KK = selfk ** 2
        selfc = x.shape[1]

        att = att.reshape(B, nset, H * W).transpose(-2, -1)

        ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset = selfk, selfg, selfc, KK, nset
        ctx.x, ctx.att, ctx.selfb, ctx.selfw = x, att, selfb, selfw

        bias = att @ selfb
        attk = att @ selfw

        uf = torch.nn.functional.unfold(x, kernel_size=selfk, padding=selfk // 2)

        # for unfold att / less memory cost
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        x = attk @ uf.unsqueeze(-1)  #
        del attk, uf
        x = x.squeeze(-1).reshape(B, H * W, selfc) + bias
        x = x.transpose(-1, -2).reshape(B, selfc, H, W)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, att, selfb, selfw = ctx.x, ctx.att, ctx.selfb, ctx.selfw
        selfk, selfg, selfc, KK, nset = ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset

        B, selfc, H, W = grad_output.size()

        dbias = grad_output.reshape(B, selfc, H * W).transpose(-1, -2)

        dselfb = att.transpose(-2, -1) @ dbias
        datt = dbias @ selfb.transpose(-2, -1)

        attk = att @ selfw
        uf = F.unfold(x, kernel_size=selfk, padding=selfk // 2)
        # for unfold att / less memory cost
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        dx = dbias.view(B, H * W, selfg, selfc // selfg, 1)

        dattk = dx @ uf.view(B, H * W, selfg, 1, selfc // selfg * KK)
        duf = attk.transpose(-2, -1) @ dx
        del attk, uf

        dattk = dattk.view(B, H * W, -1)
        datt += dattk @ selfw.transpose(-2, -1)
        dselfw = att.transpose(-2, -1) @ dattk

        duf = duf.permute(0, 2, 3, 4, 1).view(B, -1, H * W)
        dx = F.fold(duf, output_size=(H, W), kernel_size=selfk, padding=selfk // 2)

        datt = datt.transpose(-1, -2).view(B, nset, H, W)

        return dx, datt, None, None, dselfb, dselfw

# @ARCH_REGISTRY.register()
class BasicVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=15, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward, flows_backward = self.get_flow(x)
        b, n, _, h, w = x.size()

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)


@ARCH_REGISTRY.register()
class CMADIconVSR(nn.Module):
    """ MADNet mainly borrows from IconVSR.
        CMADNet add C.T and C.A
    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15.
        keyframe_stride (int): Keyframe stride. Default: 5.
        temporal_padding (int): Temporal padding. Default: 2.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        edvr_path (str): Path to the pretrained EDVR model. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_block=15,
                 keyframe_stride=5,
                 temporal_padding=2,
                 spynet_path=None,
                 edvr_path=None):
        super().__init__()

        self.num_feat = num_feat
        self.temporal_padding = temporal_padding
        self.keyframe_stride = keyframe_stride

        # keyframe_branch 关键帧特征提取分支 (使用EDVR)
        self.edvr = EDVRFeatureExtractor(temporal_padding * 2 + 1, num_feat, edvr_path)
        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        # ========== 后向传播分支 ==========
        self.backward_conv = ConvResidualBlocks(3, num_feat, 1)  # additional module 额外卷积模块
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True) # 特征融合
        self.backward_trunk = ConvResidualBlocks(num_feat*2, num_feat, num_block) # 主干网络

        # ========== 前向传播分支 ==========
        self.forward_conv = ConvResidualBlocks(3, num_feat, 1)  # # additional module 额外卷积模块
        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.forward_trunk = ConvResidualBlocks(3 * num_feat, num_feat, num_block)

        # === 复数域处理核心 C.T===
        # 特征压缩：双向特征拼接后降维
        self.compress = nn.Conv2d(2*num_feat, num_feat, 1, 1, 0)
        # 虚部生成：残差块处理压缩特征
        self.resblock1 = ResidualBlockNoBN(num_feat)

        # 3D复数注意力机制 C.A
        self.conv3D = nn.Sequential(
            nn.Conv3d(num_feat, num_feat, (3, 3, 3), (1, 1, 1), (1, 1, 1)),  # 3D空间-时间卷积
            nn.PReLU(),  # 参数化ReLU激活
            nn.Conv3d(num_feat, num_feat, (2, 1, 1), (2, 1, 1), (0, 0, 0)),  # 降维卷积
            nn.AdaptiveAvgPool3d(1)  # 全局池化得到注意力向量
        )
        # 实部增强卷积
        self.convreal = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        # 虚部增强卷积
        self.convimg = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        # reconstruction 重建模块
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True) # 上采样卷积1
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True) # 上采样卷积2
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1) # 高分辨率卷积
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1) # 最终输出卷积

        # 像素洗牌上采样
        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # multi-axis diversity enhancement block 多轴多样性增强模块
        self.made = MADE()

    def pad_spatial(self, x):
        """Apply padding spatially.

        Since the PCD module in EDVR requires that the resolution is a multiple
        of 4, we apply padding to the input LR images if their resolution is
        not divisible by 4.

        Args:
            x (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        n, t, c, h, w = x.size()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        x = x.view(-1, c, h, w)
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        return x.view(n, t, c, h + pad_h, w + pad_w)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def get_keyframe_feature(self, x, keyframe_idx):
        if self.temporal_padding == 2:
            x = [x[:, [4, 3]], x, x[:, [-4, -5]]]
        elif self.temporal_padding == 3:
            x = [x[:, [6, 5, 4]], x, x[:, [-5, -6, -7]]]
        x = torch.cat(x, dim=1)

        num_frames = 2 * self.temporal_padding + 1
        feats_keyframe = {}
        for i in keyframe_idx:
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())  # [b 64 h w]
        return feats_keyframe

    def forward(self, x):
        b, n, _, h_input, w_input = x.size()  # b 15 3 h w

        x = self.pad_spatial(x)
        h, w = x.shape[3:]

        keyframe_idx = list(range(0, n, self.keyframe_stride))  # [0 5 10]
        if keyframe_idx[-1] != n - 1:
            keyframe_idx.append(n - 1)  # last frame is a keyframe  [0 5 10 14]

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x)
        feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)

        # backward branch 后向传播分支
        out_l = [] # 存储输出特征
        feat_prop = x.new_zeros(b, self.num_feat, h, w)  # [b 64 h w]的零矩阵 初始化特征传播
        for i in range(n - 1, -1, -1):  # 14， 13， ···，
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.backward_fusion(feat_prop)

            # enhanced by MADE  应用多轴多样性增强
            feat_prop = torch.cat([self.made(self.backward_conv(x_i)), feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop) # 在列表开头插入结果

        # forward branch 前向传播分支
        feat_prop = torch.zeros_like(feat_prop) # 重置特征传播
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.forward_fusion(feat_prop)

            # enhanced by MADE 应用多轴多样性增强并融合后向分支特征
            feat_prop = torch.cat([self.made(self.forward_conv(x_i)), out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # === 复数域处理（同CVSR）===
            real = feat_prop - out_l[i]
            img = self.resblock1(self.compress(torch.cat([feat_prop, out_l[i]], dim=1)))
            sreal = real.unsqueeze(2)
            simg = img.unsqueeze(2)
            newf = torch.cat([sreal,simg],dim=2)
            att = self.conv3D(newf)
            att = att.squeeze(2)
            # 相位旋转
            attcos = torch.cos(att)
            attsin = torch.sin(att)
            real = real * attcos
            img = img *attsin
            # 特征增强
            attreal = real+self.convreal(real)
            attimg = img+self.convimg(img)
            out = torch.cat([attreal, attimg], dim=1)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)[..., :4 * h_input, :4 * w_input]


class EDVRFeatureExtractor(nn.Module):
    """EDVR feature extractor used in IconVSR.

    Args:
        num_input_frame (int): Number of input frames.
        num_feat (int): Number of feature channels
        load_path (str): Path to the pretrained weights of EDVR. Default: None.
    """

    def __init__(self, num_input_frame, num_feat, load_path):

        super(EDVRFeatureExtractor, self).__init__()

        self.center_frame_idx = num_input_frame // 2

        # extract pyramid features
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd and tsa module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=8)
        self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_input_frame, center_frame_idx=self.center_frame_idx)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

    def forward(self, x):
        b, n, c, h, w = x.size()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, n, -1, h, w)
        feat_l2 = feat_l2.view(b, n, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, n, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(n):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        # TSA fusion
        return self.fusion(aligned_feat)

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

def SDE(x, att, selfk, selfg, selfb, selfw):
    """ 空间多样性增强的包装函数 """
    return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)

class MADE(nn.Module):
    def __init__(self, c=64, DW_Expand=2, FFN_Expand=2, nset=64, k=3, gc=4, lightweight=False):
        super(MADE, self).__init__()
        self.k = k  # 卷积核大小
        self.g = c // gc # 分组数
        # 可学习参数
        self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k ** 2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))

        # 主干卷积
        self.dwconv_k5 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding=2, stride=1, groups=1, bias=True)
        # 通道多样性模块
        self.channel_diversity = MultiSpectralAttentionLayer(channel=c, dct_w=56, dct_h=56)
        # 辅助路径卷积
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv21 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c, bias=True)

        # 空间多样性路径
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=64, bias=True),
        )
        # 注意力生成路径
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, padding=1, stride=1, groups=32, bias=True),
            SimpleGate(), # 门控机制
            nn.Conv2d(32 // 2, 64, 1, padding=0, stride=1), # 通道减半后再恢复
        )
        self.conv211 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        # 可学习缩放因子
        self.attgamma = nn.Parameter(torch.zeros((1, 64, 1, 1)) + 1e-2, requires_grad=True)
        self.ga1 = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        # 特征融合层
        self.fusion = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, inp):
        x = inp
        x = self.dwconv_k5(x)

        # Three branches 三路特征增强
        # 1 通道多样性增强
        ca = self.channel_diversity(x)  # CDE

        # 2 辅助路径特征
        x1 = self.conv11(x)  # Aux

        # 3 空间多样性增强
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv21(self.conv1(x))
        sa = SDE(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf  # SDE

        # aggregation 特征聚合
        x = ca + x1 + sa  # 三路特征相加
        x = self.fusion(x)  # [b 64 h w]  1x1卷积融合

        # 门控输出
        out = x * inp
        return out

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %f M' % (num_params / 1e6))

if __name__ == '__main__':
    input = torch.rand(1, 15, 3, 160, 160).cuda()  # B T C H W
    model = CMADIconVSR().cuda()

    # FLOPs and Parameters
    flops, params = profile(model, inputs=(input,))
    logger = get_root_logger()
    logger.info("Param: {} M".format(params/1e6))
    logger.info("FLOPs: {} G".format(flops/1e9))
    print("Param: {} M".format(params/1e6))
    print("FLOPs: {} G".format(flops/1e9))