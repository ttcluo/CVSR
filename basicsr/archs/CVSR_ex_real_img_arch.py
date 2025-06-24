import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from .edvr_arch import PCDAlignment, TSAFusion
from .spynet_arch import SpyNet

@ARCH_REGISTRY.register()
class CVSRexRI(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=30, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment optical flow alignment module - Uses SpyNet to calculate inter-frame motion
        self.spynet = SpyNet(spynet_path)

        # propagation === Bidirectional Propagation Branches ===
        # Backward propagation branch (processes frames in reverse order)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        # Forward propagation branch (processes frames in forward order)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # === Complex Feature Construction Module ===
        # 1x1 convolution to compress bidirectional feature channels (for imaginary part construction)
        self.compress = nn.Conv2d(2*num_feat,num_feat,1,1,0)
        # Residual block to process compressed features (imaginary part generation)
        self.resblock1 = ResidualBlockNoBN(num_feat)

        # === Complex Attention Mechanism ===
        # 3D convolution + pooling to extract complex spatial attention
        self.conv3D = nn.Sequential(
            nn.Conv3d(num_feat,num_feat,(3,3,3),(1,1,1),(1,1,1)),  # 3D spatial convolution
            nn.PReLU(), # Parametric ReLU
            nn.Conv3d(num_feat, num_feat, (2, 1, 1), (2, 1, 1), (0,0,0)), # Dimensionality reduction convolution
            nn.AdaptiveAvgPool3d(1) # Global pooling to generate attention vector
        )
        # Real part enhancement convolution
        self.convreal = nn.Conv2d(num_feat,num_feat,3,1,1,bias=True)
        # Imaginary part enhancement convolution
        self.convimg = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        # === Reconstruction Module ===
        # Feature fusion (Complex features -> Real space)
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        # First upsampling convolution
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        # Second upsampling convolution
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        # High-resolution feature refinement
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        # Final output layer
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # PixelShuffle upsampling (×2)
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

            # === Complex Feature Construction (Experimental modification: backward is real, forward is imaginary) ===
            # Real part = Backward feature - Forward feature
            real = out_l[i] - feat_prop
            # Imaginary part = Fusion(Backward + Forward) -> Compress -> Residual block
            img = self.resblock1(self.compress(torch.cat([out_l[i], feat_prop], dim=1)))

            # === Complex Attention Mechanism ===
            sreal = real.unsqueeze(2)
            simg = img.unsqueeze(2)
            newf = torch.cat([sreal,simg],dim=2)
            att = self.conv3D(newf)
            att = att.squeeze(2)
            attcos = torch.cos(att)
            attsin = torch.sin(att)
            real = real * attcos
            img = img *attsin
            attreal = real+self.convreal(real)
            attimg = img+self.convimg(img)
            out = torch.cat([attreal, attimg], dim=1)

            # === Super-resolution Reconstruction ===
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out
            lastoutput = out

        return torch.stack(out_l, dim=1)


class ConvResidualBlocks(nn.Module):
    """Convolutional Residual Blocks Module

    Contains:
    1. Initial convolutional layer
    2. Multiple residual blocks (without BN)

    Args:
        num_in_ch (int): Number of input channels
        num_out_ch (int): Number of output channels
        num_block (int): Number of residual blocks
    """
    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)

@ARCH_REGISTRY.register()
class CIconVSRexRI(nn.Module):
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
        self.edvr = EDVRFeatureExtractor(temporal_padding * 2 + 1, num_feat, edvr_path)
        self.spynet = SpyNet(spynet_path)

        # === Propagation Branches ===
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.forward_trunk = ConvResidualBlocks(2 * num_feat + 3, num_feat, num_block)

        # === Complex Domain Processing Core C.T ===
        self.compress = nn.Conv2d(2*num_feat, num_feat, 1, 1, 0)
        self.resblock1 = ResidualBlockNoBN(num_feat)

        # 3D Complex Attention Mechanism C.A
        self.conv3D = nn.Sequential(
            nn.Conv3d(num_feat, num_feat, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.PReLU(),
            nn.Conv3d(num_feat, num_feat, (2, 1, 1), (2, 1, 1), (0, 0, 0)),
            nn.AdaptiveAvgPool3d(1)
        )
        self.convreal = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.convimg = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)


        # === Reconstruction Module ===
        # Added fusion layer to process output of complex module
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)

        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def pad_spatial(self, x):
        n, t, c, h, w = x.size()
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
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
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
        return feats_keyframe

    def forward(self, x):
        b, n, _, h_input, w_input = x.size()
        x = self.pad_spatial(x)
        h, w = x.shape[3:]
        keyframe_idx = list(range(0, n, self.keyframe_stride))
        if keyframe_idx[-1] != n - 1:
            keyframe_idx.append(n - 1)
        flows_forward, flows_backward = self.get_flow(x)
        feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.backward_fusion(feat_prop)
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
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.forward_fusion(feat_prop)
            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # === Complex Domain Processing C.T (Experimental modification: backward is real, forward is imaginary) ===
            # Real part = Backward feature - Forward feature
            real = out_l[i] - feat_prop
            # Imaginary part = Fusion(Backward + Forward) -> Compress -> Residual block
            img = self.resblock1(self.compress(torch.cat([out_l[i], feat_prop], dim=1)))

            # === C.A ===
            sreal = real.unsqueeze(2)
            simg = img.unsqueeze(2)
            newf = torch.cat([sreal,simg],dim=2)
            att = self.conv3D(newf)
            att = att.squeeze(2)
            attcos = torch.cos(att)
            attsin = torch.sin(att)
            real = real * attcos
            img = img *attsin
            attreal = real+self.convreal(real)
            attimg = img+self.convimg(img)
            out = torch.cat([attreal, attimg], dim=1)


            # === Upsampling ===
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)[..., :4 * h_input, :4 * w_input]


class EDVRFeatureExtractor(nn.Module):
    def __init__(self, num_input_frame, num_feat, load_path):
        super(EDVRFeatureExtractor, self).__init__()
        self.center_frame_idx = num_input_frame // 2
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=64)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=8)
        self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_input_frame, center_frame_idx=self.center_frame_idx)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

    def forward(self, x):
        b, n, c, h, w = x.size()
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))
        feat_l1 = feat_l1.view(b, n, -1, h, w)
        feat_l2 = feat_l2.view(b, n, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, n, -1, h // 4, w // 4)
        ref_feat_l = [
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(n):
            nbr_feat_l = [
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)
        return self.fusion(aligned_feat)