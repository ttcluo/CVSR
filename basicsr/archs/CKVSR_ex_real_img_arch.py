import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from .edvr_arch import PCDAlignment, TSAFusion
from .spynet_arch import SpyNet

@ARCH_REGISTRY.register()
class CKVSRexRI(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=30, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment module - uses SpyNet to calculate inter-frame motion
        self.spynet = SpyNet(spynet_path)

        # propagation === Bidirectional Propagation Branches ===
        # Backward propagation branch (processes frames in reverse order)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        # Forward propagation branch (processes frames in forward order)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # === Projection Head for Feature Space Separation ===
        # Used to project forward features into a new space
        self.forward_projection = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlockNoBN(num_feat) # Adds some non-linearity and expressiveness
        )
        # Used to project backward features into a new space
        self.backward_projection = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlockNoBN(num_feat) # Adds some non-linearity and expressiveness
        )

        # === Complex Feature Construction Module ===
        # 1x1 convolution to compress bidirectional feature channels (for constructing imaginary part)
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
        # Feature fusion (complex features -> real space)
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        # First upsampling convolution
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        # Second upsampling convolution
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        # High-resolution feature refinement
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        # Final output layer
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # Pixel shuffle upsampling (×2)
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

            # === Apply Projection Heads to Separate Feature Spaces ===
            # Project forward features into their independent space
            projected_fwd_feat = self.forward_projection(feat_prop)
            # Project backward features (obtained from out_l) into their independent space
            projected_bwd_feat = self.backward_projection(out_l[i])

            # ======================= [MODIFIED START] =======================
            # === Complex Feature Construction (Experimental Modification: Backward as Real, Forward as Imaginary) ===
            # Real part = Backward features - Forward features
            real = projected_bwd_feat - projected_fwd_feat
            # Imaginary part = Fusion(Backward + Forward) -> Compression -> Residual Block
            img = self.resblock1(self.compress(torch.cat([projected_bwd_feat, projected_fwd_feat], dim=1)))
            # ======================= [MODIFIED END] =======================

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
    1. Initial convolution layer
    2. Multiple residual blocks (without BN)

    Args:
        num_in_ch (int): Number of input channels
        num_out_ch (int): Number of output channels
        num_block (int): Number of residual blocks
    """
    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)