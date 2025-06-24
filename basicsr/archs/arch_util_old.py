import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.utils import get_root_logger
from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv,ModulatedDeformConvPackNOInnerOffset
# try:
# from basicsr.ops.dcn import (ModulatedDeformConvPackNOInnerOffset,
#                                         ModulatedDeformConvPack,
#                                         modulated_deform_conv)
# except ImportError:
#     print('Cannot import dcn. Ignore this warning if dcn is not used. '
#           'Otherwise install BasicSR with compiling dcn.')
#     ModulatedDeformConvPack = object
#     ModulatedDeformConvPackNOInnerOffset = object
#     modulated_deform_conv = None

import torch.fft


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, drop=False, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
        drop (boolean): wether to use dropout layer.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block-1):
        layers.append(basic_block(**kwarg))

    if drop:
        layers.append(nn.Dropout(0.1))

    layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResidualBlockNoBN(nn.Module):
    """depthwise_separable Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
        wca: with Channel Attention. default: False
    """
    def __init__(self, num_feat=64, kernel_size=3, res_scale=1, pytorch_init=False, wca=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.wca = wca

        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size, 1, kernel_size//2, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size, 1, kernel_size//2, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if self.wca:
            self.ca = CALayer(num_feat, 8)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        if self.wca:
            out = self.ca(out)
        return identity + out * self.res_scale


class RK2_Block(nn.Module):
    def __init__(self, num_feat=64, kernel_size=3, bias=True,  pytorch_init=False):

        super(RK2_Block, self).__init__()

        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size//2, bias=bias)
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size//2, bias=bias)
        self.conv3 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size//2, bias=bias)
        self.conv4 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size//2, bias=bias)
        self.relu1 = nn.PReLU(num_feat, 0.25)
        self.relu2 = nn.PReLU(num_feat, 0.25)
        self.relu3 = nn.PReLU(num_feat, 0.25)
        self.relu4 = nn.PReLU(num_feat, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)

    def forward(self, x):
        yn = x

        G_yn = self.relu1(self.conv1(x))
        G_yn = self.relu3(self.conv3(G_yn))

        yn_1 = G_yn + yn
        Gyn_1 = self.relu2(self.conv2(yn_1))
        Gyn_1 = self.relu4(self.conv4(Gyn_1))

        yn_2 = Gyn_1 + G_yn
        yn_2 = yn_2 * self.scale1
        out = yn_2 + yn
        return out


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp1(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def flow_warp2(tenInput, tenFlow):
    backwarp_tenGrid = {}
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(tenFlow.device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)


def resize_flow(flow,
                size_type,
                sizes,
                interp_mode='bilinear',
                align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        # offset_absmean = torch.mean(torch.abs(offset))
        # if offset_absmean > 50:
        #     logger = get_root_logger()
        #     logger.warning(
        #         f'Offset abs mean is {offset_absmean}, larger than 50.')

        return modulated_deform_conv(x.contiguous(), offset.contiguous(), mask.contiguous(), self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)


class DCNv2(ModulatedDeformConvPackNOInnerOffset):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        # out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(feat, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        # offset_absmean = torch.mean(torch.abs(offset))
        # if offset_absmean > 50:
        #     offset = self.conv_adjust(offset)
        # if offset_absmean > 50:
        #     logger = get_root_logger()
        #     logger.warning(
        #         f'Offset abs mean is {offset_absmean}, larger than 50.')

        return modulated_deform_conv(x.contiguous(), offset.contiguous(), mask.contiguous(), self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, num_feat=64):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        # self.fc0_r = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        # self.fc0_i = nn.Conv2d(num_feat, num_feat, 1, 1, 0)

        # self.fc1_r = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        # self.fc1_i = nn.Conv2d(num_feat, num_feat, 1, 1, 0)

        # self.fc0 = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)
        # self.fc1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        # self.fc2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0)

        self.conv_layer = nn.Sequential(
            # nn.Conv2d(num_feat*2, num_feat, 1, 1, 0),
            # nn.GELU(),
            nn.Conv2d(num_feat * 2, num_feat * 2, 1, 1, 0),
            nn.GELU(),
        )

    def forward(self, x):
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        # x_ft = torch.fft.rfftn(x)
        # # 方案1：实部和虚部cat到一起再卷积
        # x_ft = torch.cat([x_ft.real, x_ft.imag], dim=1)
        # x_ft = self.fc0(x_ft)
        # x_ft = self.fc1(F.gelu(x_ft))
        # x_ft = self.fc2(x_ft)

        # # 方案2：实部和虚部分别过卷积
        # real = self.fc0_r(x_ft.real)
        # imag = self.fc0_i(x_ft.imag)

        # real = self.fc1_r(F.gelu(x_ft.real))
        # imag = self.fc1_i(F.gelu(x_ft.imag))

        # x_ft = torch.cat([real * torch.cos(imag), real * torch.sin(imag)], dim=1)
        # x_ft = self.fc2(x_ft)

        #Return to physical space
        # x = torch.fft.irfftn(x_ft)

        # 方案3：
        print(f"x:{x.shape}")
        batch = x.shape[0]
        ffted = torch.fft.rfftn(x, dim=(-2, -1))

        print(f"ffted:{ffted.shape}")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + x.size()[2:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        print(f"ffted:{ffted.shape}")
        ffted = ffted.view((batch, -1,) + x.size()[1:]).permute(
            0, 2, 3, 4, 1).contiguous()  # (batch,c, t, h, w/2+1, 2)
        print(f"ffted:{ffted.shape}")
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        print(f"ffted:{ffted.shape}")

        output = torch.fft.rfftn(ffted)
        print(f"output:{output.shape}")

        return output


class FNO(nn.Module):
    def __init__(self, num_feat):
        super(FNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)

        self.layer = 8
        self.conv = nn.ModuleDict()
        self.w = nn.ModuleDict()
        for i in range(self.layer):
            level = f"l{i}"
            self.conv[level] = SpectralConv2d(num_feat)
            self.w[level] = nn.Conv2d(num_feat, num_feat, 1)

        self.fc1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        self.fc2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0)

    def forward(self, x, x_nbor):
        # x_diff = x - x_nbor
        # x = torch.cat((x_nbor, x_diff), dim=1)

        x = torch.cat((x, x_nbor), dim=1)
        x = self.fc0(x)
        res = x

        # fourier layer n
        for i in range(self.layer):
            level = f"l{i}"
            x1 = self.conv[level](x)
            x2 = self.w[level](x)
            x = x1 + x2
            x = F.gelu(x)
            if (i+1)%4 == 0:
                x = x + res
                res = x

        x = self.fc1(x)
        x = F.gelu(x)

        x = x + res
        x = self.fc2(x)
        return x


class LF_Block(nn.Module):
    def __init__(self, num_feat=64, kernel_size=3, bias=True, pytorch_init=False):

        super(LF_Block, self).__init__()

        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size//2, bias=bias)
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size//2, bias=bias)
        self.conv3 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size//2, bias=bias)
        self.conv4 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size//2, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        # self.relu1 = nn.PReLU(num_feat, 0.25)
        # self.relu2 = nn.PReLU(num_feat, 0.25)
        # self.relu3 = nn.PReLU(num_feat, 0.25)
        # self.relu4 = nn.PReLU(num_feat, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.scale3 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.scale4 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)

    def forward(self, x):

        yn = x

        G_yn = self.relu1(self.conv1(G_yn))
        yn_1 = G_yn*self.scale1

        Gyn_1 = self.relu1(self.conv2(yn_1))
        yn_2 = Gyn_1*self.scale2
        yn_2 = yn_2 + yn

        Gyn_2 = self.relu3(self.conv3(Gyn_2))
        yn_3 = Gyn_2*self.scale3
        yn_3 = yn_3 + yn_1

        Gyn_3 = self.relu4(self.conv4(Gyn_3))
        yn_4 = Gyn_3*self.scale4
        out = yn_4 + yn_2
        return out


class PreAct_LF_Block(nn.Module):
    def __init__(
        self, num_feat=64, kernel_size=3, bias=True, pytorch_init=False):

        super(PreAct_LF_Block, self).__init__()

        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size//2, bias=bias)
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size//2, bias=bias)
        self.conv3 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size//2, bias=bias)
        self.conv4 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size//2, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        # self.relu1 = nn.PReLU(num_feat, 0.25)
        # self.relu2 = nn.PReLU(num_feat, 0.25)
        # self.relu3 = nn.PReLU(num_feat, 0.25)
        # self.relu4 = nn.PReLU(num_feat, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.scale3 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.scale4 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)

    def forward(self, x):

        yn = x
        G_yn = self.relu1(x)
        G_yn = self.conv1(G_yn)
        yn_1 = G_yn*self.scale1
        Gyn_1 = self.relu2(yn_1)
        Gyn_1 = self.conv2(Gyn_1)
        yn_2 = Gyn_1*self.scale2
        yn_2 = yn_2 + yn
        Gyn_2 = self.relu3(yn_2)
        Gyn_2 = self.conv3(Gyn_2)
        yn_3 = Gyn_2*self.scale3
        yn_3 = yn_3 + yn_1
        Gyn_3 = self.relu4(yn_3)
        Gyn_3 = self.conv4(Gyn_3)
        yn_4 = Gyn_3*self.scale4
        out = yn_4 + yn_2
        return out



class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()

        self.features = nn.Sequential(

            # input is (3) x 128 x 128
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 128 x 128
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            #nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 44 x 64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (128) x 64 x 64
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True),
            #nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (128) x 32 x 32
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 32 x 32
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True),
            #nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 16 x 16
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 16 x 16
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 8 x 8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 8 x 8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        # self.fc2 = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):

        out = self.features(input)
        # state size. (512) x 6 x 6
        # out = out.view(out.size(0), -1)

        # print(f"out.shape:{out.shape}")
        # state size. (512 x 6 x 6)
        out = self.fc1(out)

        # state size. (1024)
        out = self.LeakyReLU(out)

        out = self.fc2(out)
        #out = self.sigmoid(out)
        return out.view(-1, 1).squeeze(1)


class DeblurModule(nn.Module):
    """Deblur module.

    Args:
        num_feat (int): Channel number of intermediate features. Default: 64.
    return:
        deblur result
    """

    def __init__(self, num_feat=64, hr_in=False):
        super(DeblurModule, self).__init__()
        self.hr_in = hr_in

        # generate feature pyramid
        self.stride_conv_l2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.stride_conv_l3 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        self.resblock_l3 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_1 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_2 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l1 = nn.ModuleList(
            [ResidualBlockNoBN(num_feat=num_feat) for i in range(5)])

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):

        # generate feature pyramid
        feat_l2 = self.lrelu(self.stride_conv_l2(x))
        feat_l3 = self.lrelu(self.stride_conv_l3(feat_l2))

        feat_l3 = self.upsample(self.resblock_l3(feat_l3))
        feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3
        feat_l2 = self.upsample(self.resblock_l2_2(feat_l2))

        for i in range(2):
            x = self.resblock_l1[i](x)
        featx_l1 = x + feat_l2
        for i in range(2, 5):
            x = self.resblock_l1[i](x)
        return x


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=13, stride=1, padding=6):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = ((kernel_size - 1) / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


# from basicsr.models.archs.swinir_arch import *
class SwinBlock(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=64, out_chans=3, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=4, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',):
        super(SwinBlock, self).__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        if in_chans != embed_dim:
            self.conv_first = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=True),
            )

        if out_chans != embed_dim:
            self.conv_out = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(embed_dim, out_chans, kernel_size=3, stride=1, padding=1, bias=True),
            )

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

    def forward(self, x):
        b, _, _, _ = x.shape
        x_size = (x.shape[-2], x.shape[-1])

        if hasattr(self, "conv_first"):
            x = self.conv_first(x)

        for layer in self.layers:
            # transformer block
            x = self.patch_embed(x)
            if self.ape:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)

            x = layer(x, x_size)

            x = self.norm(x)  # B L C
            x = self.patch_unembed(x, x_size)

            x = x.view(b, *(x.shape[-3:]))

        if hasattr(self, "conv_out"):
            x = self.conv_out(x)

        return x


if __name__ == "__main__":
    x = torch.rand([6, 64, 96, 96])
    x_nbor = torch.rand([6, 64, 96, 96])
    fno = FNO(num_feat=64)
    y = fno(x, x_nbor)

    # x = torch.rand([6, 64, 96, 96])
    # model = RK2_Block(64)
    # y = model(x)
