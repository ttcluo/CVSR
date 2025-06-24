import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings
from einops import rearrange
from basicsr.archs.arch_util import flow_warp, make_layer
from basicsr.archs.basicvsr_arch import ConvResidualBlocks
from basicsr.archs.spynet_arch import SpyNet
from basicsr.ops.dcn import ModulatedDeformConvPack
from basicsr.utils.registry import ARCH_REGISTRY
import numbers
from thop import profile

@ARCH_REGISTRY.register()
class CMHAVSR(nn.Module):
    """
    A new network combining the strengths of MHAVSR and CBasicVSRPlusPlus4.

    It uses the propagation and alignment mechanisms from MHAVSR and integrates
    the complex convolution reconstruction module from CBasicVSRPlusPlus4.
    """
    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_path=None,
                 cpu_cache_length=100):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        # optical flow
        self.spynet = SpyNet(spynet_path)

        # feature extraction module (from MHAVSR)
        if is_low_res_input:
            self.feat_extract = ConvResidualBlocks(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ConvResidualBlocks(mid_channels, mid_channels, 5))

        # propagation branches (from MHAVSR)
        self.Pyramidfusion = PyramidFusion(ConvBlock, mid_channels)
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = FDAM(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    deformable_groups=16,
                    max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ConvResidualBlocks((4+i) * mid_channels, mid_channels, num_blocks)

        # Upsampling module combining CBasicVSRPlusPlus4's reconstruction and MHAVSR's PyramidFusion
        # --- Start of CBasicVSRPlusPlus4 reconstruction block ---
        self.reconstruction = CConvResidualBlocks(5 * mid_channels, mid_channels, 3)
        self.conv3D = nn.Sequential(
            nn.Conv3d(mid_channels,mid_channels,(3,3,3),(1,1,1),(1,1,1)),
            nn.PReLU(),
            nn.Conv3d(mid_channels, mid_channels, (2, 1, 1), (2, 1, 1), (0,0,0)),
            nn.AdaptiveAvgPool3d(1)
        )
        self.convreal = nn.Conv2d(mid_channels,mid_channels,3,1,1,bias=True)
        self.convimg = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        # --- End of CBasicVSRPlusPlus4 reconstruction block ---

        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn('Deformable alignment module is not added. '
                          'Probably your CUDA is not configured correctly. DCN can only '
                          'be used with CUDA enabled. Alignment is skipped now.')

    def check_if_mirror_extended(self, lqs):
        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        if self.is_mirror_extended:
            flows_forward = flows_backward.flip(1)
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)
        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()
        return flows_forward, flows_backward

    def propagate(self, feats, flows, flows1, module_name):
        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
            feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
            h1 = flows.new_zeros(n, self.mid_channels, h, w)
            h2 = flows.new_zeros(n, self.mid_channels, h, w)
            for i, idx in enumerate(frame_idx):
                feat_current = feats['spatial'][mapping_idx[idx]]
                if self.cpu_cache:
                    feat_current = feat_current.cuda()
                    feat_prop = feat_prop.cuda()
                if i == 0:
                    flow = flows1[:, flow_idx[i + 1], :, :, :]
                    if self.cpu_cache:
                        flow = flow.cuda()
                    cond0 = torch.cat([feats['spatial'][mapping_idx[idx - 1]], feat_current], dim=1)
                    h2 = self.deform_align[module_name](feats['spatial'][mapping_idx[idx - 1]], cond0, flow)
                if 1 <= i <= t - 1:
                    flow1 = flows[:, flow_idx[i], :, :, :]
                    flow2 = flows1[:, flow_idx[i + 1], :, :, :]
                    if self.cpu_cache:
                        flow1 = flow1.cuda()
                        flow2 = flow2.cuda()
                    cond1 = torch.cat([feats['spatial'][mapping_idx[idx + 1]], feat_current], dim=1)
                    cond2 = torch.cat([feats['spatial'][mapping_idx[idx - 1]], feat_current], dim=1)
                    h1 = self.deform_align[module_name](feats['spatial'][mapping_idx[idx + 1]], cond1, flow1)
                    h2 = self.deform_align[module_name](feats['spatial'][mapping_idx[idx - 1]], cond2, flow2)
                    cond3 = torch.cat([feat_prop, feats['spatial'][mapping_idx[idx]]], dim=1)
                    feat_prop = self.deform_align[module_name](feat_prop, cond3, flow1)
                if i == t:
                    h2 = flows.new_zeros(n, self.mid_channels, h, w)
                    flow3 = flows[:, flow_idx[i], :, :, :]
                    if self.cpu_cache:
                        flow3 = flow3.cuda()
                    cond4 = torch.cat([feats['spatial'][mapping_idx[idx + 1]], feat_current], dim=1)
                    h1 = self.deform_align[module_name](feats['spatial'][mapping_idx[idx + 1]], cond4, flow3)
                    cond5 = torch.cat([feat_prop, feats['spatial'][mapping_idx[idx]]], dim=1)
                    feat_prop = self.deform_align[module_name](feat_prop, cond5, flow3)
                feat = [feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] + [feat_prop] + [h1] + [h2]
                if self.cpu_cache:
                    feat = [f.cuda() for f in feat]

                feat = torch.cat(feat, dim=1)
                feat_prop = feat_prop + self.backbone[module_name](feat)
                feats[module_name].append(feat_prop)
                if self.cpu_cache:
                    feats[module_name][-1] = feats[module_name][-1].cpu()
                    torch.cuda.empty_cache()
            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]
        if 'forward' in module_name:
            feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
            h1 = flows.new_zeros(n, self.mid_channels, h, w)
            h2 = flows.new_zeros(n, self.mid_channels, h, w)
            for i, idx in enumerate(frame_idx):
                feat_current = feats['spatial'][mapping_idx[idx]]
                if self.cpu_cache:
                    feat_current = feat_current.cuda()
                    feat_prop = feat_prop.cuda()
                if i == 0:
                    flow = flows1[:, flow_idx[i + 1], :, :, :]
                    if self.cpu_cache:
                        flow = flow.cuda()
                    cond0 = torch.cat([feats['spatial'][mapping_idx[idx + 1]], feat_current], dim=1)
                    h1 = self.deform_align[module_name](feats['spatial'][mapping_idx[idx + 1]], cond0, flow)
                elif 1 <= i <= t - 1:
                    flow1 = flows[:, flow_idx[i], :, :, :]
                    flow2 = flows1[:, flow_idx[i + 1], :, :, :]
                    if self.cpu_cache:
                        flow1 = flow1.cuda()
                        flow2 = flow2.cuda()
                    cond1 = torch.cat([feats['spatial'][mapping_idx[idx - 1]], feat_current], dim=1)
                    cond2 = torch.cat([feats['spatial'][mapping_idx[idx + 1]], feat_current], dim=1)
                    h2 = self.deform_align[module_name](feats['spatial'][mapping_idx[idx - 1]], cond1, flow1)
                    h1 = self.deform_align[module_name](feats['spatial'][mapping_idx[idx + 1]], cond2, flow2)
                    cond3 = torch.cat([feat_prop, feats['spatial'][mapping_idx[idx]]], dim=1)
                    feat_prop = self.deform_align[module_name](feat_prop, cond3, flow1)
                elif i == t:
                    h1 = flows.new_zeros(n, self.mid_channels, h, w)
                    flow3 = flows[:, flow_idx[i], :, :, :]
                    if self.cpu_cache:
                        flow3 = flow3.cuda()
                    cond4 = torch.cat([feats['spatial'][mapping_idx[idx - 1]], feat_current], dim=1)
                    h2 = self.deform_align[module_name](feats['spatial'][mapping_idx[idx - 1]], cond4,  flow3)
                    cond5 = torch.cat([feat_prop, feats['spatial'][mapping_idx[idx]]], dim=1)
                    feat_prop = self.deform_align[module_name](feat_prop, cond5, flow3)
                feat = [feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] + [
                    feat_prop] + [h1] + [h2]
                if self.cpu_cache:
                    feat = [f.cuda() for f in feat]
                feat = torch.cat(feat, dim=1)
                feat_prop = feat_prop + self.backbone[module_name](feat)
                feats[module_name].append(feat_prop)
                if self.cpu_cache:
                    feats[module_name][-1] = feats[module_name][-1].cpu()
                    torch.cuda.empty_cache()
        return feats

    def upsample(self, lqs, feats):
        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            # Pop features from propagation branches
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])

            # --- Start of CBasicVSRPlusPlus4 logic ---
            # Separate features into Real and Imaginary parts for complex convolution
            hrR = torch.cat((hr[0], hr[1], hr[3]), dim=1)
            hrI = torch.cat((hr[0], hr[2], hr[4]), dim=1)
            if self.cpu_cache:
                hrR = hrR.cuda()
                hrI = hrI.cuda()

            # Complex reconstruction and attention
            real, img = self.reconstruction(hrR - hrI, hrR + hrI)
            sreal = real.unsqueeze(2)
            simg = img.unsqueeze(2)
            newf = torch.cat([sreal, simg], dim=2)
            att = self.conv3D(newf)
            att = att.squeeze(2)
            attcos = torch.cos(att)
            attsin = torch.sin(att)
            real = real * attcos
            img = img * attsin
            attreal = real + self.convreal(real)
            attimg = img + self.convimg(img)
            out = torch.cat([attreal, attimg], dim=1)
            out = self.fusion(out)
            # --- End of CBasicVSRPlusPlus4 logic ---

            # --- MHAVSR PyramidFusion integration ---
            hrb = self.Pyramidfusion(out)

            # Upsampling layers
            hrb = self.lrelu(self.pixel_shuffle(self.upconv1(hrb)))
            hrb = self.lrelu(self.pixel_shuffle(self.upconv2(hrb)))
            hrb = self.lrelu(self.conv_hr(hrb))
            hrb = self.conv_last(hrb)
            if self.is_low_res_input:
                hrb += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hrb += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hrb = hrb.cpu()
                torch.cuda.empty_cache()

            outputs.append(hrb)

        return torch.stack(outputs, dim=1)


    def forward(self, lqs):
        n, t, c, h, w = lqs.size()
        self.cpu_cache = True if t > self.cpu_cache_length else False
        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic').view(n, t, c, h // 4, w // 4)

        self.check_if_mirror_extended(lqs)

        feats = {}
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'
                feats[module] = []
                if direction == 'backward':
                    flows = flows_backward
                    flows1 = flows_forward
                elif flows_forward is not None:
                    flows = flows_forward
                    flows1 = flows_backward
                else:
                    flows = flows_backward.flip(1)
                feats = self.propagate(feats, flows, flows1, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)

# --- Start of Custom Modules from CBasicVSRPlusPlus4 ---

class CConvResidualBlocks(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        # In CBasicVSRPlusPlus4, reconstruction input is 5 * mid_channels (320),
        # but the features hrR and hrI are constructed from 3 features, so 3 * 64 = 192
        self.preconvR = nn.Conv2d(192, 64, 3, 1, 1, bias=True)
        self.preconvI = nn.Conv2d(192, 64, 3, 1, 1, bias=True)
        self.num_out_ch = num_out_ch
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # The original CBasicVSRPlusPlus4 has a parameter for num_block,
        # but the implementation uses a fixed number of blocks.
        # Here we follow the implementation with 2 blocks.
        self.main1 = ComplexResidualBlockNoBNbase()
        self.main2 = ComplexResidualBlockNoBNbase()

    def forward(self, feaR,feaI):
        feaRew = self.preconvR(feaR) - self.preconvI(feaI)
        feaINew = self.preconvR(feaI) + self.preconvI(feaR)
        fR = self.lrelu(feaRew)
        fI = self.lrelu(feaINew)
        fR,fI = self.main1(fR,fI)
        fR,fI = self.main2(fR,fI)
        return fR,fI

class ComplexResidualBlockNoBNbase(nn.Module):
    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super().__init__()
        self.res_scale = res_scale
        self.conv1R = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv1I = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2R = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2I = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xR, xI):
        identityR = xR
        identityI = xI
        x1R = self.conv1R(xR)-self.conv1I(xI)
        x1I = self.conv1R(xI)+self.conv1I(xR)
        x1R = self.relu(x1R)
        x1I = self.relu(x1I)
        outR = self.conv2R(x1R)-self.conv2I(x1I)
        outI =  self.conv2R(x1I)+self.conv2I(x1R)
        return identityR + outR * self.res_scale, identityI+outI* self.res_scale

# --- End of Custom Modules from CBasicVSRPlusPlus4 ---


# --- Start of Helper Modules from MHAVSR ---

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class PyramidFusion(nn.Module):
    def __init__(self, block=ConvBlock, dim=64):
        super(PyramidFusion, self).__init__()
        self.dim = dim
        self.ConvBlock1 = ConvBlock(dim, dim, strides=1)
        self.pool1 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)
        self.ConvBlock3 = block(dim, dim, strides=1)
        self.upv5 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.ConvBlock5 = block(dim * 2, dim, strides=1)
        self.conv6 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)
        conv3 = self.ConvBlock3(pool1)
        up5 = self.upv5(conv3)
        up5 = torch.cat([up5, conv1], dim=1)
        conv5 = self.ConvBlock5(up5)
        conv6 = self.conv6(conv5)
        out = x + conv6
        return out

class FDAM(ModulatedDeformConvPack):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        super(FDAM, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )
        self.init_offset()
        self.conv1 = nn.Conv2d(3 * self.out_channels, self.out_channels, 3, 1, 1)

    def init_offset(self):
        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)
        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1):
        extra_feat = torch.cat([extra_feat, flow_1], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1 = offset + flow_1.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        mask = torch.sigmoid(mask)
        out1 = torchvision.ops.deform_conv2d(x, offset_1, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)
        out2 = flow_warp(x, flow_1.permute(0, 2, 3, 1))
        out3 = torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.conv1(out)
        return out

# All other helper modules from MHAVSR are included below for completeness
class SAB(nn.Module):
    def __init__(self, in_dim):
        super(SAB, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.qk_dwconv = nn.Conv2d(in_channels=in_dim // 8, out_channels=in_dim // 8, kernel_size=3, stride=1,
                                   padding=1, groups=in_dim // 8)
        self.v_dwconv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1,
                                  padding=1, groups=in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.project_out = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.qk_dwconv(self.query_conv(x)).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.qk_dwconv(self.key_conv(x)).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.v_dwconv(self.value_conv(x)).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.project_out(out)
        out = self.gamma * out + x
        return out

class CAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class ChannelAttention(nn.Module):
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

class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LGAB(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 dim=64,
                 LayerNorm_type='WithBias',
                 compress_ratio=3,
                 squeeze_factor=30,
                 mlp_ratio=2,
                 act_layer=nn.GELU,
                 drop=0.,
                 ):
        super(LGAB, self).__init__()
        mid_channels = in_channels
        self.mlp_ratio = mlp_ratio
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.sa = SAB(mid_channels)
        self.sc = CAM_Module(mid_channels)
        self.ca = CAB(mid_channels, compress_ratio, squeeze_factor)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=None, act_layer=act_layer, drop=drop)
        self.conv51 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                                    nn.ReLU())
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(2*mid_channels, out_channels, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        shortcut = x
        x = self.norm1(x)
        sa_feat = self.sa(x)
        ca_feat = self.ca(x)
        feat_sum = sa_feat + ca_feat + shortcut
        feat_sum1 = self.norm1(feat_sum).permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        feat_sum2 = self.mlp(feat_sum1).view(b, c, h, w)
        feat_sum_output = feat_sum2 + feat_sum
        return feat_sum_output

# --- End of Helper Modules from MHAVSR ---