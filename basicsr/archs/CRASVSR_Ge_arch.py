import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings

from basicsr.archs.arch_util import flow_warp
from basicsr.archs.basicvsr_arch import ConvResidualBlocks
from basicsr.archs.spynet_arch import SpyNet
from basicsr.ops.dcn import ModulatedDeformConvPack
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.basicvsr_arch import TSAFusion


@ARCH_REGISTRY.register()
class CGRASVSR(nn.Module):
    """A new network combining RASVSR's propagation with complex convolution upsampling.

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
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

        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ConvResidualBlocks(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ConvResidualBlocks(mid_channels, mid_channels, 5))

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.fusion = nn.ModuleDict()
        modules = ['backward_1', 'forward_1']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = SecondOrderDeformableAlignment(
                    2 * mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    deformable_groups=16,
                    max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ConvResidualBlocks((1 + i) * mid_channels, mid_channels, num_blocks)
            self.fusion[module] = TSAFusion(num_feat=mid_channels, num_frame=2, center_frame_idx=0)

        # --- Start: Upsampling module from CBasicVSRPlusPlus4 ---
        # Instead of a single reconstruction block, we use the complex convolution approach.
        # Input to this block will be 2 * mid_channels (spatial + one direction)
        self.reconstruction = CConvResidualBlocks(2 * mid_channels, mid_channels, 2)

        # 3D convolution for complex attention
        self.conv3D = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.PReLU(),
            nn.Conv3d(mid_channels, mid_channels, (2, 1, 1), (2, 1, 1), (0, 0, 0)),
            nn.AdaptiveAvgPool3d(1)
        )
        # Additional layers for the complex attention mechanism
        self.convreal = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.convimg = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        # Renamed to avoid conflict with TSAFusion's self.fusion
        self.fusion_complex = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)

        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)
        # --- End: Upsampling module from CBasicVSRPlusPlus4 ---

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

    def propagate(self, feats, flows, module_name):
        n, t, _, h, w = flows.size()
        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()

            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

            # concatenate and residual blocks
            last_feat = [feats[k][idx] for k in feats if k not in ['spatial', module_name]]
            feat_current = torch.cat([feat_current] + last_feat, dim=1)
            feat_current = self.backbone[module_name](feat_current)
            feat = [feat_current] + [feat_prop]

            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.stack(feat, dim=1)
            feat_prop = feat_prop + self.fusion[module_name](feat)

            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        outputs = []
        num_outputs = len(feats['spatial'])
        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            # Pop features for the current frame
            feat_bwd = feats['backward_1'].pop(0)
            feat_fwd = feats['forward_1'].pop(0)
            feat_spatial = feats['spatial'][mapping_idx[i]]

            if self.cpu_cache:
                feat_bwd = feat_bwd.cuda()
                feat_fwd = feat_fwd.cuda()
                feat_spatial = feat_spatial.cuda()

            # Define "Real" and "Imaginary" inputs for the complex reconstruction
            hr_real = torch.cat((feat_spatial, feat_bwd), dim=1)
            hr_imag = torch.cat((feat_spatial, feat_fwd), dim=1)

            # C.T and C.A: Complex attention mechanism
            real, img = self.reconstruction(hr_real, hr_imag)

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
            out = self.fusion_complex(out)

            # Standard upsampling path
            hr = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)

            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

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

        for iter_ in [1]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'
                feats[module] = []
                if direction == 'backward':
                    flows = flows_backward
                else:
                    flows = flows_forward

                feats = self.propagate(feats, flows, module)

                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)


class SecondOrderDeformableAlignment(ModulatedDeformConvPack):
   def __init__(self, *args, **kwargs):
       self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
       super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)
       self.conv_offset = nn.Sequential(
           nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
           nn.LeakyReLU(negative_slope=0.1, inplace=True),
           nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
           nn.LeakyReLU(negative_slope=0.1, inplace=True),
           nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
           nn.LeakyReLU(negative_slope=0.1, inplace=True),
           nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
       )
       self.init_offset()

   def init_offset(self):
       def _constant_init(module, val, bias=0):
           if hasattr(module, 'weight') and module.weight is not None:
               nn.init.constant_(module.weight, val)
           if hasattr(module, 'bias') and module.bias is not None:
               nn.init.constant_(module.bias, bias)
       _constant_init(self.conv_offset[-1], val=0, bias=0)

   def forward(self, x, extra_feat, flow_1, flow_2):
       extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
       out = self.conv_offset(extra_feat)
       o1, o2, mask = torch.chunk(out, 3, dim=1)
       offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
       offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
       offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
       offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
       offset = torch.cat([offset_1, offset_2], dim=1)
       mask = torch.sigmoid(mask)
       return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                            self.dilation, mask)


# --- Start: Helper modules for Complex Convolution ---

class CConvResidualBlocks(nn.Module):
    def __init__(self, num_in_ch, num_out_ch=64, num_block=2):
        super().__init__()
        self.preconvR = nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True)
        self.preconvI = nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.main = nn.Sequential(
            ComplexResidualBlockNoBNbase(num_feat=num_out_ch),
            ComplexResidualBlockNoBNbase(num_feat=num_out_ch)
        )

    def forward(self, feaR, feaI):
        # Complex convolution for the first layer
        feaRew = self.preconvR(feaR) - self.preconvI(feaI)
        feaINew = self.preconvR(feaI) + self.preconvI(feaR)

        fR = self.lrelu(feaRew)
        fI = self.lrelu(feaINew)

        fR, fI = self.main( (fR, fI) )

        return fR, fI


class ComplexResidualBlockNoBNbase(nn.Module):
    def __init__(self, num_feat=64, res_scale=1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1R = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv1I = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2R = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2I = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        xR, xI = x
        identityR = xR
        identityI = xI

        # Conv1 (Complex)
        x1R = self.conv1R(xR) - self.conv1I(xI)
        x1I = self.conv1R(xI) + self.conv1I(xR)

        x1R = self.relu(x1R)
        x1I = self.relu(x1I)

        # Conv2 (Complex)
        outR = self.conv2R(x1R) - self.conv2I(x1I)
        outI = self.conv2R(x1I) + self.conv2I(x1R)

        return identityR + outR * self.res_scale, identityI + outI * self.res_scale

# --- End: Helper modules for Complex Convolution ---