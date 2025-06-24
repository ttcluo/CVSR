import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from einops import rearrange

from basicsr.archs.arch_util_ib import ResidualBlockNoBN, flow_warp2, make_layer, FlowEstimate, DeblurModule
from basicsr.archs.arch_util import DCNv2Pack as DCN
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class IBRNNet(nn.Module):
    """

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number of residual blocks in each propagation branch.
            Default: 30.
        spynet_path (str): The path of Pre-trained SPyNet model.
            Default: None.
    """
    def __init__(self, num_feat=64, extract_block=12, num_block=30, resType="ResidualBlockNoBN", use_deblur=False, upscale=4):
        super(IBRNNet, self).__init__()
        self.num_feat = num_feat
        self.num_block = num_block
        self.use_deblur = use_deblur
        self.upscale = upscale

        # Flow-based Feature Alignment
        # self.flow_estimate = FlowEstimate(in_feat=64)

        # dcn alignment
        self.offset = ConvResBlock(num_feat*3, num_feat, 3, "ResidualBlock_CA")
        self.offset_weight = nn.Sequential(
            nn.Conv2d(num_feat*3, num_feat, kernel_size=1,stride=1,padding=0,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=1,stride=1,padding=0,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat*3, kernel_size=1,stride=1,padding=0,bias=True),

        )
        self.for_dcn = DCN(num_feat, num_feat, 3, padding=1, deformable_groups=16)
        self.bac_dcn = DCN(num_feat, num_feat, 3, padding=1, deformable_groups=16)
        self.dfe = ConvResBlock(num_feat, num_feat, 3, "ResidualBlock_CA")

        # feat extract
        self.feat_extract = ConvResBlock(3, num_feat, extract_block, resType)

        # Bidirectional Propagation
        self.forward_resblocks = ConvResBlock(num_feat * 4, num_feat, num_block, resType)
        self.backward_resblocks = ConvResBlock(num_feat * 3, num_feat, num_block, resType)

        # Concatenate Aggregation
        self.concate = nn.Conv2d(num_feat * 2, num_feat, kernel_size=1, stride=1, padding=0, bias=True)

        # Pixel-Shuffle Upsampling
        if self.upscale != 1: # 1 or 4
            self.up1 = PSUpsample(num_feat, num_feat, scale_factor=2)
            self.up2 = PSUpsample(num_feat, 64, scale_factor=2)

        # The channel of the tail layers is 64
        self.conv_hr = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # Global Residual Learning
        self.img_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # Activation Function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if self.use_deblur:
            self.deblur = DeblurModule(num_feat=num_feat)

    def comp_flow(self, lrs):
        """Compute optical flow
        Args:
            lrs (tensor): LR frames, the shape is (n, t, c, h, w)
        Return:
            tuple(Tensor): Optical flow.
        """
        n, t, c, h, w = lrs.size()
        forward_flow, backward_flow = [], []
        for i in range(t-1):
            flow = self.flow_estimate(lrs[:, i, ...], lrs[:, i+1, ...])
            forward_flow.append(flow[:, :2])
            backward_flow.append(flow[:, 2:4])
        return forward_flow, backward_flow

    def align(self, x0, x1, prop_type="forward"):
        # offset = self.offset(torch.cat([x0, x1 - x0, x1], dim=1))
        # weight = self.offset_weight(offset)
        offset = torch.cat([x0, x1 - x0, x1], dim=1)
        weight = self.offset_weight(offset)
        offset = self.offset(offset*weight)
        if "forward" == prop_type:
            aligned_feat = self.for_dcn(x0, offset)
        else:
            aligned_feat = self.bac_dcn(x0, offset)

        return self.dfe(self.lrelu(aligned_feat))

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()

        lr_feats = []
        for i in range(t):
            lr_feat = self.feat_extract(lrs[:, i, ...])
            lr_feats.append(lr_feat)
        lr_feats = torch.stack(lr_feats, dim=1)

        # forward_flow, backward_flow = self.comp_flow(lrs)
        # forward_flow, backward_flow = self.comp_flow(lr_feats)

        # Backward Propagation
        rlt = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        for i in range(t-1, -1, -1):
            curr_lr = lr_feats[:, i, :, :, :]
            if i < t-1:
                # flow = backward_flow[i]
                feat_prop = self.align(feat_prop, curr_lr, "backward")
            if i > 0:
                feat_prop_f = self.align(lr_feats[:, i-1, :, :, :].contiguous(), curr_lr, "forward")
            else:
                feat_prop_f = curr_lr

            feat_prop = torch.cat([curr_lr, feat_prop, feat_prop_f], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)
            # deblur
            if self.use_deblur:
                feat_prop = self.deblur(feat_prop)
            rlt.append(feat_prop)

        rlt = rlt[::-1]

        # Forward Propagation
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            curr_lr = lr_feats[:, i, :, :, :]
            if i > 0:
                # flow = forward_flow[i-1]
                feat_prop = self.align(feat_prop, curr_lr, "forward")
            if i < t-1:
                feat_prop_b = self.align(lr_feats[:, i+1, :, :, :].contiguous(), curr_lr, "backward")
            else:
                feat_prop_b = curr_lr

            feat_prop = torch.cat([curr_lr, feat_prop, feat_prop_b, rlt[i]], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)
            # deblur
            if self.use_deblur:
                feat_prop = self.deblur(feat_prop)

            # Fusion and Upsampling
            cat_feat = torch.cat([feat_prop, curr_lr], dim=1)
            sr_rlt = self.lrelu(self.concate(cat_feat))

            if self.upscale != 1:
                sr_rlt = self.lrelu(self.up1(sr_rlt))
                sr_rlt = self.lrelu(self.up2(sr_rlt))
            sr_rlt = self.lrelu(self.conv_hr(sr_rlt))
            sr_rlt = self.conv_last(sr_rlt)

            # Global Residual Learning
            if self.upscale != 1:
                base = self.img_up(lrs[:, i, :, :, :])
            else:
                base = lrs[:, i, :, :, :]

            sr_rlt += base
            rlt[i] = sr_rlt

        return torch.stack(rlt, dim=1)

#############################
# Conv + ResBlock
class ConvResBlock(nn.Module):
    def __init__(self, in_feat, out_feat=64, num_block=30, resType="ResidualBlockNoBN"):
        super(ConvResBlock, self).__init__()

        conv_resblock = []
        if in_feat != out_feat:
            conv_resblock.append(nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=True))
            conv_resblock.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        if "ResidualBlockNoBN" == resType:
            conv_resblock.append(make_layer(ResidualBlockNoBN, num_block, num_feat=out_feat))
        elif "ResidualBlock_CA" == resType:
            conv_resblock.append(make_layer(ResidualBlockNoBN, num_block, num_feat=out_feat, attn_name="Calayer"))
        elif "LF_Block" == resType:
            from basicsr.models.archs.arch_util import LF_Block
            conv_resblock.append(make_layer(LF_Block, num_block, num_feat=out_feat))
        elif "RK2_Block" == resType:
            from basicsr.models.archs.arch_util import RK2_Block
            conv_resblock.append(make_layer(RK2_Block, num_block, num_feat=out_feat))
        elif "SecondOrderRK2_Block" == resType:
            from basicsr.models.archs.arch_util import SecondOrderRK2_Block
            conv_resblock.append(make_layer(SecondOrderRK2_Block, num_block, num_feat=out_feat))
        else:
            print("No res module load...")
            time.sleep(20)

        self.conv_resblock = nn.Sequential(*conv_resblock)

    def forward(self, x):
        return self.conv_resblock(x)

#############################
# Upsampling with Pixel-Shuffle
class PSUpsample(nn.Module):
    def __init__(self, in_feat, out_feat, scale_factor):
        super(PSUpsample, self).__init__()

        self.scale_factor = scale_factor
        self.up_conv = nn.Conv2d(in_feat, out_feat*scale_factor*scale_factor, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.up_conv(x)
        return F.pixel_shuffle(x, upscale_factor=self.scale_factor)


if __name__ == '__main__':
    # model = IBRNNet()
    # lrs = torch.randn(3, 4, 3, 64, 64)
    # rlt = model(lrs)
    # print(rlt.size())

    input = torch.rand(3, 4, 3, 64, 64).cuda()  # B T C H W
    model = IBRNNet().cuda()

    # FLOPs and Parameters
    flops, params = profile(model, inputs=(input,))
    logger = get_root_logger()
    logger.info("Param: {} M".format(params/1e6))
    logger.info("FLOPs: {} G".format(flops/1e9))
    print("Param: {} M".format(params/1e6))
    print("FLOPs: {} G".format(flops/1e9))

