import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from .edvr_arch import PCDAlignment, TSAFusion
from .spynet_arch import SpyNet

@ARCH_REGISTRY.register()
class CVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=30, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment flow alignment module - Use SpyNet to calculate inter-frame motion
        self.spynet = SpyNet(spynet_path)
        # for p in self.spynet.parameters():
        #     p.requires_grad = False

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

        #self.unrelationalmapping = nn.Conv2d(num_feat*2,num_feat*2,3,1,1)
        #self.compress = nn.Conv2d(num_feat*2+48,num_feat,1,1,0)
        #self.compress2 = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)
        #self.resblock1 = ResidualBlockNoBN(num_feat)
        # self.realesa = ESA(num_feat)
        # self.imgesa = ESA(num_feat)
        #self.resblock2 = ResidualBlockNoBN(num_feat*2)
        # self.resblock3 = ResidualBlockNoBN(num_feat*2)
        # self.resblock4 = ResidualBlockNoBN(num_feat*2)
        # self.Capattention = ESA(num_feat)
        # self.Capattentio2 = ESA(num_feat)
        #self.deconv = nn.Conv2d(num_feat*2,num_feat*4,1,1,0)
        #self.phaserotat = ComplexRotationMapping()
        # self.resblock5 = ComplexResidualBlockNoBN(num_feat)
        # self.resblock6 = ComplexResidualBlockNoBN(num_feat)
        # self.resblock7 = ComplexResidualBlockNoBN(num_feat)
        # self.resblock8 = ComplexResidualBlockNoBN(num_feat)

        # self.wavedec = WaveletTransform(scale=2, dec=True, params_path='wavelet_weights_c2.pkl', transpose=True,
        #                                 channel=3)
        # reconstruction
        # self.mlp = nn.Sequential(
        #     nn.Conv2d(128, 96, 1, padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(96, 64, 1, padding=0, bias=True),
        # )

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
        """Calculate forward and backward optical flow

        Args:
            x (Tensor): Input video sequence [b, n, c, h, w]

        Returns:
            Tuple: Forward flow and backward flow
        """
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, x):
        """Forward propagation

        Args:
            x (Tensor): Low-resolution input sequence [b, n, c, h, w]

        Returns:
            Tensor: High-resolution output sequence [b, n, c, 4h, 4w]
        """

        # Step 1: Calculate optical flow
        flows_forward, flows_backward = self.get_flow(x)
        b, n, _, h, w = x.size()

        # backward branch === Backward Propagation Branch (Reverse Processing) ===
        out_pre = []
        out_l = [] # Store backward features
        feat_prop = x.new_zeros(b, self.num_feat, h, w) # Initialize features

        # Process from the last frame backwards
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :] # Current frame
            # Optical flow alignment (except for the first frame)
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

             # Feature extraction
            feat_prop = torch.cat([x_i, feat_prop], dim=1) # Concatenate current frame and propagated features
            feat_prop = self.backward_trunk(feat_prop) # Through residual blocks
            # Store features (insert in forward order)
            out_l.insert(0, feat_prop)

        # forward branch === Forward Propagation Branch (Forward Processing) ===
        feat_prop = torch.zeros_like(feat_prop)  # Reset features
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            #f48 = self.wavedec(lastoutput)
            # real = feat_prop - out_l[i]
            # img = feat_prop + out_l[i]
            # z1 = torch.zeros((b,128,1,1)).cuda()
            # z1 = z1+3.1415926535/4
            # att = self.mlp(z1)
            # c1 = real*torch.cos(att)
            # c2 = img*torch.sin(att)

            # === Complex Feature Construction ===
            # Real part = Forward feature - Backward feature
            real = feat_prop - out_l[i]
            # Imaginary part = Fusion(Forward + Backward) -> Compress -> Residual block
            img = self.resblock1(self.compress(torch.cat([feat_prop, out_l[i]], dim=1)))

            # === Complex Attention Mechanism ===
            # Construct 3D feature tensor [b, c, 2, h, w]
            sreal = real.unsqueeze(2) # Add dimension to real part
            simg = img.unsqueeze(2) # Add dimension to imaginary part
            newf = torch.cat([sreal,simg],dim=2) # Concatenate real and imaginary parts

            # 3D convolution to extract attention
            att = self.conv3D(newf) # [b, c, 1, 1, 1]
            att = att.squeeze(2) # Reduce dimension [b, c, 1, 1]

            # Euler's formula rotation (complex multiplication)
            attcos = torch.cos(att) # Rotation factor real part
            attsin = torch.sin(att) # Rotation factor imaginary part
            real = real * attcos # Rotate real part
            img = img *attsin # Rotate imaginary part

            # Enhance rotated features
            attreal = real+self.convreal(real) # Enhance real part
            attimg = img+self.convimg(img)  # Enhance imaginary part

            # Merge rotated complex features
            out = torch.cat([attreal, attimg], dim=1)
            #out = self.resblock2(out)

            #c2 = out_l[i] + feat_prop
            #c2 = feat_prop + out_l[i]
            # b,c,h,w = c1.size()
            # newc1 = c1.reshape(b,c,-1)
            # newc2 = c2.reshape(b,c,-1)
            #
            # simmatrix = torch.bmm(newc1.detach(),newc2.transpose(1,2).detach())
            # meansim = torch.mean(simmatrix,dim=-1,keepdim=True).unsqueeze(-1)
            # simchannel = self.mlp(meansim)
            # c1 = c1 * torch.cos(simchannel)
            # c2 = c2 * torch.sin(simchannel)
            # rr = self.realesa(c1,c1)
            # ii = self.imgesa(c2,c2)
            #
            # newc1 = rr-ii
            # newc2 =self.resblock2(self.compress2(torch.cat([rr,ii],dim=1)))
            # real = self.Capattention(feat_prop,feat_prop)
            # img = self.Capattentio2(out_l[i],out_l[i])

            # real = feat_prop * a1
            # img = out_l[i] * a2
            #
            # c1 = preforward.clone()
            # c2 = out_pre[i].clone()
            # with torch.no_grad():
            #     c1[torch.abs(c1) < 0.001] = 0.001
            #     c2[torch.abs(c2) < 0.001] = 0.001

            # c1.requires_grad_(False)
            # # c2.requires_grad_(False)
            # k1 = (real+img)/(torch.abs(2*preforward)+0.0001)
            # k2 = -img/(torch.abs(2*out_pre[i])+0.0001)
            # img = k1*out_pre[i]+k2*preforward

            # upsample === Super-resolution Reconstruction ===
            # Feature fusion (Complex -> Real)
            out = self.lrelu(self.fusion(out))
            # First upsampling (×2)
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            # Second upsampling (×2 → Total ×4)
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            # High-resolution refinement
            out = self.lrelu(self.conv_hr(out))
            # Final output
            out = self.conv_last(out)

            #lastoutput = out

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

# class ComplexConvResidualBlocks(nn.Module):

#     def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
#         super().__init__()
#         self.preconv = nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True)
#         self.num_out_ch = num_out_ch
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#         self.main = nn.Sequential(
#             make_layer(ComplexResidualBlockNoBNbase, num_block, num_feat=num_out_ch))

#     def forward(self, fea):
#         f = self.lrelu(self.preconv(fea))
#         f = self.main(f)
#         return f





# if __name__=="__main__":
#     network=VSR_FCN(64).cuda()
#     print('# model parameters:', sum(param.numel() for param in generator.parameters()))
#     inputdata=torch.rand(4,7,3,50,50).cuda()
#     output,_=network(inputdata)
#     print(output.shape)

@ARCH_REGISTRY.register()
class CIconVSR(nn.Module):
    """IconVSR, proposed also in the BasicVSR paper
    Improved IconVSR model, introducing complex domain feature representation and 3D attention mechanism

    Args:
        num_feat (int): Number of feature channels, default 64
        num_block (int): Number of residual blocks, default 15
        keyframe_stride (int): Keyframe stride, default 5
        temporal_padding (int): Temporal padding, default 2
        spynet_path (str): Path to SpyNet pretrained weights
        edvr_path (str): Path to EDVR pretrained weights
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

        # Keyframe feature extractor (EDVR architecture)
        self.edvr = EDVRFeatureExtractor(temporal_padding * 2 + 1, num_feat, edvr_path)

        # Optical flow estimation network (SpyNet)
        self.spynet = SpyNet(spynet_path)

        # === Propagation Branches ===
        # Backward fusion: concatenate features + keyframe features
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        # Backward trunk: input (LR frame + propagated features)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # Forward fusion: concatenate features + keyframe features
        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        # Forward trunk: input (LR frame + backward features + propagated features)
        self.forward_trunk = ConvResidualBlocks(2 * num_feat + 3, num_feat, num_block)

        # === Complex Domain Processing Core C.T ===
        # Feature compression: dimensionality reduction after concatenating bidirectional features
        self.compress = nn.Conv2d(2*num_feat, num_feat, 1, 1, 0)
        # Imaginary part generation: residual block processing compressed features
        self.resblock1 = ResidualBlockNoBN(num_feat)

        # 3D Complex Attention Mechanism C.A
        self.conv3D = nn.Sequential(
            nn.Conv3d(num_feat, num_feat, (3, 3, 3), (1, 1, 1), (1, 1, 1)),  # 3D spatial-temporal convolution
            nn.PReLU(),  # Parametric ReLU activation
            nn.Conv3d(num_feat, num_feat, (2, 1, 1), (2, 1, 1), (0, 0, 0)),  # Dimensionality reduction convolution
            nn.AdaptiveAvgPool3d(1)  # Global pooling to get attention vector
        )
        # Real part enhancement convolution
        self.convreal = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        # Imaginary part enhancement convolution
        self.convimg = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        # === Reconstruction Module ===
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)  # First upsampling convolution
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)  # Second upsampling convolution
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)  # High-resolution feature refinement
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)  # Final output convolution

        # PixelShuffle upsampler (2x upsampling)
        self.pixel_shuffle = nn.PixelShuffle(2)

        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def pad_spatial(self, x):
        """ Apply padding spatially.

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
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
        return feats_keyframe

    def forward(self, x):
        b, n, _, h_input, w_input = x.size()

        x = self.pad_spatial(x)
        h, w = x.shape[3:]

        keyframe_idx = list(range(0, n, self.keyframe_stride))
        if keyframe_idx[-1] != n - 1:
            keyframe_idx.append(n - 1)  # last frame is a keyframe

        # compute flow and keyframe features
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

            # Main network (fuses backward features)
            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # === Complex Domain Processing C.T ===
            real = feat_prop - out_l[i]
            img = self.resblock1(self.compress(torch.cat([feat_prop, out_l[i]], dim=1)))

            # === C.A ===
            sreal = real.unsqueeze(2)
            simg = img.unsqueeze(2)
            newf = torch.cat([sreal,simg],dim=2)
            att = self.conv3D(newf)
            att = att.squeeze(2)
            # Phase rotation
            attcos = torch.cos(att)
            attsin = torch.sin(att)
            real = real * attcos
            img = img *attsin
            # Feature enhancement
            attreal = real+self.convreal(real)
            attimg = img+self.convimg(img)


            out = torch.cat([attreal, attimg], dim=1)
            #outimg =
            out = self.lrelu(self.fusion(out))
            
            # upsample
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

        # extract pyramid features
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=64)
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