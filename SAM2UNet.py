import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sam2.build_sam import build_sam2
from models.grid_attention_layer import GridAttentionBlock2D
from models.utils import UnetDsv2, UnetUp2_CT, UnetConv2, UnetGridGatingSignal2
from ukan_kan import UkanFeatureMapBlock
from typing import Optional


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, use_kan: bool = True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.use_kan = use_kan

        if use_kan:
            self.kan = UkanFeatureMapBlock(
                dim=out_channels
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_kan:
            x = self.kan(x)

        return x


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class SkipFusion(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, topo):
        x = torch.cat([rgb, topo], dim=1)
        return self.fusion(x)


class SAM2UNet(nn.Module):
    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 mode='concatenation',
                 attention_dsample=(2,2),
                 is_batchnorm=True,
                 topo_in_chans: int = 1,
                 topo_backbone: str = "convnext_tiny",
                 pretrained_topo: bool = True,
                 use_kan: bool = False,
                ) -> None:
        super(SAM2UNet, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.attention_dsample = attention_dsample
        self.is_batchnorm = is_batchnorm
        self.topo_in_chans = topo_in_chans
        self.topo_backbone = topo_backbone
        self.pretrained_topo = pretrained_topo
        self.use_kan = use_kan

        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )

        self.topo_encoder = timm.create_model(
            topo_backbone,
            pretrained=pretrained_topo,
            in_chans=topo_in_chans,
            features_only=True,
        )
        topo_channels = self.topo_encoder.feature_info.channels()

        # self.rfb1 = RFB_modified(144, 64)
        # self.rfb2 = RFB_modified(288, 64)
        # self.rfb3 = RFB_modified(576, 64)
        # self.rfb4 = RFB_modified(1152, 64)

        self.proj1 = SkipFusion(144 + topo_channels[0], topo_channels[0])
        self.proj2 = SkipFusion(288 + topo_channels[1], topo_channels[0])
        self.proj3 = SkipFusion(576 + topo_channels[2], topo_channels[0])
        self.proj4 = SkipFusion(1152 + topo_channels[3], topo_channels[0])

        # self.proj1 = SkipFusion(144 + topo_channels[0], 64)
        # self.proj2 = SkipFusion(288 + topo_channels[1], 64)
        # self.proj3 = SkipFusion(576 + topo_channels[2], 64)
        # self.proj4 = SkipFusion(1152 + topo_channels[3], 64)

        # self.up1 = Up(128, 64, use_kan=use_kan)
        # self.up2 = Up(128, 64, use_kan=use_kan)
        # self.up3 = Up(128, 64, use_kan=use_kan)
        # self.up4 = Up(128, 64, use_kan=use_kan)  # Unused in 3-stage decoder

        self.center = UnetConv2(64, 64, is_batchnorm)
        self.gating = UnetGridGatingSignal2(64, 64, kernel_size=(1, 1), is_batchnorm=is_batchnorm)
    
        # attention blocks
        self.attentionblock2 = GridAttentionBlock2D(in_channels=64, gating_channels=64,
                                                    inter_channels=None, sub_sample_factor=attention_dsample, mode=mode)
        self.attentionblock3 = GridAttentionBlock2D(in_channels=64, gating_channels=64,
                                                    inter_channels=None, sub_sample_factor=attention_dsample, mode=mode)
        self.attentionblock4 = GridAttentionBlock2D(in_channels=64, gating_channels=64,
                                                    inter_channels=None, sub_sample_factor=attention_dsample, mode=mode)

        self.up_concat4 = UnetUp2_CT(64, 64, is_batchnorm)
        self.up_concat3 = UnetUp2_CT(64, 64, is_batchnorm)
        self.up_concat2 = UnetUp2_CT(64, 64, is_batchnorm)
        self.up_concat1 = UnetUp2_CT(64, 64, is_batchnorm)

        # self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        # self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        # self.head = nn.Conv2d(64, 1, kernel_size=1)
        self.dsv4 = UnetDsv2(in_size=64, out_size=64, scale_factor=8)
        self.dsv3 = UnetDsv2(in_size=64, out_size=64, scale_factor=4)
        self.dsv2 = UnetDsv2(in_size=64, out_size=64, scale_factor=2)
        self.dsv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

    def forward(self, x, x_topo=None):
        if x_topo is None:
            if x.shape[1] > 3:
                x_rgb = x[:, :3, :, :]
                x_topo = x[:, 3:, :, :]
            else:
                raise ValueError(
                    f"Model is configured with {self.topo_in_chans} topography channels, "
                    f"but input has only {x.shape[1]} channels. Pass both RGB and topography channels, "
                    f"or pass x_topo explicitly."
                )
        else:
            x_rgb = x

        x1, x2, x3, x4 = self.encoder(x_rgb)

        topo_feats = self.topo_encoder(x_topo)
        t1, t2, t3, t4 = topo_feats[0], topo_feats[1], topo_feats[2], topo_feats[3]

        # x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x1 = self.proj1(x1, t1)
        x2 = self.proj2(x2, t2)
        x3 = self.proj3(x3, t3)
        x4 = self.proj4(x4, t4)

        # Gating Signal Generation
        center = self.center(x4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(x4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(x3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(x2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(x1, up2)

        # x = self.up1(x4, x3)
        # out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        # x = self.up2(x, x2)
        # out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        # x = self.up3(x, x1)
        # out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        out4 = self.dsv4(up4)
        out3 = self.dsv3(up3)
        out2 = self.dsv2(up2)
        out1 = self.dsv1(up1)
        # final = self.final(torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1))

        return out1, out2, out3, out4


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        # Example 1: 4-channel single tensor (3 RGB + 1 DTM)
        model = SAM2UNet(topo_in_chans=1, topo_backbone="convnext_tiny", pretrained_topo=False).to(device)
        x = torch.randn(1, 4, 512, 512).to(device)
        out, out1, out2 = model(x)
        print("Single tensor forward outputs:", out.shape, out1.shape, out2.shape)

        # Example 2: Explicit two-stream forward (3 RGB and 5 Topo)
        model_custom = SAM2UNet(topo_in_chans=5, topo_backbone="convnext_nano", pretrained_topo=False).to(device)
        x_rgb = torch.randn(1, 3, 512, 512).to(device)
        x_topo = torch.randn(1, 5, 512, 512).to(device)
        out, out1, out2 = model_custom(x_rgb, x_topo=x_topo)
        print("Explicit two-stream forward outputs:", out.shape, out1.shape, out2.shape)