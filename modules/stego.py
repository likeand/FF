from curses import newpad
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from . import backbone
from .dino import vision_transformer as vit

class PanopticFPN(nn.Module):
    def __init__(self, args):
        super(PanopticFPN, self).__init__()

        arch = "vit_base"
        self.patch_size = 8
        self.backbone = vit.__dict__[arch](
            patch_size=8,
            num_classes=0
        )
        self.args = args
        if args.arch_local_save is not None:
            RESUME = f'{args.model_dir}/dino_vitbase8_pretrain.pth'
            self.backbone.load_state_dict(torch.load(RESUME))
            print(f'loaded state dict from {RESUME}')
        self.decoder  = FPNDecoder(args)

    def forward(self, img, n=1):
        with torch.no_grad():
            feat, attn, qkv = self.backbone.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)

        # outs  = self.decoder(image_feat) 
        # return outs 
        return image_feat
    def forward_lbl(self, img):
        res = self.args.res
        if not img.shape[3] == res:
            img = F.interpolate(img, (res, res))
        return self.forward(img)

class FPNDecoder(nn.Module):
    def __init__(self, args):
        super(FPNDecoder, self).__init__()
        # self.conv = nn.Conv2d(768, args.in_dim, kernel_size=1, stride=1, padding=0)
        # self.conv = DoubleConv(768, args.in_dim, 256)

    def forward(self, x):
        # x = feats[-1]
        # return self.conv(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


