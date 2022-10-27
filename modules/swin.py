import einops
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from . import backbone
from .dino import vision_transformer as vit
from .swin_transformer_v2 import SwinTransformerV2
from einops import rearrange


class PanopticFPN(nn.Module):
    def __init__(self, args):
        super(PanopticFPN, self).__init__()

        self.backbone = SwinTransformerV2(img_size=384, embed_dim=128, depths=[ 2, 2, 18, 2 ], num_heads=[ 4, 8, 16, 32 ], window_size=24, num_classes=1000)
        self.num_features = self.backbone.num_features
        RESUME = f'{args.model_dir}/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth'
        checkpoint = torch.load(RESUME, map_location=args.device)
        msg = self.backbone.load_state_dict(checkpoint['model'], strict=False)
        self.backbone.to(args.device)
        print(f'loaded stated dict, \n{msg}')

    def forward_features(self, x):
        net = self.backbone
        with torch.no_grad():
            x = net.patch_embed(x)
            if net.ape:
                x = x + net.absolute_pos_embed
            x = net.pos_drop(x)

            feats = []
            for layer in net.layers:
                x = layer(x)
                # feats.append(x)

            x = net.norm(x)  # B L C
            feats.append(x)
            b, l, c = x.shape
            h = int(l ** 0.5)
            w = h
            # feats.append(x)
            rearr_x = rearrange(x, 'b l c -> (b l) c') 
            rearr_classification = net.head(rearr_x) # ((bl), c) -> ((bl), n_classes)
            classification = rearrange(rearr_classification, '(b h w) n -> b n h w', b=b, h=h, w=w)
            # x = net.avgpool(x.transpose(1, 2))  # B C 1
            # x = torch.flatten(x, 1) # B C
            return classification, feats

    def forward(self, img, n=1):
        # feats = self.backbone(x)
        with torch.no_grad():
            x, feats = self.forward_features(img)
        img_feats = []
        for f in feats:
            b, n, c = f.shape
            h = int(n ** 0.5)
            f = torch.reshape(f, (b, h, h, c)).permute(0,3,1,2)
            img_feats.append(f)
        outs  = self.decoder(img_feats) 
        return outs 

class FPNDecoder(nn.Module):
    def __init__(self, args, num_features):
        super(FPNDecoder, self).__init__()
        if args.linear:
            self.classifier = nn.Sequential(
                nn.BatchNorm2d(num_features), 
                nn.Conv2d(num_features, args.K_test, 1, 1, 0)
            )
        else:
            self.classifier = nn.Sequential(
                nn.BatchNorm2d(num_features), 
                DoubleConv(num_features, args.K_test, 512)
            )
        
    def forward(self, feats):
        res4 = feats[0]
        out = self.classifier(res4)
        out = F.softmax(out, dim=1)
        return out

    def upsample_add(self, x, y):
        _, _, H, W = y.size()

        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y 

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