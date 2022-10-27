from curses import newpad
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from . import backbone
from .dino import vision_transformer as vit
from einops import rearrange


class PanopticFPN(nn.Module):
    def __init__(self, args):
        super(PanopticFPN, self).__init__()
        arch = args.arch
        self.patch_size = 8
        self.backbone = vit.__dict__[arch](
            patch_size=8,
            num_classes=0
        )
        self.args = args 

        embed_dim = self.backbone.embed_dim * 2

        self.head = LinearClassifier(embed_dim, 1000)
        # self.head = 
        # self.head = vit.DINOHead(768, 1000, norm_last_layer=False)
        path = f'{args.model_dir}/dino_vitbase8_pretrain.pth'
        hpath = f'{args.model_dir}/dino_vitbase8_linearweights.pth'
        self.backbone.load_state_dict(torch.load(path))
        
        state_dict = {k.replace("module.", ""): v for k, v in torch.load(hpath)['state_dict'].items()}
        self.head.load_state_dict(state_dict)
        print(f'loaded state dict from \n {path} \n and {hpath}')
        self.decoder  = FPNDecoder(args)

    def forward(self, img, n=1):
        # feats = self.backbone(x)
        with torch.no_grad():
            feat, attn, qkv = self.backbone.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)

        outs  = self.decoder(image_feat) 
        return outs 
    
    def forward_classification(self, img, n=1):
        with torch.no_grad():
            intermediate_output, attns, qkvs = self.backbone.get_intermediate_feat(img, n)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            # print(f'{output.shape = }')
            # x = torch.cat((output.unsqueeze_(1).expand_as(intermediate_output[-1][:, 1:]).unsqueeze_(-1), intermediate_output[-1][:, 1:].unsqueeze_(-1)), dim=-1)
            # x = rearrange(x, 'b l c n -> b l (c n)')
            # b, l, c = intermediate_output[-1].shape
            # h = int((l-1) ** 0.5)
            # w = h
            # rearr_x = rearrange(x, 'b l c -> (b l) c') 
            # rearr_classification = self.head(rearr_x) # ((bl), c) -> ((bl), n_classes)
            # print(f'{rearr_classification.max() = } {rearr_classification.min() = }')
            # classification = rearrange(rearr_classification, '(b h w) n -> b n h w', b=b, h=h, w=w)
            # return classification   
            # cam = torch.einsum('b c, b l c -> b l', output, intermediate_output[-1][:, 1:])
            # rearr_cam = rearrange(cam, 'b (h w) -> b h w', h=h, w=w)
            attentions = attns[-1]
            print(f'{attentions.shape = }') 
            # attentions.shape = torch.Size([1, 12, 2305, 2305])
            nh = attentions.shape[1] # number of head
            # we keep only the output patch attention
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size
            attentions = attentions.reshape(nh, feat_w, feat_h)
            print(f'after reshape {attentions.shape = }')
            # after reshape attentions.shape = torch.Size([12, 48, 48])
            output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            
            output = output.reshape(output.shape[0], -1)
            return self.head(output), attentions


    def forward_lbl(self, img, eval=False):
        b, c, h, w = img.shape 
        if not (h == 384 and w == 384):
            img = F.interpolate(img, (384, 384)) 
        img = img.to('cuda')
        with torch.no_grad():
            lbl, cam = self.forward_classification(img) 
            return lbl, cam 


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


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)