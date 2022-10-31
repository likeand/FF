import einops
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from . import backbone
from .dino import vision_transformer as vit
from .swin_transformer_v2 import SwinTransformerV2
from einops import rearrange
import matplotlib.pyplot as plt 
from module_utils import make_crop, make_one_from_crop
class PanopticFPN(nn.Module):
    def __init__(self, args):
        super(PanopticFPN, self).__init__()
        self.swin = SwinTransformerV2(img_size=384, embed_dim=128, depths=[ 2, 2, 18, 2 ], num_heads=[ 4, 8, 16, 32 ], window_size=24, num_classes=1000)
        self.num_features = self.swin.num_features
        RESUME = f'{args.model_dir}/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth'
        checkpoint = torch.load(RESUME, map_location=args.device)
        msg = self.swin.load_state_dict(checkpoint['model'], strict=False)
        self.swin.to(args.device)
        print(f'loaded stated dict for swin, \n{msg}')

        self.patch_size = 8
        # self.dino = vit.__dict__[args.arch](
        #     patch_size=8,
        #     num_classes=0
        # )
        self.dino = DinoFeaturizer()
        dic = torch.load(f'{args.model_dir}/cityscapes_vit_base_1.ckpt')['state_dict']
        nd = {}
        for k in dic:
            if 'net.' in k:
                key = k.split('net.')[-1]
                nd[key] = dic[k]
        self.dino.load_state_dict(nd)
        # # RESUME = f'{args.model_dir}/dino_vitbase8_pretrain.pth'
        # # self.dino.load_state_dict(torch.load(RESUME))
        # # print(f'loaded state dict from {RESUME}')
        # self.obj_decoder  = FPNDecoder(args, self.num_features, args.obj_classes)
        # obj_decoder_path = '/home/zhulifu/unsup_seg/old_3090/picie_and_mynet/PiCIE/test_cut_swin_35_wtrim_on_val_wnorm_centroids.pth'
        # # self.obj_decoder.load_state_dict(torch.load(obj_decoder_path))
        # self.obj_decoder.classifier.weight.data = torch.load(obj_decoder_path).unsqueeze(-1).unsqueeze(-1)
        # print(f'loaded state dict for swin decoder from {obj_decoder_path}')
        # self.obj_decoder.to(args.device)
        
        # self.things_decoder  = FPNDecoder(args, 100, args.things_classes)
        # # things_decoder_path = '/home/zhulifu/unsup_seg/old_3090/picie_and_mynet/PiCIE/test_dino_on_val_wnorm_centroids.pth'
        # things_decoder_path = '/home/zhulifu/unsup_seg/STEGO-master/models/cityscapes_vit_base_1.ckpt'
        # data = dic['cluster_probe.clusters'].unsqueeze(-1).unsqueeze(-1)
        # # self.things_decoder.load_state_dict(torch.load(things_decoder_path))
        # self.things_decoder.classifier.weight.data = data
        # print(f'loaded state dict for swin decoder from {things_decoder_path}')
        # self.things_decoder.to(args.device)
        self.device = args.device
        self.args = args 
        self.conv = nn.Conv2d(768, 1024, 1)
        torch.nn.init.kaiming_uniform_(self.conv.weight)

    def get_classification(self, cuts):        
        predictions = []
        all_features = []
        for cut in cuts:
            image = F.interpolate(cut, (384, 384)).to(self.device)
            with torch.no_grad():
                classifications, features = self.forward_classification(image) # (b, n, h, w) n = 1000
            # classifications[classifications < 35] = 0
            predictions.append(classifications)
            all_features.append(features)
        return predictions, all_features

    def forward_swin(self, x):
        net = self.swin 
        x = x.to(self.device)
        with torch.no_grad():
            x = net.patch_embed(x)
            if net.ape:
                x = x + net.absolute_pos_embed
            x = net.pos_drop(x)
            for layer in net.layers:
                x = layer(x)
                # feats.append(x)
            x = net.norm(x)  # B L C
            b, l, c = x.shape
            h = int(l ** 0.5)
            w = h
            rearr_x = rearrange(x, 'b (h w) c -> b c h w', b=b, h=h, w=w)
            return x, rearr_x
            
    def forward_classification(self, x):
        net = self.swin 
        # x = x.to(self.device)
        # with torch.no_grad():
        #     x = net.patch_embed(x)
        #     if net.ape:
        #         x = x + net.absolute_pos_embed
        #     x = net.pos_drop(x)
        #     for layer in net.layers:
        #         x = layer(x)
        #         # feats.append(x)
        #     x = net.norm(x)  # B L C
        x, feature = self.forward_swin(x)
        b, c, h, w = feature.shape
        rearr_x = rearrange(x, 'b l c -> (b l) c') 
        rearr_classification = net.head(rearr_x) # ((bl), c) -> ((bl), n_classes)
        classification = rearrange(rearr_classification, '(b h w) n -> b n h w', b=b, h=h, w=w)
        return classification, feature
    
    # def forward_features(self, x):
    #     x = x.to(self.device)
    #     net = self.dino  
    #     with torch.no_grad():
    #         x = net.patch_embed(x)
    #         if net.ape:
    #             x = x + net.absolute_pos_embed
    #         x = net.pos_drop(x)
    #         # feats = []
    #         for layer in net.layers:
    #             x = layer(x)
    #             # feats.append(x)
    #         x = net.norm(x)  # B L C
    #         # feats.append(x)
    #         return x

    def forward_dino(self, img, n=1):
        if not img.shape[3] == 384:
            img = F.interpolate(img, (384, 384)) 
        img = img.to(self.device)
        with torch.no_grad():
            feat, attn, qkv = self.dino.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
        image_feat = F.interpolate(image_feat, self.args.tar_res, mode='bilinear')
        return image_feat
    
    def forward(self, img):
        if self.args.method == '':
            _, x = self.forward_swin(img)
            return x
        elif self.args.method == 'cam':
            low_feature = self.forward_dino(img)

            classification, high_feature = self.forward_classification(img)
            mask = classification.amax(dim=1) > 20 # (b, h, w)
            mask = mask.unsqueeze(1) # (b, 1, h, w)
            mask = F.interpolate(mask.float(), low_feature.shape[-2:])
            
            high_feature = F.interpolate(high_feature, low_feature.shape[-2:])
            low_feature = self.conv(low_feature)
            return mask * high_feature + (1 - mask) * low_feature

        elif self.args.method == 'multiscale':
            assert img.shape[-1] == 2048, "size not right"
            all_cuts = make_crop(img)
            all_feature = []
            for cuts in all_cuts:
                preds, feats = self.get_classification(cuts)
                all_feature.append(feats)
            outs = make_one_from_crop(*all_feature, out_shape=self.args.tar_res)
            return outs 

        elif self.args.method == 'cam_multiscale':
            assert img.shape[-1] == 2048, "size not right"
            all_cuts = make_crop(img)
            all_preds = []
            all_feats = []
            for cuts in all_cuts:
                preds, feats = self.get_classification(cuts)
                all_preds.append(preds)
                all_feats.append(feats)
            classification = make_one_from_crop(*all_preds, out_shape=self.args.tar_res)
            feature = make_one_from_crop(*all_feats, out_shape=self.args.tar_res)
            x = F.interpolate(img, (self.args.res, self.args.res))
            low_feats = self.forward_dino(x)
            # high_feats = self.conv(feature)
            high_feats = F.interpolate(feature, low_feats.shape[-2:])
            mask = classification.amax(dim=1) > 20 # (b, h, w)
            mask = mask.unsqueeze(1) # (b, 1, h, w)
            mask = F.interpolate(mask.float(), low_feats.shape[-2:])
            low_feats = self.conv(low_feats)
            return mask * high_feats + (1 - mask) * low_feats

        else:
            print(f'Unknow method {self.args.method}')

 
class FPNDecoder(nn.Module):
    def __init__(self, args, num_features, n_classes):
        super(FPNDecoder, self).__init__()
        if args.linear:
            # self.classifier = nn.Sequential(
            #     nn.BatchNorm2d(num_features), 
            #     nn.Conv2d(num_features, n_classes, 1, 1, 0)
            # )
            self.classifier = nn.Conv2d(num_features, n_classes, 1, 1, 0)
        else:
            self.classifier = nn.Sequential(
                nn.BatchNorm2d(num_features), 
                DoubleConv(num_features, n_classes, 512)
            )
        
    def forward(self, feats):
        out = self.classifier(feats)
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


class DinoFeaturizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.dim = 100
        patch_size = 8
        self.patch_size = patch_size
        self.feat_type = "feat"
        arch = "vit_base"
        self.model = vit.__dict__[arch](
            patch_size=patch_size,
            num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()
        self.dropout = torch.nn.Dropout2d(p=.1)

        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768
        self.cluster1 = self.make_clusterer(self.n_feats)
        self.proj_type = "nonlinear"
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))  # ,

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            if self.feat_type == "feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.feat_type == "KK":
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))

            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        if self.proj_type is not None:
            code = self.cluster1(self.dropout(image_feat))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_feat))
        else:
            code = image_feat

        return image_feat, code
