import einops
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from . import backbone
from .dino import vision_transformer as vit
from .swin_transformer_v2 import SwinTransformerV2
from einops import rearrange
import matplotlib.pyplot as plt 
from module_utils import LayerFusion, make_crop, make_one_from_crop, MSCAM, LayerFusion, make_one_from_crop_with_weights


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

        self.device = args.device
        self.args = args 
        self.conv = nn.Conv2d(768, 1024, 1)
        torch.nn.init.kaiming_uniform_(self.conv.weight)
        self.mscam = MSCAM(1024, 1024 // 2)
        if "LF" in args.method:
            if "swin_only" in args.method:
                # self.lf = LayerFusion([256, 512, 1024, 1024], args.in_dim, args.method.split('LF')[-1])
                self.decoder = FPNDecoder(args)
            else:
                self.lf = LayerFusion([768, 256, 512, 1024, 1024], args.in_dim, args.method.split('LF')[-1])
        if args.method == 'dino_multiscale_ww':
            self.patch_weights = nn.Conv2d(768, 1, 1)

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
    
    def get_dino(self, cuts):
        
        all_features = []
        for cut in cuts:
            image = F.interpolate(cut, (384, 384)).to(self.device)
            with torch.no_grad():
                feat = self.forward_dino(image) # (b, n, h, w) n = 1000
            all_features.append(feat)
        return all_features
        

    def forward_swin(self, x, return_feats=False):
        # if return_feats:
        feats = []
        net = self.swin 
        x = x.to(self.device)
        with torch.no_grad():
            x = net.patch_embed(x)
            if net.ape:
                x = x + net.absolute_pos_embed
            x = net.pos_drop(x)
            for layer in net.layers:
                x = layer(x)
                if return_feats:
                    feats.append(x)
            org_x = net.norm(x)  # B L C
            feats.append(x)
            rets = []
            for x in feats:
                b, l, c = x.shape
                h = int(l ** 0.5)
                w = h
                rearr_x = rearrange(x, 'b (h w) c -> b c h w', b=b, h=h, w=w)
                rets.append(rearr_x)
            if return_feats:
                return rets 
            else:
                return org_x, rets[-1]
            
    def forward_classification(self, x):
        net = self.swin 
        x, feature = self.forward_swin(x)
        b, c, h, w = feature.shape
        rearr_x = rearrange(x, 'b l c -> (b l) c') 
        rearr_classification = net.head(rearr_x) # ((bl), c) -> ((bl), n_classes)
        classification = rearrange(rearr_classification, '(b h w) n -> b n h w', b=b, h=h, w=w)
        return classification, feature
    
    def get_dino_features(self, img, n=1):
        b, c, h, w = img.shape 
        chunk_size = 4 
        res = []
        if b > chunk_size:
            i = 0
            while i < b:
                img_slice = img[i: i + chunk_size]
                res.append(self.forward_dino(img_slice))
                i += chunk_size
            return torch.cat(res, dim=0)
        else:
            return self.forward_dino(img)
    
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
        # image_feat = F.interpolate(image_feat, self.args.tar_res, mode='bilinear')
        return image_feat
    
    def forward(self, img):
        if self.args.method == 'dino':
            feat = self.forward_dino(img)
            x = self.conv(feat)
            return x
        elif self.args.method == 'cam':
            low_feature = self.forward_dino(img)

            classification, high_feature = self.forward_classification(img)
            mask = classification.amax(dim=1) > 20 # (b, h, w)
            mask = mask.unsqueeze(1) # (b, 1, h, w)
            mask = F.interpolate(mask.float(), low_feature.shape[-2:])
            
            high_feature = F.interpolate(high_feature, low_feature.shape[-2:])
            low_feature = self.conv(low_feature)
            fusion = mask * high_feature + (1 - mask) * low_feature
            return F.interpolate(fusion, self.args.tar_res, mode='bilinear')
        elif self.args.method.startswith('swin_only_LF') :
            feats = self.forward_swin(img, return_feats=True)
            # fusion = self.lf(feats[:-1])
            # print(f'{len(feats) = }')
            # fusion = self.decoder(feats[:-1])
            return feats 
        elif self.args.method == 'swin_dino_LFaff':
            feats = self.forward_swin(img, return_feats=True) # img (b, 3, 384, 384) feats[ (b, 256, 48, 48), (b, 512, 24, 24), (b, 1024, 12, 12), (b, 1024, 12, 12), (b, 1024, 12, 12)] -2, -1 same.
            dinofeat = self.forward_dino(img) #(b, 768, 160, 160)
            feats = [dinofeat] + feats[:-1]  # [dinofeat, f0, f1, f2, f3]
            fusion = self.lf(feats)
            # return dinofeat
            return F.interpolate(fusion, self.args.tar_res, mode='bilinear')
            
        elif self.args.method == 'aff_nodino':
            low_feature = self.forward_dino(img)

            classification, high_feature = self.forward_classification(img)
            high_feature = F.interpolate(high_feature, low_feature.shape[-2:])
            low_feature = self.conv(low_feature)
            mask = self.mscam(high_feature + low_feature)
            fusion = mask * high_feature + (1 - mask) * low_feature
            return F.interpolate(fusion, self.args.tar_res, mode='bilinear')

        elif self.args.method == 'aff_dino':
            low_feature = self.forward_dino(img)
            feats = self.forward_swin(img, True)
            classification, high_feature = self.forward_classification(img)
            high_feature = F.interpolate(high_feature, low_feature.shape[-2:])
            low_feature = self.conv(low_feature)
            mask = self.mscam(high_feature + low_feature)
            fusion = mask * high_feature + (1 - mask) * low_feature
            return F.interpolate(fusion, self.args.tar_res, mode='bilinear')

        elif self.args.method == 'multiscale':
            assert img.shape[-1] == 2048, "size not right"
            all_cuts = make_crop(img)
            all_feature = []
            for cuts in all_cuts:
                preds, feats = self.get_classification(cuts)
                all_feature.append(feats)
            outs = make_one_from_crop(*all_feature, out_shape=self.args.tar_res)
            outs = self.conv(outs)
            return outs 
        elif self.args.method == 'dino_multiscale':
            assert img.shape[-1] == 2048, "size not right"
            cuts = make_crop(img, self.args.res)
            feats = self.get_dino_features(cuts)
            outs = make_one_from_crop(feats, out_shape=self.args.tar_res)
            outs = self.conv(outs)
            return outs 
        elif self.args.method == 'dino_multiscale_ww':
            assert img.shape[-1] == 2048, "size not right"
            cuts = make_crop(img, self.args.res)
            feats = self.get_dino_features(cuts)
            weights = self.patch_weights(feats)
            outs = make_one_from_crop_with_weights(feats, weights, out_shape=self.args.tar_res)
            outs = self.conv(outs)
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
            fusion = mask * high_feats + (1 - mask) * low_feats
            return F.interpolate(fusion, self.args.tar_res, mode='bilinear')
        else:
            print(f'Unknow method {self.args.method}')

class FPNDecoder(nn.Module):
    def __init__(self, args):
        super(FPNDecoder, self).__init__()
        if args.arch == 'resnet18':
            self.mfactor = 1
            self.out_dim = args.in_dim 
        else:
            self.mfactor = 4
            self.out_dim = args.in_dim
        self.tar_res = args.tar_res
        # 256, 512, 1024, 1024
        self.layer4 = nn.Conv2d(256, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(512, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(1024, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.layer1 = nn.Conv2d(1024, self.out_dim, kernel_size=1, stride=1, padding=0)
        torch.nn.init.kaiming_uniform_(self.layer1.weight)
        torch.nn.init.kaiming_uniform_(self.layer2.weight)
        torch.nn.init.kaiming_uniform_(self.layer3.weight)
        torch.nn.init.kaiming_uniform_(self.layer4.weight)
        
        
    def forward(self, feats):
        # print(f'{len(feats)}')
        # for feat in feats:
        #     print(f'{feat.shape = }')
        x1, x2, x3, x4 = feats
        o1 = self.layer1(x4)
        o2 = self.upsample_add(o1, self.layer2(x3))
        o3 = self.upsample_add(o2, self.layer3(x2))
        o4 = self.upsample_add(o3, self.layer4(x1))
        F.interpolate(o4, self.tar_res, mode='bilinear')
        return o4

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
        # self.model.cuda()
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
        # self.model.eval()
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
