from pyexpat import features
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from . import backbone
from module_utils import make_crop, make_one_from_crop, MSCAM, LayerFusion, make_one_from_crop_with_weights
from einops import rearrange

class PanopticFPN(nn.Module):
    def __init__(self, args):
        super(PanopticFPN, self).__init__()
        if args.arch == 'resnet18':
            out_dim = 128 
        else:
            out_dim = 256
        self.backbone = backbone.__dict__[args.arch](pretrained=args.pretrain)
        # self.fc = self.backbone.fc
        self.decoder  = FPNDecoder(args)
        
        # self.conv = nn.Conv2d(self.backbone.fc.weight.shape[-1], args.in_dim, 1)
        self.args = args
        
        self.ms_cam = MSCAM(args.in_dim, args.in_dim // 2)
        self.lf = LayerFusion([256, 512, 1024, 2048], args.in_dim, args.method.split('LF')[-1])
        self.ms_conv = nn.Conv2d(args.in_dim, args.in_dim, 1)
        self.patch_weights = nn.Conv2d(args.in_dim, 1, 1)

    def forward(self, x):
        if self.args.method == 'cam':
            x = F.interpolate(x, (self.args.res, self.args.res))
            feats = self.backbone(x)
            outs  = self.decoder(feats) 
            classification, high_feats = self.forward_classification(feats, use_feats=True)
            high_feats = self.conv(high_feats)
            high_feats = F.interpolate(high_feats, outs.shape[-2:])
            # print(f'{classification.max() = } {classification.mean() = }')
            mask = classification.amax(dim=1) > 20 # (b, h, w)
            mask = mask.unsqueeze(1) # (b, 1, h, w)
            mask = F.interpolate(mask.float(), outs.shape[-2:])
            return mask * high_feats + (1-mask) * outs
        elif self.args.method == 'aff':
            x = F.interpolate(x, (self.args.res, self.args.res))
            feats = self.backbone(x)
            outs  = self.decoder(feats) 
            classification, high_feats = self.forward_classification(feats, use_feats=True)
            high_feats = self.conv(high_feats)
            high_feats = F.interpolate(high_feats, outs.shape[-2:])
            ms_cam = self.ms_cam(high_feats + outs)
            return ms_cam * high_feats + (1 - ms_cam) * outs
        elif self.args.method == 'LFaff':
            x = F.interpolate(x, (self.args.res, self.args.res))
            feats = self.backbone(x)
            feats = [feats[f'res{i}'] for i in range(2, 6)]
            return self.lf(feats)
            
        elif self.args.method == 'multiscale':
            assert x.shape[-1] == 2048, f"size not right, requires (1024,2048), receives {x.shape[-2:]}"
            all_cuts = make_crop(x, self.args.res)
            # all_feature = []
            # for cuts in all_cuts:
            #     feature = self.get_feature(cuts)
            #     all_feature.append(feature)
            all_feats = self.backbone(all_cuts)
            all_feature = self.decoder(all_feats)
            outs = make_one_from_crop(all_feature, out_shape=self.args.tar_res)
            return self.ms_conv(outs) 

        elif self.args.method == 'multiscale_ww':
            assert x.shape[-1] == 2048, f"size not right, requires (1024,2048), receives {x.shape[-2:]}"
            all_cuts = make_crop(x, self.args.res)
            all_feature = self.backbone(all_cuts) 
            all_feature = self.decoder(all_feature)
            weights = self.patch_weights(all_feature)
            outs = make_one_from_crop_with_weights(all_feature, weights, out_shape=self.args.tar_res)
            return self.ms_conv(outs) 
        elif self.args.method == 'cam_multiscale':
            assert x.shape[-1] == 2048, "size not right"
            all_cuts = make_crop(x)
            all_preds = []
            all_feats = []
            for cuts in all_cuts:
                preds, feats = self.get_classification(cuts)
                all_preds.append(preds)
                all_feats.append(feats)
            classification = make_one_from_crop(*all_preds, out_shape=self.args.tar_res)
            feature = make_one_from_crop(*all_feats, out_shape=self.args.tar_res)
            x = F.interpolate(x, (self.args.res, self.args.res))

            feats = self.backbone(x)
            low_feats = self.decoder(feats) 

            high_feats = self.conv(feature)
            high_feats = F.interpolate(high_feats, low_feats.shape[-2:])
            mask = classification.amax(dim=1) > 20# (b, h, w)
            mask = mask.unsqueeze(1) # (b, 1, h, w)
            mask = F.interpolate(mask.float(), low_feats.shape[-2:])
            return mask * high_feats + (1 - mask) * low_feats

        elif self.args.method == '' or self.args.method.startswith('layer'):
            feats = self.backbone(x)
            outs  = self.decoder(feats) 
            return outs 
            
        else:
            print(f'Unknow method {self.args.method}')

    def get_classification(self, cuts):        
        predictions = []
        all_features = []
        for cut in cuts:
            image = F.interpolate(cut, (self.args.res, self.args.res)).to(cut.device)
            # with torch.no_grad():
            classifications, features = self.forward_classification(image) # (b, n, h, w) n = 1000
            # classifications[classifications < 35] = 0
            predictions.append(classifications)
            all_features.append(features)
        return predictions, all_features

    def get_feature(self, cuts):
        features = []
        for cut in cuts:
            image = F.interpolate(cut, (self.args.res, self.args.res)).to(cut.device)
            # with torch.no_grad():
            feats = self.backbone(image)
            outs  = self.decoder(feats) 
            features.append(outs)
        return features  

    def forward_classification(self, x, use_feats=False):
        if not use_feats:
            x = self.backbone(x)
        x = x['res5'] # (b, c, h, w)
        b, c, h, w = x.shape 
        rearr_x = rearrange(x, 'b c h w -> (b h w) c')
        rearr_classification = self.fc(rearr_x) # ((bhw)c -> (bhw)1000)
        classification = rearrange(rearr_classification, '(b h w) n -> b n h w', b=b, h=h, w=w)
        return classification, x
        

class FPNDecoder(nn.Module):
    def __init__(self, args):
        super(FPNDecoder, self).__init__()
        self.args = args 
        if args.arch == 'resnet18':
            self.mfactor = 1
            self.out_dim = args.in_dim 
        else:
            self.mfactor = 4
            self.out_dim = args.in_dim
        self.tar_res = args.tar_res
        self.layer4 = nn.Conv2d(512*self.mfactor//8, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(512*self.mfactor//4, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(512*self.mfactor//2, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.layer1 = nn.Conv2d(512*self.mfactor, self.out_dim, kernel_size=1, stride=1, padding=0)
        torch.nn.init.kaiming_uniform_(self.layer1.weight)
        torch.nn.init.kaiming_uniform_(self.layer2.weight)
        torch.nn.init.kaiming_uniform_(self.layer3.weight)
        torch.nn.init.kaiming_uniform_(self.layer4.weight)
        
        
    def forward(self, x):
        if self.args.method == 'layer3':
            return F.interpolate(self.layer3(x['res3']), self.tar_res, mode='bilinear')
        elif self.args.method == 'layer2':
            return F.interpolate(self.layer4(x['res2']), self.tar_res, mode='bilinear')
        elif self.args.method == 'layer4':
            return F.interpolate(self.layer2(x['res4']), self.tar_res, mode='bilinear')
        o1 = self.layer1(x['res5'])
        o2 = self.upsample_add(o1, self.layer2(x['res4']))
        o3 = self.upsample_add(o2, self.layer3(x['res3']))
        o4 = self.upsample_add(o3, self.layer4(x['res2']))
        return F.interpolate(o4, self.tar_res, mode='bilinear')
        # return o4

    def upsample_add(self, x, y):
        _, _, H, W = y.size()

        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y 



