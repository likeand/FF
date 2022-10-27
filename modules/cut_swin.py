from pickle import TRUE
import einops
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from . import backbone
from .dino import vision_transformer as vit
from .swin_transformer_v2 import SwinTransformerV2
from einops import rearrange

def make_crop(img):
    '''
        @params img: Tensor of shape (b, 3, 1024, 2048)
        @returns img: Tensors of crop.
    '''
    ## 多尺度，从0，到800截图4张，从100到600截图6张，有重叠的。
    cut_num1 = 4
    cut_num2 = 6
    cut_range1 = (0, 800)
    cut_range2 = (100, 600)
    cut_range3 = (0, 800)
    cut_stride1 = (2048 - (cut_range1[1] - cut_range1[0])) // (cut_num1 - 1)
    cut_stride2 = (2048 - (cut_range2[1] - cut_range2[0])) // (cut_num2 - 1)

    cuts1 = []
    for i in range(cut_num1):
        cuts1.append(img[:, :, cut_range1[0]:cut_range1[1], i*cut_stride1 : i*cut_stride1 + cut_range1[1] - cut_range1[0]])

    cuts2 = []
    for i in range(cut_num2):
        cuts2.append(img[:, :, cut_range2[0]:cut_range2[1], i*cut_stride2 : i*cut_stride2 + cut_range2[1] - cut_range2[0]])
    
    cuts3 = [img[:,:,cut_range3[0]:cut_range3[1]]]
    return cuts1, cuts2, cuts3

def get_int_tuple(x, y):
    return (int(x), int(y))

def make_one_from_crop(cuts1, cuts2, cuts3, out_shape=384):

    cut_num1 = 4
    cut_num2 = 6
    cut_range1 = (0, 800)
    cut_range2 = (100, 600)
    cut_range3 = (0, 800)
    cut_stride1 = (2048 - (cut_range1[1] - cut_range1[0])) // (cut_num1 - 1)
    cut_stride2 = (2048 - (cut_range2[1] - cut_range2[0])) // (cut_num2 - 1)

    cut1_size = cut_range1[1] - cut_range1[0]
    cut2_size = cut_range2[1] - cut_range2[0]
    origin_size = (1024, 2048)
    target_size = out_shape

    w_scale = out_shape / origin_size[1]
    h_scale = out_shape / origin_size[0]  

    B, C, H, W = cuts3[0].shape
    out = torch.zeros((B, C, out_shape, out_shape), device=cuts3[0].device)
    weight = torch.zeros_like(out)

    resized_h_range = get_int_tuple(cut_range1[0] * h_scale, cut_range1[1] * h_scale)
    # resized_w_stride = int(cut_stride1 * w_scale)
    resized_w_stride1 = int (out_shape - (cut_range1[1] - cut_range1[0]) * w_scale) // (cut_num1 - 1)
    resized_w_stride2 = int (out_shape - (cut_range2[1] - cut_range2[0]) * w_scale) // (cut_num2 - 1)
    for i in range(cut_num1):
        # cuts1.append(img[:, :, cut_range1[0]:cut_range1[1], i*cut_stride1 : i*cut_stride1 + cut_range1[1] - cut_range1[0]])
        resized_cut = F.interpolate(cuts1[i], get_int_tuple(cut1_size * h_scale, cut1_size * w_scale))
        b,c,h,w = resized_cut.shape
        patch_weight = torch.ones_like(resized_cut)
        out[:,:, resized_h_range[0]:resized_h_range[0]+h, i * resized_w_stride1: i * resized_w_stride1 + w] += resized_cut
        weight[:,:, resized_h_range[0]:resized_h_range[0]+h, i * resized_w_stride1: i * resized_w_stride1 + w] += patch_weight

    resized_h_range = get_int_tuple(cut_range2[0] * h_scale, cut_range2[1] * h_scale)
    # resized_w_stride = int(cut_stride2 * w_scale)
    
    for i in range(cut_num2):
        # cuts1.append(img[:, :, cut_range1[0]:cut_range1[1], i*cut_stride1 : i*cut_stride1 + cut_range1[1] - cut_range1[0]])
        resized_cut = F.interpolate(cuts2[i], get_int_tuple(cut2_size * h_scale, cut2_size * w_scale))
        b,c,h,w = resized_cut.shape
        patch_weight = torch.ones_like(resized_cut)
        out[:,:, resized_h_range[0]:resized_h_range[0] + h, i * resized_w_stride2: i * resized_w_stride2 + w] += resized_cut
        weight[:,:, resized_h_range[0]:resized_h_range[0] + h, i * resized_w_stride2: i * resized_w_stride2 + w] += patch_weight
    
    resized_h_range = get_int_tuple(cut_range3[0] * h_scale, cut_range3[1] * h_scale)
    cut3_size = (cut_range3[1]-cut_range3[0])
    for i in range(1):
        resized_cut = F.interpolate(cuts3[i], get_int_tuple( cut3_size * h_scale, out_shape))
        patch_weight = torch.ones_like(resized_cut)
        out[:,:, resized_h_range[0]:resized_h_range[1], :] += resized_cut
        weight[:,:, resized_h_range[0]:resized_h_range[1], :] += patch_weight
    
    weight[weight == 0] = 1
    out = out / weight 
    return out

class PanopticFPN(nn.Module):
    def __init__(self, args):
        super(PanopticFPN, self).__init__()
        swinv2 = True
        if swinv2:
            self.backbone = SwinTransformerV2(img_size=384, embed_dim=128, depths=[ 2, 2, 18, 2 ], num_heads=[ 4, 8, 16, 32 ], window_size=24, num_classes=1000)
            self.num_features = self.backbone.num_features
            RESUME = f'{args.model_dir}/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth'
            checkpoint = torch.load(RESUME, map_location=args.device)
            msg = self.backbone.load_state_dict(checkpoint['model'], strict=False)
            self.backbone.to(args.device)
            print(f'loaded stated dict, \n{msg}')
        else:
            arch = args.arch
            self.patch_size = 8
            self.backbone = vit.__dict__[arch](
                patch_size=8,
                num_classes=0
            )
            if args.arch_local_save is not None:
                    self.backbone.load_state_dict(torch.load(args.arch_local_save))
                    print(f'loaded state dict from {args.arch_local_save}')
        self.obj_decoder  = FPNDecoder(args, self.num_features, args.obj_classes)
        self.obj_decoder.to(args.device)
        self.things_decoder  = FPNDecoder(args, self.num_features, args.things_classes)
        self.things_decoder.to(args.device)
        self.device = 'cuda'

    def get_classification(self, cuts):        
        predictions = []
        for cut in cuts:
            image = F.interpolate(cut, (384, 384))
            with torch.no_grad():
                classifications = self.forward_classification(image) # (b, n, h, w) n = 1000
            classifications[classifications < 35] = 0
            predictions.append(classifications)
        return predictions

    def forward_classification(self, x):
        net = self.backbone
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
            rearr_x = rearrange(x, 'b l c -> (b l) c') 
            rearr_classification = net.head(rearr_x) # ((bl), c) -> ((bl), n_classes)
            classification = rearrange(rearr_classification, '(b h w) n -> b n h w', b=b, h=h, w=w)
            return classification
    
    def forward_features(self, x):
        x = x.to(self.device)
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
            return feats
    
    def forward_lbl(self, img, eval=False):
        # feats = self.backbone(x)
        with torch.no_grad():
            all_cuts = make_crop(img)
            all_preds = []
            for cuts in all_cuts:
                preds = self.get_classification(cuts)
                all_preds.append(preds)
            lbl = make_one_from_crop(*all_preds, out_shape=48)
            lbl_view = rearrange(lbl, 'b c h w -> (b h w) c')
            lbl_view_mask = lbl_view.amax(1) > 0
            # if not eval:
            #     return lbl_view[lbl_view_mask]
            # # x, feats = self.forward_features(img)
            # # lbl.shape (b,c,h,w)
            # else:
            if True:
                return lbl
    
    def forward_object_cluster(self, img):
        _, _, h, w = img.shape 
        assert h == 384 and w == 384, 'wrong size image' 
        with torch.no_grad():
            feats = self.forward_features(img) 
        out = self.obj_decoder(feats[0])
        return out 
    
    def forward_things_cluster(self, img):
        _, _, h, w = img.shape 
        assert h == 384 and w == 384, 'wrong size image' 
        with torch.no_grad():
            feats = self.forward_features(img) 
        out = self.things_decoder(feats[0])
        return out 

    def forward_seg(self, img):
        _, _, h, w = img.shape 
        assert h == 384 and w == 384, 'wrong size image' 
        with torch.no_grad():
            feats = self.forward_features(img) 
        objects = self.things_decoder(feats[0])
        things = self.obj_decoder(feats[0])
        things[objects > 0] = objects
        return things 
 
class FPNDecoder(nn.Module):
    def __init__(self, args, num_features, n_classes):
        super(FPNDecoder, self).__init__()
        if args.linear:
            self.classifier = nn.Sequential(
                nn.BatchNorm2d(num_features), 
                nn.Conv2d(num_features, n_classes, 1, 1, 0)
            )
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