import einops
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import List

class MSCAM(nn.Module):
    def __init__(self, inchannel: int, midchannel: int):
        super().__init__()
        self.channel_conv_d = nn.Conv2d(inchannel, midchannel, 1)
        self.spatial_conv_d = nn.Conv2d(inchannel, midchannel, 1)
        self.channel_conv_u = nn.Conv2d(midchannel, inchannel, 1)
        self.spatial_conv_u = nn.Conv2d(midchannel, inchannel, 1)
        
    def forward(self, x):
        channel_attn = F.adaptive_avg_pool2d(x, 1)
        channel_attn = self.channel_conv_d(channel_attn)
        channel_attn = F.relu(channel_attn, inplace=True)
        channel_attn = self.channel_conv_u(channel_attn)
        
        spatial_attn = self.spatial_conv_d(x)
        spatial_attn = F.relu(spatial_attn, inplace=True)
        spatial_attn = self.spatial_conv_u(spatial_attn)
        fusion = channel_attn.expand_as(spatial_attn) + spatial_attn 
        fusion = F.sigmoid(fusion)
        return x * fusion 
        

class LayerFusion(nn.Module):
    def __init__(self, in_channels: List[int], out_dim: int, fusion_mode: str):
        '''
            @params in_channels: list of ints which indicates each layer in_channel, [256, 512, 1024, 2048] for resnet.
            @params out_dim: integer that indicates final layer output dimension.
            @params fusion_mode: str, possible choices 'aff', 'cat', 'add'. 
        '''
        super().__init__()   
        self.in_channels = in_channels
        self.out_dim = out_dim 
        self.fusion_mode = fusion_mode 
        
        self.convs = []
        for i in range(len(in_channels) - 1):
            outc = in_channels[i]
            inc = in_channels[i+1]
            self.convs.append(nn.Conv2d(inc, outc, 1))
                
        if fusion_mode == 'aff':
            self.ms_cams = []
            for i in range(len(in_channels) - 1):
                inc = in_channels[i]
                self.ms_cams.append(MSCAM(inc, inc // 2))
        
        if fusion_mode == 'cat':
            self.convs = []
            for i in range(len(in_channels) - 1):
                outc = in_channels[i]
                inc = in_channels[i+1]
                self.convs.append(nn.Conv2d(inc + outc, outc, 1))
                
    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y 
    
    def forward(self, feats):
        fusion_feats = feats[-1]
        for i in range(len(feats) - 2, -1, -1):
            if self.fusion_mode == 'aff':
                fusion_feats = self.convs[i](fusion_feats)
                fusion_feats = F.interpolate(fusion_feats, feats[i].shape[-2:], mode='bilinear')
                mask = self.ms_cams[i](fusion_feats + feats[i])
                fusion_feats = mask * fusion_feats + (1 - mask) * feats[i]

            elif self.fusion_mode == 'add':
                fusion_feats = self.convs[i](fusion_feats)
                fusion_feats = F.interpolate(fusion_feats, feats[i].shape[-2:], mode='bilinear')
                fusion_feats = fusion_feats + feats[i]
                
            elif self.fusion_mode == 'cat':
                fusion_feats = F.interpolate(fusion_feats, feats[i].shape[-2:], mode='bilinear')
                fusion_feats = torch.cat([fusion_feats, feats[i]], dim=1)
                fusion_feats = self.convs[i](fusion_feats)
                
        return fusion_feats 
        
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
    cut_range3 = (0, 1024)
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
    # cut_range3 = (0, 800)
    cut_range3 = (0, 1024)
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
        resized_cut = F.interpolate(cuts1[i], get_int_tuple(cut1_size * h_scale, cut1_size * w_scale), mode='bilinear')
        b,c,h,w = resized_cut.shape
        patch_weight = torch.ones_like(resized_cut)
        out[:,:, resized_h_range[0]:resized_h_range[0]+h, i * resized_w_stride1: i * resized_w_stride1 + w] += resized_cut
        weight[:,:, resized_h_range[0]:resized_h_range[0]+h, i * resized_w_stride1: i * resized_w_stride1 + w] += patch_weight

    resized_h_range = get_int_tuple(cut_range2[0] * h_scale, cut_range2[1] * h_scale)
    # resized_w_stride = int(cut_stride2 * w_scale)
    
    for i in range(cut_num2):
        # cuts1.append(img[:, :, cut_range1[0]:cut_range1[1], i*cut_stride1 : i*cut_stride1 + cut_range1[1] - cut_range1[0]])
        resized_cut = F.interpolate(cuts2[i], get_int_tuple(cut2_size * h_scale, cut2_size * w_scale), mode='bilinear')
        b,c,h,w = resized_cut.shape
        patch_weight = torch.ones_like(resized_cut)
        out[:,:, resized_h_range[0]:resized_h_range[0] + h, i * resized_w_stride2: i * resized_w_stride2 + w] += resized_cut
        weight[:,:, resized_h_range[0]:resized_h_range[0] + h, i * resized_w_stride2: i * resized_w_stride2 + w] += patch_weight
    
    # resized_h_range = get_int_tuple(cut_range3[0] * h_scale, cut_range3[1] * h_scale)
    # cut3_size = (cut_range3[1]-cut_range3[0])
    # for i in range(1):
    #     resized_cut = F.interpolate(cuts3[i], get_int_tuple( cut3_size * h_scale, out_shape))
    #     patch_weight = torch.ones_like(resized_cut)
    #     out[:,:, resized_h_range[0]:resized_h_range[1], :] += resized_cut
    #     weight[:,:, resized_h_range[0]:resized_h_range[1], :] += patch_weight
    
    weight[weight == 0] = 1
    out = out / weight 
    return out



if __name__ == "__main__":
    feats = [
        torch.randn(1, 256, 160, 160),
        torch.randn(1, 512, 80, 80),
        torch.randn(1, 1024, 40, 40)
        ]
    for mode in ['aff', 'add', 'cat']:    
        lf = LayerFusion([256, 512, 1024], 256, mode)
        out = lf.forward(feats)
        print(out.shape)