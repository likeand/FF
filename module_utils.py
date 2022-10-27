import einops
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

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
