import matplotlib.pyplot as plt 
from PIL import Image
import cv2 
import os 


def draw_comparison(output_name, image_list, descriptions, filenames, city=True):
    assert len(descriptions) == len(filenames), 'len(descriptions) not match len(filenames)'
    nrow = len(filenames)
    ncol = len(image_list)
    fig, ax = plt.subplots(ncol, nrow, figsize=(nrow*4, ncol*4))
    for i in range(nrow):
        
        for j in range(ncol):
            stri = (4 - len(str(image_list[j]))) * '0' + str(image_list[j])
            if city:
                fn = f'./new_dir/{stri}_{filenames[i]}.png' if 'seg' in filenames[i] else f'./draw_image_result/{stri}_{filenames[i]}.png' 
            else:
                fn = f'./draw_image_result_coco/{stri}_{filenames[i]}.png' 
            img = Image.open(fn)
            # if 'multiscale' in fn:
            #     width, height = img.size 
            #     newsize = (width * 2, height) 
            #     img = img.resize(newsize)
            #     # width, height = img.size 
            #     left = width // 2
            #     top = 0
            #     right = width * 3 // 2
            #     bottom = height
            #     img = img.crop((left, top, right, bottom)) 
            ax[j, i].imshow(img)
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
     
        # ax[0, j].set_xlabel(descriptions[i], fontsize=16)
        ax[0, i].set_title(descriptions[i], fontsize=36, fontweight='bold')
    # plt.title()
    plt.tight_layout() 
    plt.savefig(output_name, dpi=400)
            
            
if __name__ == "__main__":
    
    
    # filenames = ['img', 'lbl', 'seg_linear_resnet50_320_', 'seg_resnet50_320_multiscale', 'seg_swinv2_384_dino_multiscale', 'seg_swinv2_384_dino']
    # descriptions = ['Image', "Label", "resnet50", "resnet50 + MS", "Dino ViT", "Dino ViT + MS"]
    # # image_list = [515, 530, 2644, 896, 2029, 566, 2742]
    # # image_list = [2570, 905, 643]
    # layer_fusion = [643, 49, 120]
    # scale_fusion = [25, 104, 107]
    
    
    # ## draw layer fusion
    # # layer_fusion = [3109, 3373, 515, 530, 2742] # [3109, 3373, 515, 530, 2742]
    # layer_fusion = [643, 53, 57] # (bkup  36, 34, )
    # filenames = ['img', 'lbl', 'seg_picie_320_', 'seg_resnet18_320_', 'seg_stego_320_', 'seg_swinv2_384_dino' ]
    # descriptions = ['Image', "Label", "PiCIE", "ResNet18 + LF", "STEGO", "SwinV2 + LF"]
    # draw_comparison('./seg_comparisons/layer_fusion1.png', layer_fusion, descriptions, filenames)
    # # draw_comparison('./seg_comparisons/layer_fusion.png', layer_fusion, descriptions[:2] + ['ResNet', "ResNet w/ LF", "ViT", "ViT w/ LF"], filenames[:2] + [filenames[i] for i in [3, 2, 4, 5]])
    
    # # ## draw scale fusion
    # scale_fusion = [25,  516, 2277,  ] # [ 643, 25,  516, 2277,  ]
    # filenames = ['img', 'lbl', 'seg_picie_320_', 'seg_resnet50_320_multiscale', 'seg_stego_320_', 'seg_swinv2_384_dino_multiscale' ]
    # descriptions = ['Image', "Label", "PiCIE", "ResNet18 + MS", "STEGO", "SwinV2 + MS"]
    # draw_comparison('./seg_comparisons/scale_fusion1.png', scale_fusion, descriptions, filenames)
    # # draw_comparison('./seg_comparisons/scale_fusion.png', scale_fusion, descriptions[:2] + ['ResNet', "ResNet w/ MS", "ViT", "ViT  w/ MS"], filenames)
    
    # ## draw both fusion
    # both_fusion = [498,1366,3008,611, 1360]
    # filenames = ['img', 'seg_picie_320_', 'seg_stego_320_',  'seg_resnet18_320_', 'lbl']
    # descriptions = ['Image', "PiCIE", "STEGO", "Ours", "Label"]
    # draw_comparison('./seg_comparisons/both_fusion.png', both_fusion, descriptions, filenames)
    
    # # /home/zhulifu/unsup_seg/trials_unsupervised_segmentation/draw_image_result/0035_img.png
    
    # ## draw coco both fusion
    # coco_both = [4780, 468, 2880, 4102, 4884 ]
    # filenames = ['img', 'seg_coco_picie_320_', 'seg_coco_stego_320_', 'seg_coco_resnet50_320_', 'lbl',] #  'seg_coco_swinv2_384_swin_only_LFadd' 
    # descriptions = ['Image', "PiCIE",  "STEGO", "Ours", "Label", ] # "SwinV2 + LF", 
    # draw_comparison('./seg_comparisons/coco_both_fusion1.png', coco_both, descriptions, filenames, city=False)
    
    
    
    ## draw scale examples
    examples = [62, 80, 82]
    filenames = ['img', 'lbl', 'seg_resnet18_320_',] #  'seg_coco_swinv2_384_swin_only_LFadd' 
    # descriptions = ['Image', "PiCIE",  "STEGO", "Ours", "Label", ] # "SwinV2 + LF", 
    descriptions = ['Image', "Label", "Seg."] 
    draw_comparison('./seg_comparisons/scale_example.png', examples, descriptions, filenames, city=True)
