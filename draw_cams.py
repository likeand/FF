from utils import *
from module_utils import make_crop, make_one_from_crop
import os 
import numpy as np 
import torch 
import torch.nn as nn 
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
import argparse
from tqdm import tqdm 
from data.cityscapes_eval_dataset import EvalCityscapesRAW
from torchcam.methods import SmoothGradCAMpp, XGradCAM, LayerCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image, resize
from torchvision.models.resnet import *
from torchvision.models.vision_transformer import vit_b_16
from modules.dino.vision_transformer import vit_base
from modules.swin_transformer_v2 import SwinTransformerV2
from torchvision.models import swin_v2_b
from PIL import Image

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="/home/zhulifu/unsup_seg/STEGO-master/seg_dataset/cityscapes")
    parser.add_argument('--save_root', type=str, default="/home/zhulifu/unsup_seg/train_out")
    parser.add_argument('--model_dir', type=str, default="/home/zhulifu/unsup_seg/STEGO-master/models")
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--restart_path', type=str)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed for reproducability.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers.')
    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--num_epoch', type=int, default=10) 
    parser.add_argument('--repeats', type=int, default=0)  
    parser.add_argument('--linear', type=bool, default=True)
    # Train. 
    
    ## arch to choose:
    ## resnet18, resnet50, dino, swinv2
    arch = 'swinv2'
    parser.add_argument('--arch', type=str, default=arch)
    parser.add_argument('--arch_local_save', type=str, default="/data0/zx_files/models/mae_visualize_vit_large.pth")  
    parser.add_argument('--pretrain', action='store_true', default=True)
    ## res to choose: 320, 384, 640
    RES = 384 if 'swin' in arch else 320 if 'resnet' in arch else 224
    parser.add_argument('--res', type=int, default=RES, help='Input size.')
    # parser.add_argument('--res1', type=int, default=RES, help='Input size scale from.')
    # parser.add_argument('--res2', type=int, default=RES, help='Input size scale to.')
    parser.add_argument('--tar_res', type=int, default=40, help='Output Feature size.')
    
    ## methods to choose:
    ## cam, multiscale, cam_multiscale
    parser.add_argument('--method', type=str, default='multiscale')
    parser.add_argument('--batch_size_cluster', type=int, default=256)
    parser.add_argument('--batch_size_train', type=int, default=4)
    parser.add_argument('--batch_size_test', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optim_type', type=str, default='Adam')
    parser.add_argument('--num_init_batches', type=int, default=30)
    parser.add_argument('--num_batches', type=int, default=1)
    parser.add_argument('--kmeans_n_iter', type=int, default=20)
    parser.add_argument('--in_dim', type=int, default=1024)
    parser.add_argument('--X', type=int, default=80)

    # Loss. 
    parser.add_argument('--metric_train', type=str, default='cosine')   
    parser.add_argument('--metric_test', type=str, default='cosine')
    parser.add_argument('--K_cluster', type=int, default=27) # number of clusters, which will be further classified into K_train
    parser.add_argument('--K_train', type=int, default=27) # COCO Stuff-15 / COCO Thing-12 / COCO All-27
    parser.add_argument('--K_test', type=int, default=27) 
    parser.add_argument('--obj_classes', type=int, default=27) 
    parser.add_argument('--things_classes', type=int, default=27) 
    
    parser.add_argument('--no_balance', action='store_true', default=False)
    parser.add_argument('--mse', action='store_true', default=False)

    # Dataset. 
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--equiv', action='store_true', default=False)
    parser.add_argument('--min_scale', type=float, default=0.5)
    parser.add_argument('--stuff', action='store_true', default=False)
    parser.add_argument('--thing', action='store_true', default=False)
    parser.add_argument('--jitter', action='store_true', default=False)
    parser.add_argument('--grey', action='store_true', default=False)
    parser.add_argument('--blur', action='store_true', default=False)
    parser.add_argument('--h_flip', action='store_true', default=False)
    parser.add_argument('--v_flip', action='store_true', default=False)
    parser.add_argument('--random_crop', action='store_true', default=False)
    parser.add_argument('--val_type', type=str, default='train')
    parser.add_argument('--version', type=int, default=7)
    parser.add_argument('--fullcoco', action='store_true', default=False)

    # Eval-only
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_path', type=str)

    # Cityscapes-specific.
    parser.add_argument('--cityscapes', action='store_true', default=True)
    parser.add_argument('--label_mode', type=str, default='gtFine')
    parser.add_argument('--long_image', action='store_true', default=False)
    # parser.add_argument('--model_dir', type=str, default='/data0/zx_files/models')
 
    return parser.parse_args()

from commons import get_model

import matplotlib.pyplot as plt 
import cv2
def to_tuple(integer):
    return (1, 3, integer, integer)

def test_metrics_on_val(args, model, dataloader, prefix='test_on_val'):
    prefix = prefix
    # model.eval()
    if args.arch.startswith('resnet'):
        # backbone = model.backbone
        backbone = eval(args.arch)(pretrained=True).cuda()
        cam_extractor = SmoothGradCAMpp(backbone, target_layer=backbone.layer4, input_shape=to_tuple(args.res))
    elif args.arch == 'dino':
        backbone = vit_b_16(pretrained=True).cuda()
        
        # backbone = vit_base(patch_size=8)
        # dic = torch.load(f'{args.model_dir}/cityscapes_vit_base_1.ckpt')['state_dict']
        # nd = {}
        # for k in dic:
        #     if 'net.model.' in k:
        #         key = k.split('net.model.')[-1]
        #         nd[key] = dic[k]
        # backbone.load_state_dict(nd)
        # backbone.cuda()
        # backbone = model.dino.model
        # cam_extractor = SmoothGradCAMpp(backbone, target_layer=[backbone.patch_embed], input_shape=to_tuple(args.res))
        # cam_extractor = LayerCAM(backbone, target_layer=[backbone.norm, backbone.patch_embed], input_shape=to_tuple(args.res))
        cam_extractor = LayerCAM(backbone, backbone.conv_proj, input_shape=to_tuple(args.res))
    elif args.arch == 'swinv2':
        backbone = swin_v2_b(pretrained=True).cuda()
        # backbone = model.swin
        cam_extractor = LayerCAM(backbone, target_layer=[backbone.features[-1], backbone.norm], input_shape=to_tuple(args.res))
    else:
        cam_extractor = None 
        
    unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    save_cam_path = f'cams/{prefix}'
    if not os.path.exists('cams'):
        os.mkdir('cams')
    if not os.path.exists(save_cam_path):
        os.mkdir(save_cam_path)
        
    for i, (_, image, label) in enumerate(tqdm(dataloader)):
        # print(f'in loader {image.shape = }')
        image = image.to('cuda')
        if 'multiscale' in args.method:
            all_cuts = make_crop(image)
            all_feature = []
            for cuts in all_cuts:
                ams = []
                for cut in cuts:
                    cut = F.interpolate(cut, args.res)
                    out = backbone(cut)
                    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
                    ams.append(activation_map[0].unsqueeze(1))
                all_feature.append(ams)
            activation_map = make_one_from_crop(*all_feature, out_shape=args.res)
            activation_map = activation_map.detach().cpu()
        else:
            out = backbone(image)
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        pil_image = to_pil_image(unorm(image[0]))
        data = np.array(pil_image)
        # print(f'{data.shape = }')
        red, green, blue = data.T
        data = np.array([blue, green, red])
        data = data.transpose()
        sub = Image.fromarray(data)
        result = overlay_mask(sub , to_pil_image(activation_map[0].squeeze_(0), mode='F'), alpha=0.7)
        stri = str(i)
        stri = (4 - len(stri)) * '0' + stri 
        result.save(f'{save_cam_path}/{stri}.png')


if __name__ == "__main__":
    args = parse_arguments()
    args.save_model_path = os.path.join(args.save_root, args.comment, 'K_train={}_{}'.format(args.K_train, args.metric_train))

    args.save_eval_path  = os.path.join(args.save_model_path, 'K_test={}_{}'.format(args.K_test, args.metric_test))
    if not os.path.exists(args.save_eval_path):
        os.makedirs(args.save_eval_path)
    
    logger = set_logger(os.path.join(args.save_eval_path, 'train.log'))
    
    model = get_model(args)
    if not 'multiscale' in args.method:
        testset    = get_dataset(args, mode='train_val')
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True,
                                                 collate_fn=collate_eval,
                                                 worker_init_fn=worker_init_fn(args.seed))
    else:
        testset = EvalCityscapesRAW(args.data_root, res=args.res, split='val', mode='test',
                                        label_mode=args.label_mode, long_image=args.long_image)
        testloader = torch.utils.data.DataLoader(testset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                pin_memory=False,
                                                collate_fn=collate_eval,
                                                worker_init_fn=worker_init_fn(args.seed))


    prefix = f'{args.arch}_{args.res}_{args.method}'

    dic = test_metrics_on_val(args, model, testloader, prefix=prefix)

    print(f'{dic}')

