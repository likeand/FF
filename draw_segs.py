import argparse
import os
import time as t
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 
from PIL import Image
from utils import *
from commons import * 
from modules import fpn 
from data.cityscapes_eval_dataset import EvalCityscapesRAW
from data.cityscapes_train_dataset import TrainCityscapesRAW
from tqdm import tqdm 

def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_root', type=str, default="/home/zhulifu/unsup_seg/STEGO-master/seg_dataset/cityscapes")
    ds = 'coco' # 'city'
    parser.add_argument('--data_root', type=str, default=f"/home/zhulifu/unsup_seg/STEGO-master/seg_dataset/{'cityscapes' if ds == 'city' else 'coco_stuff'}")
    parser.add_argument('--save_root', type=str, default="/home/zhulifu/unsup_seg/train_out")
    parser.add_argument('--save_model_path', type=str, default="/home/zhulifu/unsup_seg/train_out")
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
    ## resnet18, resnet50, dino, swinv2, stego, picie
    arch = 'resnet50'
    parser.add_argument('--arch', type=str, default=arch)
    # parser.add_argument('--arch_local_save', type=str, default="/data0/zx_files/models/mae_visualize_vit_large.pth")  
    parser.add_argument('--pretrain', action='store_true', default=True)
    ## res to choose: 320, 384, 640
    RES = 384 if 'swinv2' in arch else 320
    parser.add_argument('--res', type=int, default=RES, help='Input size.')
    parser.add_argument('--res1', type=int, default=RES, help='Input size scale from.')
    parser.add_argument('--res2', type=int, default=RES, help='Input size scale to.')
    parser.add_argument('--tar_res', type=int, default=80, help='Output Feature size.')
    
    ## methods to choose:
    ## cam, multiscale, cam_multiscale
    method = ''
    parser.add_argument('--method', type=str, default=method)
    parser.add_argument('--batch_size_cluster', type=int, default=256)
    
    bs = 1 if 'multiscale' in method else 8
    parser.add_argument('--batch_size_train', type=int, default=bs)
    parser.add_argument('--batch_size_test', type=int, default=bs)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optim_type', type=str, default='Adam')
    parser.add_argument('--num_init_batches', type=int, default=20)
    parser.add_argument('--num_batches', type=int, default=1)
    parser.add_argument('--kmeans_n_iter', type=int, default=20)
    
    in_dim = 256 if arch.startswith('resnet') or arch == 'swinv2' else 1024
    parser.add_argument('--in_dim', type=int, default=in_dim)
    parser.add_argument('--X', type=int, default=80)

    # Loss. 
    parser.add_argument('--metric_train', type=str, default='cosine')   
    parser.add_argument('--metric_test', type=str, default='cosine')
    K_ = 27 if arch == 'picie' or arch == 'stego' else 28
    parser.add_argument('--K_cluster', type=int, default=K_) # number of clusters, which will be further classified into K_train
    parser.add_argument('--K_train', type=int, default=K_) # COCO Stuff-15 / COCO Thing-12 / COCO All-27
    parser.add_argument('--K_test', type=int, default=K_) 
    
    parser.add_argument('--no_balance', action='store_true', default=False)
    parser.add_argument('--mse', action='store_true', default=False)

    # Dataset. 
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--equiv', action='store_true', default=True)
    parser.add_argument('--min_scale', type=float, default=0.5)
    parser.add_argument('--stuff', action='store_true', default=False)
    parser.add_argument('--thing', action='store_true', default=False)
    parser.add_argument('--jitter', action='store_true', default=True)
    parser.add_argument('--grey', action='store_true', default=True)
    parser.add_argument('--blur', action='store_true', default=True)
    parser.add_argument('--h_flip', action='store_true', default=True)
    parser.add_argument('--v_flip', action='store_true', default=True)
    parser.add_argument('--random_crop', action='store_true', default=False)
    parser.add_argument('--val_type', type=str, default='train')
    parser.add_argument('--version', type=int, default=7)
    parser.add_argument('--fullcoco', action='store_true', default=False)

    # Eval-only
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_path', type=str)

    # Cityscapes-specific.
    parser.add_argument('--cityscapes', action='store_true', default=False)
    parser.add_argument('--label_mode', type=str, default='gtFine')
    parser.add_argument('--long_image', action='store_true', default=False)
    return parser.parse_args()

def draw_segs(args, logger, epoch=9):
    model, optimizer, classifier1 = get_model_and_optimizer(args, logger)
    ds = 'coco_' if not args.cityscapes else ''
    prefix = f'train_{ds}{args.arch}_{args.res}_{args.method}'
    if args.arch == 'stego':
        mapping = torch.load('/home/zhulifu/unsup_seg/trials_unsupervised_segmentation/gen_files/' + ds + 'stego_320_assignments.pth')
        mymap = list(enumerate(mapping))
        mymap.sort(key=lambda x: x[1])
        mapping = np.array([x[0] for x in mymap])
    elif args.arch == 'picie':
        mapping = torch.load('./gen_files/' + ds + 'picie_320_assignments.pth')
        mymap = list(enumerate(mapping))
        mymap.sort(key=lambda x: x[1])
        mapping = np.array([x[0] for x in mymap])
        if args.cityscapes:
            pth = torch.load('/home/zhulifu/unsup_seg/STEGO-master/saved_models/picie_city.tar')
        else:
            pth = torch.load('/home/zhulifu/unsup_seg/STEGO-master/saved_models/picie_coco.pkl')
        nd = {}
        for key in pth['state_dict'].keys():
            if key.startswith('module.'):
                nd[key[7:]] = pth['state_dict'][key]
        model.load_state_dict(nd)
        nd = {}
        for key in pth['classifier1_state_dict'].keys():
            if key.startswith('module.'):
                nd[key[7:]] = pth['classifier1_state_dict'][key]
        classifier1.load_state_dict(nd)
        
        classifier1.eval()
    else:
        pth = torch.load(os.path.join(args.save_model_path, f'{prefix}_checkpoint_{epoch}.pth.tar'))

        model.load_state_dict(pth['state_dict'])
        classifier1.load_state_dict(pth['classifier1_state_dict'])
        
        classifier1.eval()
    model.eval()
    testset    = get_dataset(args, mode='train_val')

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size_test,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=collate_eval,
                                             worker_init_fn=worker_init_fn(args.seed))
    
    logger.info('\n============================= [Epoch {}] =============================\n'.format(epoch))
    logger.info('Start computing segmentations.')
    t1 = t.time()
    all_results = []
    out_dir='draw_image_result' if args.cityscapes else 'draw_image_result_coco'
    cmap = create_cityscapes_colormap()
    unorm = UnNormalize(*getStat())
    trial_name = prefix[6:]
    with torch.no_grad():
        for index, (_, image, label) in enumerate(tqdm(testloader, mininterval=10)):
            if (not image.shape[-1] == args.res) and (not 'multiscale' in args.method):
                image = F.interpolate(image, (args.res, args.res))
            # label = F.interpolate(label.unsqueeze_(1).float(), (args.tar_res, args.tar_res), mode='bilinear').long().squeeze_(1)
            image = image.cuda()
            label = label.cuda()
            
            label[label<0] = 27
            if args.arch == 'stego':
                segs = model(image)
            else:
                feat = model(image)
                segs = classifier1(feat)
            image = F.interpolate(image, (384, 384), mode='bilinear')
            segs = F.interpolate(segs, (384, 384), mode='bilinear')
            label = F.interpolate(label.float().unsqueeze_(1), (384, 384), mode='nearest').long()
            segs = torch.argmax(segs, dim=1, keepdim=True)
            for j in range(segs.shape[0]):
                i = index * args.batch_size_test + j
                stri = (4 - len(str(i))) * '0' + str(i) 
                img_name = f'{out_dir}/{stri}_img.png'
                lbl_name = f'{out_dir}/{stri}_lbl.png'
                seg_name = f'{out_dir}/{stri}_seg_{trial_name}.png'
                im = image[j]
                se = segs[j, 0].cpu().numpy()
                lb = label[j, 0].cpu().numpy()
                
                if args.arch == 'stego' or args.arch == 'picie':
                    se = mapping[se]
                se[lb == 27] = 27
                if not args.cityscapes:
                    lb[lb > 27] = 0
                hist = scores(lb, se, args.K_test)
                result = get_result_metrics(hist)
                all_results.append((result['mean_iou'], stri))
                # print(all_results[-1])
                if not os.path.exists(img_name):
                # if True:
                    cv2.imwrite(img_name, 255 * cv2.cvtColor(unorm(im).permute(1,2,0).cpu().numpy(), cv2.COLOR_BGR2RGB))
                if not os.path.exists(lbl_name):
                    cv2.imwrite(lbl_name, cv2.cvtColor(cmap[lb].astype(np.uint8), cv2.COLOR_BGR2RGB))
                # if not os.path.exists(seg_name):
                cv2.imwrite(seg_name, cv2.cvtColor(cmap[se].astype(np.uint8), cv2.COLOR_BGR2RGB))
    all_results.sort(key=lambda x: x[0], reverse=True)
    good_results = all_results[:30]
    logger.info(f'{good_results = }')
        
        
def redraw_segs(testloader, tar_dir='/home/zhulifu/unsup_seg/trials_unsupervised_segmentation/draw_image_result'):
    files = os.listdir(tar_dir)
    dic = {}
    for file in files:
        prefix = file[:4]
        dic[prefix] = dic.get(prefix, []) + [file]
        
    for index, (_, image, label) in enumerate(tqdm(testloader, mininterval=10)):
        label = F.interpolate(label.float().unsqueeze_(1), (384, 384), mode='nearest').long()
        for j in range(label.shape[0]):
            lb = label[j, 0].cpu().numpy()
            i = index * args.batch_size_test + j
            stri = (4 - len(str(i))) * '0' + str(i) 
            seg_prefix = f'{stri}_seg_'
            for file in dic[stri]:
                
                if file.startswith(seg_prefix):
                    seg_name = os.path.join(tar_dir, file)
                    img = Image.open(seg_name)
                    
                    if 'multiscale' in file:
                        width, height = img.size 
                        newsize = (width * 2, height) 
                        img = img.resize(newsize)
                        # width, height = img.size 
                        left = width // 2
                        top = 0
                        right = width * 3 // 2
                        bottom = height
                        img = img.crop((left, top, right, bottom)) 
                    
                    img = np.array(img)
                    img[lb < 0] = 0
                    img = Image.fromarray(img.astype('uint8')).convert('RGB')
                    img.save('new_dir/'+file)
                    # cv2.imwrite(seg_name, img)

def get_stego_picie_hist():
    for arch in ['stego', 'picie']:
    
        args = parse_arguments()
        args.cityscapes = False
        args.arch = arch 
        args.method = ''
        args.res = 320
        args.K_cluster = 27 #if arch == 'picie' else 28
        args.in_dim = 128 if arch == 'picie' else 1024
        ds = 'coco_' if not args.cityscapes else ''
        
        prefix = f'train_linear_{ds}{args.arch}_{args.res}_{args.method}'
        
        args.save_root = './train_results/' + prefix
        if not os.path.exists(args.save_root):
            os.mkdir(args.save_root)
        args.save_model_path = args.save_root + '/models/'
        if not os.path.exists(args.save_model_path):
            os.mkdir(args.save_model_path)
        # args.save_model_path = os.path.join(args.save_root, args.comment, 'K_train={}_{}'.format(args.K_train, args.metric_train))
        args.save_eval_path  = os.path.join(args.save_model_path, 'K_test={}_{}'.format(args.K_test, args.metric_test))
        if not os.path.exists(args.save_eval_path):
            os.mkdir(args.save_eval_path)
        logger = set_logger(os.path.join(args.save_eval_path, 'train.log'))
        logger.info(f'evaluating {arch}')
        
        model, optimizer, classifier1 = get_model_and_optimizer(args, logger)
        # prefix = f'train_{args.arch}_{args.res}_{args.method}'
        if args.arch == 'stego':
            pass 
        elif args.arch == 'picie':
            if args.cityscapes:
                pth = torch.load('/home/zhulifu/unsup_seg/STEGO-master/saved_models/picie_city.tar')
            else:
                pth = torch.load('/home/zhulifu/unsup_seg/STEGO-master/saved_models/picie_coco.pkl')
            nd = {}
            for key in pth['state_dict'].keys():
                if key.startswith('module.'):
                    nd[key[7:]] = pth['state_dict'][key]
            model.load_state_dict(nd)
            nd = {}
            for key in pth['classifier1_state_dict'].keys():
                if key.startswith('module.'):
                    nd[key[7:]] = pth['classifier1_state_dict'][key]
            classifier1.load_state_dict(nd)
            classifier1.eval()

        model.eval()
        testset    = get_dataset(args, mode='train_val')
        # testset = EvalCityscapesRAW(args.data_root, res=args.res, split='val', mode='train_val',
                                            # label_mode=args.label_mode, long_image=args.long_image)
        testloader = torch.utils.data.DataLoader(testset,
                                                batch_size=4,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                collate_fn=collate_eval,
                                                worker_init_fn=worker_init_fn(args.seed))
        
        evaluate(args, logger, testloader, classifier1, model)


if __name__ == "__main__":
    args = parse_arguments()
    args.cityscapes = False
    ds = 'coco_' if not args.cityscapes else ''
    prefix = f'train_linear_{args.arch}_{args.res}_{args.method}'
    # prefix = f'train_picie_{args.arch}_{args.res}_{args.method}'
    args.save_root = './train_results/' + prefix
    if not os.path.exists(args.save_root):
        os.mkdir(args.save_root)
    args.save_model_path = args.save_root + '/models/'
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)
    # args.save_model_path = os.path.join(args.save_root, args.comment, 'K_train={}_{}'.format(args.K_train, args.metric_train))
    args.save_eval_path  = os.path.join(args.save_model_path, 'K_test={}_{}'.format(args.K_test, args.metric_test))
    if not os.path.exists(args.save_eval_path):
        os.mkdir(args.save_eval_path)
    logger = set_logger(os.path.join(args.save_eval_path, 'train.log'))
    draw_segs(args, logger, epoch=2)
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # get_stego_picie_hist()
    # testset    = get_dataset(args, mode='train_val')
    # # testset = EvalCityscapesRAW(args.data_root, res=args.res, split='val', mode='train_val',
    #                                     # label_mode=args.label_mode, long_image=args.long_image)
    # testloader = torch.utils.data.DataLoader(testset,
    #                                         batch_size=1,
    #                                         shuffle=False,
    #                                         num_workers=args.num_workers,
    #                                         pin_memory=True,
    #                                         collate_fn=collate_eval,
    #                                         worker_init_fn=worker_init_fn(args.seed))
    # redraw_segs(testloader)