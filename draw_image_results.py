from utils import *
import os 
import numpy as np 
import torch 
import torch.nn as nn 
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from modules import fpn, myfpn, myfpn_multi, stego, swin, cut_swin, dino_full, cut_swin_and_dino
import argparse
from tqdm import tqdm 
from data.cityscapes_eval_dataset import EvalCityscapesRAW


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
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--arch_local_save', type=str, default="/data0/zx_files/models/mae_visualize_vit_large.pth")  
    parser.add_argument('--pretrain', action='store_true', default=True)
    ## res to choose: 320, 384, 640
    RES = 320
    parser.add_argument('--res', type=int, default=RES, help='Input size.')
    # parser.add_argument('--res1', type=int, default=RES, help='Input size scale from.')
    # parser.add_argument('--res2', type=int, default=RES, help='Input size scale to.')
    parser.add_argument('--tar_res', type=int, default=160, help='Output Feature size.')
    
    ## methods to choose:
    ## cam, multiscale, cam_multiscale
    parser.add_argument('--method', type=str, default='multiscale')
    parser.add_argument('--batch_size_cluster', type=int, default=256)
    parser.add_argument('--batch_size_train', type=int, default=1)
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

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
def getStat():
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    # stat = ([-0.8600419, -0.5789229, -0.5359389], [0.7740311, 0.8117282, 0.792767]) # calulated after normalize by imagenet mean std;
    stat = ([0.28805077, 0.32632148, 0.2854132], [0.17725338, 0.18182696, 0.17837301]) # raw image calculated.
    if stat is not None:
        return stat
    
def get_model(args):
    if args.arch.startswith('resnet'):
        model = fpn.PanopticFPN(args)
    elif args.arch == 'dino':
        model = stego.PanopticFPN(args)
    elif args.arch == 'swinv2':
        model = cut_swin_and_dino.PanopticFPN(args)
    # model = cut_swin_and_dino.PanopticFPN(args)
    else:
        raise NotImplementedError(f"arch {args.arch} not implemented.")
    model = model.cuda()
    return model

import matplotlib.pyplot as plt 
import cv2

def run_mini_batch_kmeans(args, logger, dataloader, model, view):
    """
    num_init_batches: (int) The number of batches/iterations to accumulate before the initial k-means clustering.
    num_batches     : (int) The number of batches/iterations to accumulate before the next update. 
    """
    kmeans_loss  = AverageMeter()
    faiss_module = get_faiss_module(args)
    data_count   = np.zeros(args.K_cluster)
    featslist    = []
    num_batches  = 0
    first_batch  = True
    
    # Choose which view it is now. 
    dataloader.dataset.view = view

    model.train()
    print(f'start minibatch kmeans')
    with torch.no_grad():
        for i_batch, (_, image, label) in enumerate(tqdm(dataloader)):
            # 1. Compute initial centroids from the first few batches. 
            # if view == 1:
            #     image = eqv_transform_if_needed(args, dataloader, indice, image.cuda(non_blocking=True))
            #     feats = model(image)
            # elif view == 2:
            #     image = image.cuda(non_blocking=True)
            #     feats = eqv_transform_if_needed(args, dataloader, indice, model(image))
            # else:
                # For evaluation. 
            # print(len(image))
            image = image.cuda(non_blocking=True)
            # print(f'{image.shape = }')
            feats = model.forward(image)
            # print(f'{feats.shape = }')

            # Normalize.
            if args.metric_test == 'cosine':
                feats = F.normalize(feats, dim=1, p=2)
            
            if i_batch == 0:
                logger.info('Batch input size : {}'.format(list(image.shape)))
                logger.info('Batch feature : {}'.format(list(feats.shape)))
            
            feats = feature_flatten(feats).detach().cpu()
            if num_batches < args.num_init_batches:
                featslist.append(feats)
                num_batches += 1
                
                if num_batches == args.num_init_batches or num_batches == len(dataloader):
                    if first_batch:
                        # Compute initial centroids. 
                        # By doing so, we avoid empty cluster problem from mini-batch K-Means. 
                        featslist = torch.cat(featslist).cpu().numpy().astype('float32')
                        centroids = get_init_centroids(args, args.K_cluster, featslist, faiss_module).astype('float32')
                        D, I = faiss_module.search(featslist, 1)

                        kmeans_loss.update(D.mean())
                        logger.info('Initial k-means loss: {:.4f} '.format(kmeans_loss.avg))
                        
                        # Compute counts for each cluster. 
                        for k in np.unique(I):
                            data_count[k] += len(np.where(I == k)[0])
                        first_batch = False
                    else:
                        b_feat = torch.cat(featslist)
                        faiss_module = module_update_centroids(faiss_module, centroids)
                        D, I = faiss_module.search(b_feat.numpy().astype('float32'), 1)

                        kmeans_loss.update(D.mean())

                        # Update centroids. 
                        for k in np.unique(I):
                            idx_k = np.where(I == k)[0]
                            data_count[k] += len(idx_k)
                            centroid_lr    = len(idx_k) / (data_count[k] + 1e-6)
                            centroids[k]   = (1 - centroid_lr) * centroids[k] + centroid_lr * b_feat[idx_k].mean(0).numpy().astype('float32')
                    
                    # Empty. 
                    featslist   = []
                    num_batches = args.num_init_batches - args.num_batches

            if (i_batch % 100) == 0:
                logger.info('[Saving features]: {} / {} | [K-Means Loss]: {:.4f}'.format(i_batch, len(dataloader), kmeans_loss.avg))
    print(f'end kmeans')
    centroids = torch.tensor(centroids, requires_grad=False).cuda()

    return centroids, kmeans_loss.avg

def hist_update(hist, preds, target, n_classes, extra_clusters):
    with torch.no_grad():
        actual = target.reshape(-1)
        preds = preds.reshape(-1)
        mask = (actual >= 0) & (actual < n_classes) & (preds >= 0) & (preds < n_classes)
        actual = actual[mask]
        preds = preds[mask]
        hist += torch.bincount(
            (n_classes + extra_clusters) * actual + preds,
            minlength=n_classes * (n_classes + extra_clusters)) \
            .reshape(n_classes, n_classes + extra_clusters).t()
        return hist
import pandas as pd 

def remap_values(remapping, x):
    index = torch.bucketize(x.ravel(), remapping[0])
    return remapping[1][index].reshape(x.shape)

def draw_image_result_on_val(model, centroids, dataloader, prefix='test_on_val', out_dir='draw_image_result'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    assignments = torch.load(prefix + 'asgn.pth')

    extra_clusters = 0
    n_classes = args.K_test
    prefix = prefix
    classifier = initialize_classifier(args)
    classifier.module.weight.data = centroids.unsqueeze(-1).unsqueeze(-1)
    model.eval()
    cmap = create_cityscapes_colormap()
    total, non_zeros = 0, 0
    dic = {
        'index': [],
        'tp': []
    }
    unorm = UnNormalize(*getStat())
    trial_name = prefix.split('/')[-1]
    with torch.no_grad():
        for i, (_, image, label) in enumerate(tqdm(dataloader)):
            stri = str(i)
            stri = (4 - len(stri)) * '0' + stri 
            img_name = f'{out_dir}/{stri}_img.png'
            lbl_name = f'{out_dir}/{stri}_lbl.png'
            seg_name = f'{out_dir}/{stri}_seg_{trial_name}.png'
            # if view == 1:
            #     image = eqv_transform_if_needed(args, dataloader, indice, image.cuda(non_blocking=True))
            image = image.to('cuda')
            feats = model.forward(image)
            feats = F.interpolate(feats, (384, 384))
            label = F.interpolate(label.float().unsqueeze_(1), (384, 384)).long()

            if args.metric_train == 'cosine':
                feats = F.normalize(feats, dim=1, p=2)

            probs = classifier(feats)
            histogram = torch.zeros((args.K_test, args.K_test))

            preds = probs.topk(1, dim=1)[1].cpu()
            # print(f'{preds.shape = }')
            # print(f'{assignments[1].shape = }')
            # preds = preds[0,0][assignments[1]]
            preds = preds[0,0]
            for num in range(27):
                preds[preds == num] = assignments[1][num]
            # values_map = torch.arange(0, 27), torch.tensor(assignments[1])
            # print(torch.unique(preds))
            # preds = remap_values(values_map, preds)
            # print(f'preds shape after assignments {preds.shape}')
            # [cmap]
            label = label.cpu()[0,0]
            # print(f'{label.shape = }')
            # [cmap]
            tp = torch.sum(preds == label)
            dic['index'].append(stri)
            dic['tp'].append(tp)
            if not os.path.exists(img_name):
                
                cv2.imwrite(img_name, 255 * cv2.cvtColor(unorm(image[0]).permute(1,2,0).cpu().numpy(), cv2.COLOR_BGR2RGB))
            if not os.path.exists(lbl_name):
                print(f'{cmap[label].shape = }')
                cv2.imwrite(lbl_name, cv2.cvtColor(cmap[label].astype(np.uint8), cv2.COLOR_BGR2RGB))
            if not os.path.exists(seg_name):
                cv2.imwrite(seg_name, cmap[preds].astype(np.uint8))
    save_dir = f'{out_dir}/table.pth'
    import pickle
    if not os.path.exists(save_dir):
        table = pd.DataFrame(dic)
    else:
        with open(save_dir, 'r') as f:
            table = pickle.load(f)
    
            nt = pd.DataFrame(dic)
            table = pd.merge(table, nt, right_on='index', left_on='index')
    with open(save_dir, 'w') as f:
        pickle.dump(table, f)


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
                                                 batch_size=args.batch_size_train,
                                                 shuffle=False,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True,
                                                 collate_fn=collate_eval,
                                                 worker_init_fn=worker_init_fn(args.seed))
    else:
        testset = EvalCityscapesRAW(args.data_root, res=args.res, split='val', mode='test',
                                        label_mode=args.label_mode, long_image=args.long_image)
        testloader = torch.utils.data.DataLoader(testset,
                                                batch_size=args.batch_size_train,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                pin_memory=False,
                                                collate_fn=collate_eval,
                                                worker_init_fn=worker_init_fn(args.seed))
    # for i, (_, image, label) in enumerate(tqdm(testloader)):
    #     print(f'in main {label.shape = }')
    #     label = F.interpolate(label.float().unsqueeze_(0), (384, 384)).long()
    #     print(f'in main {label.shape = }')
    #     break 
    # centroids, kmloss1 = run_mini_batch_kmeans(args, logger, testloader, model, view=-1)
    # prefix = 'test_only_stego_on_val'
    prefix = f'gen_files/{args.arch}_{args.res}_{args.method}'
    # torch.save(centroids, f'{prefix}_centroids.pth')
    centroids = torch.load(f'{prefix}_centroids.pth')
    # centroids = torch.randn(args.K_test, args.in_dim).cuda()
    # dic = test_metrics_on_val(model, centroids, testloader, prefix=prefix)
    # dic = test_metrics_on_val_for_cutswindino(model, testloader, prefix)
    # print(f'{dic}')
    draw_image_result_on_val(model, centroids, testloader, prefix)

