from utils import *
import os 
import numpy as np 
import torch 
import torch.nn as nn 
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
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
    parser.add_argument('--batch_size_train', type=int, default=2)
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

# def get_model(args):
#     if args.arch.startswith('resnet'):
#         model = fpn.PanopticFPN(args)
#     elif args.arch == 'dino':
#         model = stego.PanopticFPN(args)
#     elif args.arch == 'swinv2':
#         model = cut_swin_and_dino.PanopticFPN(args)
#     # model = cut_swin_and_dino.PanopticFPN(args)
#     else:
#         raise NotImplementedError("arch " + args.arch + " not implemented.")
#     model = model.cuda()
#     return model
from commons import get_model

import matplotlib.pyplot as plt 
import cv2
def test_metrics_on_val_for_cutswindino(model, dataloader, prefix='test_on_val'):
    save_dir = 'cut_swin_dino_segs'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    histogram = torch.zeros((args.K_test*2, args.K_test))
    extra_clusters = args.K_test 
    n_classes = args.K_test
    prefix = prefix
    model.eval()
    cmap = cut_swin_and_dino.create_cityscapes_colormap()
    with torch.no_grad():
        for i, (_, image, label) in enumerate(tqdm(dataloader)):
            image = image.to('cuda')
            label = F.interpolate(label.float().unsqueeze_(1), (384,768)).long()
            # obj_seg, things_seg, mask = model.forward_seg(image)
            # obj_seg, things_seg = F.interpolate(obj_seg.float().unsqueeze(1), (384,768))[:,0].long(), F.interpolate(things_seg.float().unsqueeze(1), (384,768))[:,0].long()
            # mask = F.interpolate(mask.float().unsqueeze(1), (384, 768))[:,0].detach().cpu()
            # obj_seg, things_seg, label = obj_seg.detach().cpu(), things_seg.detach().cpu(), label.detach().cpu()
            things_seg = model.forward_seg(image)
            things_seg= F.interpolate(things_seg.float().unsqueeze(1), (384,768))[:,0].long().detach().cpu()
            ts1 = things_seg.clone()
            # ts1[mask>0] = obj_seg[mask>0] + 27 
            B,_,_ = ts1.shape
            preds = ts1.view(B, -1)
            # print(f'{mask.shape = }')
            # for j in range(image.shape[0]):
            #     plt.figure()
            #     ax = plt.subplot(4,1,1)
            #     # print(f'{cmap[obj_seg[j]].shape = }')
            #     ax.imshow(cmap[ts1[j]])
            #     ax = plt.subplot(4,1,2)
            #     ax.imshow(mask[j])
            #     ax = plt.subplot(4,1,3)
            #     ax.imshow(cmap[things_seg[j]])
            #     ax = plt.subplot(4,1,4)
            #     ax.imshow(cmap[label[j]])
            #     plt.savefig(f'{save_dir}/{image.shape[0]*i + j}.png', dpi=300)
            # B,_,_ = segs.shape
            # print(f'{ts1.shape = } \n{label.shape = }')
            
            # preds = segs.view(B, -1).cpu()
            label = label.view(B, -1).cpu()
            histogram = hist_update(histogram, preds, label, n_classes, extra_clusters)
            # if i > 20: break 
    # Hungarian Matching. 
    assignments = linear_sum_assignment(histogram.detach().cpu(), maximize=True)
    if extra_clusters == 0:
        histogram = histogram[np.argsort(assignments[1]), :]
    if extra_clusters > 0:
        assignments_t = linear_sum_assignment(histogram.detach().cpu().t(), maximize=True)
        histogram_1 = histogram[assignments_t[1], :]
        # print(f'in if {histogram.shape = }')
        missing = list(set(range(n_classes + extra_clusters)) - set(assignments[0]))
        new_row = histogram[missing, :].sum(0, keepdim=True)
        histogram = torch.cat([histogram_1, new_row], axis=0)
        new_col = torch.zeros(n_classes + 1, 1, device=histogram.device)
        histogram = torch.cat([histogram, new_col], axis=1)
    
    torch.save(histogram, prefix + 'hist.pth')
    tp = torch.diag(histogram)
    fp = torch.sum(histogram, dim=0) - tp
    fn = torch.sum(histogram, dim=1) - tp

    iou = tp / (tp + fp + fn)
    prc = tp / (tp + fn)
    opc = torch.sum(tp) / torch.sum(histogram)

    metric_dict = {prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
                    prefix + "Accuracy": opc.item(),
    }

    return {k: 100 * v for k, v in metric_dict.items()}


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

def test_metrics_on_val(model, centroids, dataloader, prefix='test_on_val'):
    metric_function = get_metric_as_conv(centroids)
    histogram = torch.zeros((args.K_test, args.K_test))
    extra_clusters = 0
    n_classes = args.K_test
    prefix = prefix
    classifier = initialize_classifier(args)
    classifier.module.weight.data = centroids.unsqueeze(-1).unsqueeze(-1)
    model.eval()
    total, non_zeros = 0, 0
    with torch.no_grad():
        for i, (_, image, label) in enumerate(tqdm(dataloader)):
            # if view == 1:
            #     image = eqv_transform_if_needed(args, dataloader, indice, image.cuda(non_blocking=True))
            image = image.to('cuda')
            feats = model.forward(image)
            feats = F.interpolate(feats, (384, 384))
            label = F.interpolate(label.float().unsqueeze_(1), (384, 384)).long()
            # elif view == 2:
            #     image = image.cuda(non_blocking=True)
            #     feats = eqv_transform_if_needed(args, dataloader, indice, model(image))
            # Normalize.
            # print(f'{label.shape = }')
            if args.metric_train == 'cosine':
                feats = F.normalize(feats, dim=1, p=2)

            fc = feats.amax(dim=1, keepdim=True)
            B, C, H, W = fc.shape
            total += B*C*H*W 
            non_zeros += torch.sum(fc > 0)

            # Compute distance and assign label. 
            # probs = compute_negative_euclidean(feats, centroids, metric_function) 
            
            probs = classifier(feats)
            # probs = F.interpolate(probs, label.shape[-2:], mode='bilinear', align_corners=False).detach()
            preds = probs.topk(1, dim=1)[1].view(B, -1).cpu()
            label = label.view(B, -1).cpu()

            histogram = hist_update(histogram, preds, label, n_classes, extra_clusters)
            # histogram += scores(label, preds, args.K_test)
            # Save labels and count. 
            # for idx, idx_img in enumerate(indice):
            #     counts += postprocess_label(args, K, idx, idx_img, scores, n_dual=view)

    # Hungarian Matching. 
    assignments = linear_sum_assignment(histogram.detach().cpu(), maximize=True)
    if extra_clusters == 0:
        histogram = histogram[np.argsort(assignments[1]), :]
    if extra_clusters > 0:
        assignments_t = linear_sum_assignment(histogram.detach().cpu().t(), maximize=True)
        histogram = histogram[assignments_t[1], :]
        missing = list(set(range(n_classes + extra_clusters)) - set(assignments[0]))
        new_row = histogram[missing, :].sum(0, keepdim=True)
        histogram = torch.cat([histogram, new_row], axis=0)
        new_col = torch.zeros(n_classes + 1, 1, device=histogram.device)
        histogram = torch.cat([histogram, new_col], axis=1)
    
    torch.save(histogram, prefix + 'hist.pth')
    torch.save(assignments, prefix + 'asgn.pth')
    tp = torch.diag(histogram)
    fp = torch.sum(histogram, dim=0) - tp
    fn = torch.sum(histogram, dim=1) - tp

    iou = tp / (tp + fp + fn)
    prc = tp / (tp + fn)
    opc = torch.sum(tp) / torch.sum(histogram)

    metric_dict = {prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
                    prefix + "Accuracy": opc.item(),
                    prefix + "Nonzero%": non_zeros.item() / (total + 1)}

    return {k: 100 * v for k, v in metric_dict.items()}

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
    centroids, kmloss1 = run_mini_batch_kmeans(args, logger, testloader, model, view=-1)
    # prefix = 'test_only_stego_on_val'
    if not os.path.exists('gen_files'):
        os.mkdir('gen_files')
    prefix = f'gen_files/{args.arch}_{args.res}_{args.method}'
    torch.save(centroids, f'{prefix}_centroids.pth')
    # centroids = torch.load(f'{prefix}_centroids.pth')
    # centroids = torch.randn(args.K_test, args.in_dim).cuda()
    dic = test_metrics_on_val(model, centroids, testloader, prefix=prefix)
    # dic = test_metrics_on_val_for_cutswindino(model, testloader, prefix)
    print(f'{dic}')

