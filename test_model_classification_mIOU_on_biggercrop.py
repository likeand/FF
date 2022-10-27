from utils import *
import os 
import numpy as np 
import torch 
import torch.nn as nn 
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from modules import fpn, myfpn, myfpn_multi, stego, swin
import argparse
from tqdm import tqdm 

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="/data0/zx_files/datasets/cityscapes")
    parser.add_argument('--save_root', type=str, default="/data0/zx_files/picie_results/")
    parser.add_argument('--restart_path', type=str)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed for reproducability.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers.')
    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--num_epoch', type=int, default=10) 
    parser.add_argument('--repeats', type=int, default=0)  

    # Train. 
    parser.add_argument('--arch', type=str, default='vit_large')
    parser.add_argument('--arch_local_save', type=str, default="/data0/zx_files/models/mae_visualize_vit_large.pth")
    
    parser.add_argument('--pretrain', action='store_true', default=True)
    RES = 384
    parser.add_argument('--res', type=int, default=RES, help='Input size.')
    parser.add_argument('--res1', type=int, default=RES, help='Input size scale from.')
    parser.add_argument('--res2', type=int, default=RES, help='Input size scale to.')
    parser.add_argument('--tar_res', type=int, default=80, help='Output Feature size.')
    
    parser.add_argument('--batch_size_cluster', type=int, default=256)
    parser.add_argument('--batch_size_train', type=int, default=8)
    parser.add_argument('--batch_size_test', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optim_type', type=str, default='Adam')
    parser.add_argument('--num_init_batches', type=int, default=20)
    parser.add_argument('--num_batches', type=int, default=1)
    parser.add_argument('--kmeans_n_iter', type=int, default=20)
    parser.add_argument('--in_dim', type=int, default=256)
    parser.add_argument('--X', type=int, default=80)

    # Loss. 
    parser.add_argument('--metric_train', type=str, default='cosine')   
    parser.add_argument('--metric_test', type=str, default='cosine')
    parser.add_argument('--K_cluster', type=int, default=27) # number of clusters, which will be further classified into K_train
    parser.add_argument('--K_train', type=int, default=27) # COCO Stuff-15 / COCO Thing-12 / COCO All-27
    parser.add_argument('--K_test', type=int, default=27) 
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
    parser.add_argument('--model_dir', type=str, default='/data0/zx_files/models')
    
    return parser.parse_args()


from einops import rearrange


def get_model(args):
    model = swin.PanopticFPN(args)
    # model = nn.DataParallel(model)
    model = model.cuda()

    return model


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
            feats = model(image)

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
        # print(f'{preds.shape = } \n {target.shape = }')

        # print(f'{torch.unique(preds) = } \n {torch.unique(target) = }')
        actual = target.reshape(-1)
        preds = preds.reshape(-1)
        mask = (actual >= 0) & (actual < n_classes) & (preds >= 0) & (preds < n_classes + extra_clusters)
        actual = actual[mask]
        preds = preds[mask]
        batch_hist = torch.bincount(
            (n_classes + extra_clusters) * actual + preds,
            minlength=n_classes * (n_classes + extra_clusters)) \
            .reshape(n_classes, n_classes + extra_clusters).t()
        # print(f'{hist.shape = }  {batch_hist.shape = }')
        hist += batch_hist
        return hist


import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from torchmetrics import Metric


class UnsupervisedMetrics(Metric):
    def __init__(self, prefix: str, n_classes: int, extra_clusters: int, compute_hungarian: bool,
                 dist_sync_on_step=True):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_classes = n_classes
        self.extra_clusters = extra_clusters
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix
        self.add_state("stats",
                       default=torch.zeros(n_classes + self.extra_clusters, n_classes, dtype=torch.int64),
                       dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            actual = target.reshape(-1)
            preds = preds.reshape(-1)
            mask = (actual >= 0) & (actual < self.n_classes) & (preds >= 0) & (preds < self.n_classes + self.extra_clusters)
            actual = actual[mask]
            preds = preds[mask]
            self.stats += torch.bincount(
                (self.n_classes + self.extra_clusters) * actual + preds,
                minlength=self.n_classes * (self.n_classes + self.extra_clusters)) \
                .reshape(self.n_classes, self.n_classes + self.extra_clusters).t().to(self.stats.device)

    def map_clusters(self, clusters):
        if self.extra_clusters == 0:
            return torch.tensor(self.assignments[1])[clusters]
        else:
            missing = sorted(list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0])))
            cluster_to_class = self.assignments[1]
            for missing_entry in missing:
                if missing_entry == cluster_to_class.shape[0]:
                    cluster_to_class = np.append(cluster_to_class, -1)
                else:
                    cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, -1)
            cluster_to_class = torch.tensor(cluster_to_class)
            return cluster_to_class[clusters]

    def compute(self):
        if self.compute_hungarian:
            self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True)
            # print(self.assignments)
            if self.extra_clusters == 0:
                self.histogram = self.stats[np.argsort(self.assignments[1]), :]
            if self.extra_clusters > 0:
                self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)
                histogram = self.stats[self.assignments_t[1], :]
                missing = list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0]))
                new_row = self.stats[missing, :].sum(0, keepdim=True)
                histogram = torch.cat([histogram, new_row], axis=0)
                new_col = torch.zeros(self.n_classes + 1, 1, device=histogram.device)
                self.histogram = torch.cat([histogram, new_col], axis=1)
        else:
            self.assignments = (torch.arange(self.n_classes).unsqueeze(1),
                                torch.arange(self.n_classes).unsqueeze(1))
            self.histogram = self.stats
        torch.save(histogram, self.prefix + '_hist.pth')
        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)

        metric_dict = {self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
                       self.prefix + "Accuracy": opc.item()}
        return {k: 100 * v for k, v in metric_dict.items()}


def test_metrics_on_val_bigger_crop(model, dataloader, prefix='test_on_val'):
    n_classes = args.K_test
    extra_clusters = 1000 - n_classes
    metrics = UnsupervisedMetrics(prefix, n_classes, extra_clusters, compute_hungarian=True)

    # histogram = torch.zeros((1000, n_classes)).to('cuda')
    
    prefix = prefix

    model.eval()
    with torch.no_grad():
        for i, (_, image, label) in enumerate(tqdm(dataloader)):
            image, label = image.cuda(), label.cuda()
            B, C, H, W = image.shape
            patches = (2, 4)
            kernels = (H // patches[0], W // patches[1])

            # label = F.interpolate(label, (patches[0] * 12, patches[1] * 12), mode='')
            label = TF.resize(label, (patches[0] * 12, patches[1] * 12), InterpolationMode.NEAREST)
            # image.shape = torch.Size([32, 3, 1024, 2048]) label.shape = torch.Size([32, 1024, 2048])
            # print(f'{image.shape = } {label.shape = }') 
            
            ## divide image into patches
            image_patch = F.unfold(image, kernel_size=kernels, padding=0, stride=kernels)
            # label_patch = F.unfold(label, kernel_size=kernels, stride=kernels)
            # image_patch.shape = torch.Size([32, 348843, 18])
            # print(f'{image_patch.shape = }')
            patched_image = rearrange(image_patch, 'b (c k1 k2) (h w) -> (h w b) c k1 k2', c=C, k1=kernels[0], k2=kernels[1], h=patches[0], w=patches[1])
            patched_image = F.interpolate(patched_image, (384, 384))
            # print(f'{patched_image.shape = }')
            # feats = model(image)
            classifications, _ = model.forward_features(patched_image) # (b, n, h, w) n = 1000
            n = classifications.shape[1]
            # print(f'before interp {classifications.shape = }')
            # classifications = F.interpolate(classifications, kernels)
            # print(f'before rearrange {classifications.shape = }')
            classifications = rearrange(classifications, '(h w b) c k1 k2 -> b (c k1 k2) (h w)', c=n, k1=12, k2=12, h=patches[0], w=patches[1])
            # print(f'after rearrange {classifications.shape = }')
            preds = F.fold(classifications, output_size=(patches[0] * 12, patches[1] * 12), kernel_size=(12, 12), stride=(12,12))
            # print(f'{preds.shape = }')
            preds = preds.argmax(dim=1)

            metrics.update(preds, label)
            # histogram = hist_update(histogram, preds, label, n_classes, extra_clusters=extra_clusters)
            # histogram += scores(label, preds, args.K_test)
            # Save labels and count. 
            # for idx, idx_img in enumerate(indice):
            #     counts += postprocess_label(args, K, idx, idx_img, scores, n_dual=view)
    res = metrics.compute()
    return res
    # Hungarian Matching. 
    # assignments = linear_sum_assignment(histogram.detach().cpu(), maximize=True)
    # if extra_clusters == 0:
    #     histogram = histogram[np.argsort(assignments[1]), :]
    # if extra_clusters > 0:
    #     assignments_t = linear_sum_assignment(histogram.detach().cpu().t(), maximize=True)
    #     histogram = histogram[assignments_t[1], :]
    #     missing = list(set(range(n_classes + extra_clusters)) - set(assignments[0]))
    #     new_row = histogram[missing, :].sum(0, keepdim=True)
    #     histogram = torch.cat([histogram, new_row], axis=0)
    #     new_col = torch.zeros(n_classes + 1, 1, device=histogram.device)
    #     histogram = torch.cat([histogram, new_col], axis=1)
    
    # torch.save(histogram, prefix + 'hist.pth')
    # tp = torch.diag(histogram)
    # fp = torch.sum(histogram, dim=0) - tp
    # fn = torch.sum(histogram, dim=1) - tp

    # iou = tp / (tp + fp + fn)
    # prc = tp / (tp + fn)
    # opc = torch.sum(tp) / torch.sum(histogram)

    # metric_dict = {prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
    #                 prefix + "Accuracy": opc.item()}
    # return {k: 100 * v for k, v in metric_dict.items()}

from data.cityscapes_eval_dataset import EvalCityscapesRAW

if __name__ == "__main__":
    args = parse_arguments()
    args.save_model_path = os.path.join(args.save_root, args.comment, 'K_train={}_{}'.format(args.K_train, args.metric_train))

    args.save_eval_path  = os.path.join(args.save_model_path, 'K_test={}_{}'.format(args.K_test, args.metric_test))
    if not os.path.exists(args.save_eval_path):
        os.makedirs(args.save_eval_path)
    
    logger = set_logger(os.path.join(args.save_eval_path, 'train.log'))
    
    model = get_model(args)
    # testset    = get_dataset(args, mode='train_val')
    testset = EvalCityscapesRAW(args.data_root, res=args.res, split='val', mode='test',
                                     label_mode=args.label_mode, long_image=args.long_image)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=8,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=collate_eval,
                                             worker_init_fn=worker_init_fn(args.seed))

    # centroids, kmloss1 = run_mini_batch_kmeans(args, logger, testloader, model, view=-1)
    # centroids = torch.randn(args.K_test, args.in_dim).cuda()
    dic = test_metrics_on_val_bigger_crop(model, testloader, prefix='test_on_val')
    print(f'{dic}')
