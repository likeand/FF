import matplotlib.pyplot as plt 
import os 
# from PIL import Image 
import cv2 

def plot(dirname, image_indices):
    dirs = os.listdir(dirname)
    fig, ax = plt.subplots(len(dirs), len(image_indices), figsize=(len(image_indices)*3, len(dirs)*3))
    for i, image_dir in enumerate(dirs):
        for j, ind in enumerate(image_indices):
            stri = (4 - len(str(ind))) * '0' + str(ind)
            path = os.path.join(dirname, image_dir, stri + '.png')
            image = cv2.imread(path)
            image = cv2.resize(image, (320, 320))
            ax[i, j].imshow(image)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(image_dir, fontsize=16)
    # for i in range(self.cfg.n_images):
    #     ax[0, i].imshow(prep_for_plot(output["img"][i]))
    #     ax[1, i].imshow(self.label_cmap[output["label"][i]])
    #     ax[2, i].imshow(self.label_cmap[output["linear_preds"][i]])
    #     ax[3, i].imshow(self.label_cmap[self.cluster_metrics.map_clusters(output["cluster_preds"][i])])
    # ax[0, 0].set_ylabel("Image", fontsize=16)
    # ax[1, 0].set_ylabel("Label", fontsize=16)
    # ax[2, 0].set_ylabel("Linear Probe", fontsize=16)
    # ax[3, 0].set_ylabel("Cluster Probe", fontsize=16)
    # remove_axes(ax)
    # ax.xaxis.set_major_formatter(plt.NullFormatter())
    # ax.yaxis.set_major_formatter(plt.NullFormatter())
    
    # plt.axis('off')
    plt.tight_layout()
    plt.savefig('pltcams.png', dpi=400)
    
if __name__ == "__main__":
    plot('/home/zhulifu/unsup_seg/trials_unsupervised_segmentation/cams', range(7))