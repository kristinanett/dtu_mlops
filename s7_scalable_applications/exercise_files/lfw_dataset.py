"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F



class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        path_to_folder = "./lfw/"
        self.transform = transform
        self.image_paths = glob.glob(path_to_folder + "**/*.jpg", recursive = True)

    def __len__(self):
        # TODO: fill out
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        img = Image.open(self.image_paths[index])
        return self.transform(img)

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def errorplot(workers, results, stds):
    plt.errorbar(workers, results, stds)
    plt.xlabel('nr of workers')
    plt.ylabel('time')
    plt.savefig("errorbar_plot.jpg")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='', type=str)
    parser.add_argument('-num_workers', default=0, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers
    dataloader1 = DataLoader(dataset, batch_size=512, shuffle=False,
                            num_workers=args.num_workers)
    
    #args.visualize_batch = True
    if args.visualize_batch:
        # TODO: visualize a batch of images
        batch_vis = next(iter(dataloader1)) # batch[0] one image as a tensor
        grid = make_grid(batch_vis)
        show(grid)
        plt.savefig("batch_visualization.jpg")

    args.get_timing = True
    if args.get_timing:
        workers = [1, 2, 3, 4, 5]
        results = []
        stds = []
        for worker in workers:
            dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=worker)
            res = [ ]
            for _ in range(5):
                start = time.time()
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx > 10:
                        break
                end = time.time()

                res.append(end - start)
            
            res = np.array(res)
            results.append(np.mean(res))
            stds.append(np.std(res))
        
        errorplot(workers, results, stds)

        print('Timing:', np.mean(res), "+-", np.std(res))
