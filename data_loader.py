import os
from PIL import Image
import numpy as np
import os.path as osp
import torch
from IPython import embed
from torch.utils.data import Dataset

def read_image(img_path):
    got_img = False

    if not osp.exists(img_path):
        raise IOError("{} does not exits".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("{} does not read image".format(img_path))
            pass
    return img

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid

# if __name__ == '__main__':
#     import datamanger
#
#     dataset = datamanger.init_img_dataset(name='market1501', root='E:/deep-person-reidentification/data')
#     imgdataset = ImageDataset(dataset.train)
#     embed()