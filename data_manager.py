
import os
import os.path as osp
import numpy as np
import glob
import re
from utils import mkdir_if_missing, write_json, read_json

from IPython import embed

class Market1501(object):

    dataset_dir = 'Market-1501'

    def __init__(self, root='data', **kwargs): # not sure parameters
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        # 文件路径 相机id 行人id 图片的数量
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}'is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        #glob.glob()查询一个匹配路径
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # re.compile 将正则表达式编译为一个pattern对象
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container =set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(img_paths)
        return dataset, num_pids, num_imgs

class CUHK03(object):
    def __init__(self):
        print('CUHK03')


class DukeMTMCreID(object):
    def __init__(self):
        print('DukeMTMCreID')

class MSMT17(object):
    def __init__(self):
        print('MSMT17')

class Mars(object):
    def __init__(self):
        print('Mars')

class iLIDSVID(object):
    def __init__(self):
        print('iLIDSVID')

class PRID(object):
    def __init__(self):
        print('PRID')

class DukeMTMCVidReID(object):
    def __init__(self):
        print('DukeMTMCVidReID')

# if __name__ == '__main__':
#     data = Market1501(root='E:/deep-person-reidentification/data')
"""Create dataset"""

__img_factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmcreid': DukeMTMCreID,
    'msmt17': MSMT17,
}

__vid_factory = {
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'prid': PRID,
    'dukemtmcvidreid': DukeMTMCVidReID,
}

def get_names():
    return __img_factory.keys() + __vid_factory.keys()

def init_img_dataset(name, **kwargs):
    if name not in __img_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))
    return __img_factory[name](**kwargs)

def init_vid_dataset(name, **kwargs):
    if name not in __vid_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __vid_factory.keys()))
    return __vid_factory[name](**kwargs)
