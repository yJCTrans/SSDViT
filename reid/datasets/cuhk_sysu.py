# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings
import pdb
from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json

class CUHK_SYSU(BaseImageDataset):
    r"""CUHK SYSU datasets.

    The dataset is collected from two sources: street snap and movie.
    In street snap, 12,490 images and 6,057 query persons were collected
    with movable cameras across hundreds of scenes while 5,694 images and
    2,375 query persons were selected from movies and TV dramas.

    Dataset statistics:
        - identities: xxx.
        - images: 12936 (train).
    """

    dataset_name = "cuhksysu"

    def __init__(self, root='', verbose=True, **kwargs):
        super(CUHK_SYSU, self).__init__()

        self.root = root
        self.dataset_dir = self.root
        self.data_dir = osp.join(self.dataset_dir, "cropped_images")

        self._check_before_run()

        train = self._process_dir(self.data_dir)
        query = []
        gallery = []

        if verbose:
            print("=> CUHK SYSU loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

    def _process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'p([-\d]+)_s(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid = int(pid) - 1
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}  # index and their corres pid

        data = []
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid = int(pid)-1

            pid = pid2label[pid]
            data.append((img_path, pid, 0, 2))
        return data

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

