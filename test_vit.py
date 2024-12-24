from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2
from torchvision.transforms import InterpolationMode
import os
import random
import numpy as np
from PIL import Image
import random
import sys
import time
from datetime import timedelta
from reid import datasets
from reid import models
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint
from GradCAM import *
import matplotlib
matplotlib.use('Agg')
import pdb

start_epoch = best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, norm=True, BNNeck=args.BNNeck, SD=args.SD)

    model.cuda()
    model = nn.DataParallel(model)
    return model

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)

def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log_text.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    print("==> Load datasets")

    dataset = get_data(args.dataset, args.data_dir)     # load target dataset
    test_loader = get_test_loader(dataset, args.height, args.width, args.test_batch_size, args.workers)

    # Create model
    model = create_model(args)
    print(model)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.module.parameters()) / 1000000.0))

    # Evaluator
    evaluator = Evaluator(model)

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DG learner")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())

    parser.add_argument('--test-batch-size', type=int, default=256)    # 256
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")  #  256
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")      # 64/4=16 ids

    # model
    parser.add_argument('-a', '--arch', type=str, default='vit_base',
                        choices=models.names())
    parser.add_argument('--BNNeck', action='store_true',
                        help="use triplet and BNNeck")
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")

    ##loss
    parser.add_argument('--margin', type=float, default=0.3,
                        help="margin of the triplet loss, default: 0.3")

    ## self distillation loss
    parser.add_argument('--SD', action='store_true', default=False,
                        help="whether to use self distill loss ")
    parser.add_argument('--sd_temp', type=float, default=5.0,
                        help="temperature tao of self distill loss, default: 5.0")
    parser.add_argument('--sd_weight', type=float, default=0.2,
                        help="weight lamda of self disitll loss, default: 0.2")

    # random occlusion
    parser.add_argument('--min_size', type=float, default=0, help="minimal size for the random occlusion, default: 0")
    parser.add_argument('--max_size', type=float, default=0.8,
                        help="maximal size for the ramdom occlusion. default: 0.8")
    parser.add_argument('--clip_value', type=float, default=0, help="the gradient clip value, default: 8")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.008,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--iters', type=int, default=200)

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=5)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='./data')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, './logs/70'))
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    main()
