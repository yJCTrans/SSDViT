from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import os

import random
import numpy as np
from PIL import Image
import random
import sys
import collections
import copy
import time
from datetime import timedelta
from reid import datasets
from reid import models
from reid.models.memory import MemoryClassifier
from reid.trainers import Trainer
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import IterLoader
from reid.utils.data import transforms as T
from reid.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.loss.triplet import TripletLoss, WeightedRegularizedTriplet, TripletLoss_Soft
from reid.solver.lr_scheduler import WarmupMultiStepLR, CosineLRScheduler
from reid.autoaugment import ImageNetPolicy
# import kmeans1d
import pdb

start_epoch = best_mAP = 0

def get_data(name, data_dir, combine_all= False):
    root = osp.join(data_dir, name)
    # dataset = datasets.create(name, root, combine_all=combine_all)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(5),
        T.ColorJitter(brightness=(0.5, 2.0), contrast=(0.5, 2.0), saturation=(0.5, 2.0), hue=(-0.1, 0.1)),
        T.RandomOcclusion(args.min_size, args.max_size),
        T.ToTensor(),
        normalizer
    ])
    train_transformer_Imagenet = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(0.5),
        ImageNetPolicy(),
        T.RandomRotation(5),
        T.ColorJitter(brightness=(0.5, 2.0), contrast=(0.5, 2.0), saturation=(0.5, 2.0), hue=(-0.1, 0.1)),
        T.RandomOcclusion(args.min_size, args.max_size),
        T.ToTensor(),
        normalizer
    ])

    train_set = sorted(dataset.mix_dataset) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        # sampler = RandomMultipleGallerySampler(train_set, num_instances)
        sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=None)

    train_loader_Imagenet = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer_Imagenet),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=None)
    return train_loader, train_loader_Imagenet

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

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.copyWeight_eval(checkpoint['state_dict'])

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

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load datasets")

    if args.multi_source:                         # for multi-source
        dataset_src1 = get_data(args.dataset_src1, args.data_dir)
        dataset_src2 = get_data(args.dataset_src2, args.data_dir)
        dataset_src3 = get_data(args.dataset_src3, args.data_dir)


        datasets_src = [dataset_src1, dataset_src2, dataset_src3]

        train_loader_src1, train_loader_Imagenet1 = get_train_loader(args, dataset_src1, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters)
        train_loader_src2, train_loader_Imagenet2 = get_train_loader(args, dataset_src2, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters)
        train_loader_src3, train_loader_Imagenet3 = get_train_loader(args, dataset_src3, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters)


        train_loader = [train_loader_src1, train_loader_src2, train_loader_src3]
        train_loader_Imagenet = [train_loader_Imagenet1, train_loader_Imagenet2, train_loader_Imagenet3]
        num_classes1 = dataset_src1.num_mix_pids
        num_classes2 = dataset_src2.num_mix_pids
        num_classes3 = dataset_src3.num_mix_pids

        num_classes = [num_classes1, num_classes2, num_classes3]
        print(' number classes = ', num_classes)
    else:
        dataset_src = get_data(args.dataset_src1, args.data_dir, combine_all=True)     # for single source
        datasets_src = [dataset_src]
        train_loader_src, train_loader_Imagenet1 = get_train_loader(args, dataset_src, args.height, args.width,
                                             args.batch_size, args.workers, args.num_instances, iters)
        train_loader = [train_loader_src]
        train_loader_Imagenet = [train_loader_Imagenet1]
        num_classes = dataset_src.num_mix_pids
        num_classes = [num_classes]
        print(' number classes = ', num_classes)

    dataset = get_data(args.dataset, args.data_dir)     # load target dataset
    test_loader = get_test_loader(dataset, args.height, args.width, args.test_batch_size, args.workers)

    # Create model
    model = create_model(args)
    print(model)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.module.parameters()) / 1000000.0))

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
        return

    print("==> Initialize source-domain class centroids and memorys ")

    source_centers_all = []
    memories = []

    for dataset_i in range(len(datasets_src)):

        dataset_source = datasets_src[dataset_i]
        sour_cluster_loader = get_test_loader(dataset_source, args.height, args.width,
                                              args.test_batch_size, args.workers,
                                              testset=sorted(dataset_source.mix_dataset))
        print('sour_cluster_loader长度={}'.format(len(sour_cluster_loader)))
        source_features, _ = extract_features(model, sour_cluster_loader, print_freq=50)
        sour_fea_dict = collections.defaultdict(list)

        for f, pid, _ in sorted(dataset_source.mix_dataset):  # f: filename pid:1(person id )
            sour_fea_dict[pid].append(source_features[f].unsqueeze(0))

        source_centers = [torch.cat(sour_fea_dict[pid], 0).mean(0) for pid in
                          sorted(sour_fea_dict.keys())]
        source_centers = torch.stack(source_centers, 0)
        source_centers = F.normalize(source_centers, dim=1).cuda()
        source_centers_all.append(source_centers)  # save the centroids

        # Create  memory
        curMemo = MemoryClassifier(model.module.num_features, source_centers.shape[0],
                                  temp=args.temp, momentum=args.momentum).cuda()
        curMemo.features = source_centers
        curMemo.labels = torch.arange(num_classes[dataset_i]).cuda()
        curMemo = nn.DataParallel(curMemo)  # current memory
        memories.append(curMemo)

        del source_centers, sour_cluster_loader, sour_fea_dict

    # Optimizer
    params = [{"params": [value]} for value in model.module.parameters() if value.requires_grad]

    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        lr_min=0.002 * args.lr,
        t_mul=1.,
        decay_rate=0.1,
        warmup_lr_init=0.01 * args.lr,
        warmup_t=5,
        cycle_limit=1,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.,
        noise_seed=42,
    )

    criterion = TripletLoss(args.margin, args.num_instances, False).cuda()

    trainer = Trainer(args, model, memories, criterion)

    for epoch in range(args.epochs):
        # Calculate distance
        print('==> start training epoch {} \t ==> learning rate = {}'.format(epoch, optimizer.param_groups[0]['lr']))

        torch.cuda.empty_cache()

        trainer.train(epoch, train_loader, train_loader_Imagenet, optimizer,
                      print_freq=args.print_freq, train_iters=args.iters)

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step(epoch)

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DG learner")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='',
                        choices=datasets.names())

    parser.add_argument('--dataset_src1', type=str, default='',
                        choices=datasets.names())
    parser.add_argument('--dataset_src2', type=str, default='',
                        choices=datasets.names())
    parser.add_argument('--dataset_src3', type=str, default='',
                        choices=datasets.names())

    parser.add_argument('--combine_all', action='store_true', default=False,
                        help="combine all data for training, default: False")
    parser.add_argument('--multi_source', action='store_true', default=False,
                        help="multiple source datasets for training, default: False")

    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")

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
