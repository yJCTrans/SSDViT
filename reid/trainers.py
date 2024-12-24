from __future__ import print_function, absolute_import, division
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .utils.meters import AverageMeter
from .models import *
from .evaluation_metrics import accuracy
import pdb
from torch.nn.utils import clip_grad_norm_

class Trainer(object):
    def __init__(self, args, model, memory, criterion, clip_value=16.0):
        super(Trainer, self).__init__()
        self.model = model
        self.memory = memory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.clip_value = clip_value
        self.args = args
        self.sm = nn.Softmax(dim=1)
        self.kl_loss = nn.KLDivLoss(size_average=False)
        self.log_sm = nn.LogSoftmax(dim=1)

    def train(self, epoch, data_loaders, data_loaders_Imagenet, optimizer, print_freq=10, train_iters=400):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        source_count = len(data_loaders)
        print('source_count={}'.format(source_count))
        end = time.time()
        for i in range(train_iters):

            if True:
                data_loader_index = [i for i in range(source_count)]
                batch_data = [data_loaders[i].next() for i in range(source_count)]
                batch_data_Imagenet = [data_loaders_Imagenet[i].next() for i in range(source_count)]

                data_time.update(time.time() - end)

                loss_train = 0.
                for t in data_loader_index: # 0 1 2
                    data_time.update(time.time() - end)

                    traininputs = batch_data[t]
                    traininputs_Imagenet = batch_data_Imagenet[t]
                    inputs, targets = self._parse_data(traininputs)
                    inputs_Imagenet, targets_Imagenet = self._parse_data(traininputs_Imagenet)

                    if self.args.SD:
                        alpha_T = 0.8
                        n_steps = self.args.epochs
                        n_classes = 768
                        alpha_t = alpha_T * ((epoch + 1) / n_steps)
                        alpha_t = max(0, alpha_t)

                        f_out, f_out_rb, tri_features, rb_features = self.model(inputs)
                        f_out_Imagenet, _, _, _ = self.model(inputs_Imagenet)

                        loss_tri1 = self.criterion(tri_features, targets)
                        loss_tri2 = self.criterion(rb_features, targets)
                        loss_tri = 0.8 * loss_tri1 + 0.2 * loss_tri2
                        loss_s1 = self.memory[t](f_out, targets).mean()
                        loss_s2 = self.memory[t](f_out_Imagenet, targets_Imagenet).mean()
                        loss_s = 0.5 * loss_s1 + 0.5 * loss_s2

                        targets_numpy = targets.cpu().detach()
                        targets_numpy = targets_numpy.clamp(0, n_classes - 1)
                        identity_matrix = torch.eye(n_classes)
                        targets_one_hot = identity_matrix[targets_numpy]

                        soft_output = ((1 - alpha_t) * targets_one_hot).to('cuda') + (alpha_t * F.softmax(f_out, dim=1))
                        soft_output_rb = ((1 - alpha_t) * targets_one_hot).to('cuda') + (alpha_t * F.softmax(f_out_rb, dim=1))
                        loss_rb = F.kl_div(
                            torch.log(soft_output_rb),
                            torch.log(soft_output),
                            reduction='sum',
                            log_target=True
                        ) * (self.args.sd_temp * self.args.sd_temp) / f_out_rb.numel()
                        loss_rb = self.args.sd_weight * loss_rb

                        targets_numpy1 = targets_Imagenet.cpu().detach()
                        targets_numpy1 = targets_numpy1.clamp(0, n_classes - 1)
                        identity_matrix = torch.eye(n_classes)
                        targets_one_hot1 = identity_matrix[targets_numpy1]

                        soft_output_Imagenet = ((1 - alpha_t) * targets_one_hot1).to('cuda') + (alpha_t * F.softmax(f_out_Imagenet, dim=1))

                        with torch.no_grad():
                            loss_aug = F.kl_div(
                                torch.log(soft_output_Imagenet),
                                torch.log(soft_output),
                                reduction='sum',
                                log_target=True
                            ) * (self.args.sd_temp * self.args.sd_temp) / f_out_Imagenet.numel()
                            loss_aug = self.args.sd_weight * loss_aug

                        loss_train = loss_train + loss_s + loss_tri + loss_rb + loss_aug

                    else:
                        f_out, tri_features = self.model(inputs)
                        loss_tri = self.criterion(tri_features, targets)
                        loss_s = self.memory[t](f_out, targets).mean()
                        loss_train = loss_train + loss_s + loss_tri

                loss_train = loss_train / source_count

                optimizer.zero_grad()

                loss_train.backward()
                optimizer.step()

                losses.update(loss_train.item())

                with torch.no_grad():
                    for m_ind in range(source_count):
                        imgs_Imagenet, pids_Imagenet = self._parse_data(batch_data_Imagenet[m_ind])
                        if self.args.SD:
                            f_new_Imagenet, _, _, _ = self.model(imgs_Imagenet)
                        else:
                            f_new, _ = self.model(imgs_Imagenet)
                        self.memory[m_ind].module.MomentumUpdate(f_new_Imagenet, pids_Imagenet)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Total loss {:.3f} ({:.3f})\t'
                      'loss_s {:.3f}\t'
                      'loss_tri {:.3f}\t'
                      'loss_rb {:.3e}\t'
                      'loss_aug {:.3f}\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg,
                              loss_s,
                              loss_tri,
                              loss_rb,
                              loss_aug
                              ))
    def _parse_data(self, inputs):
        imgs, _, pids, _, _ = inputs

        imgs = imgs.cuda()
        pids = pids.cuda()

        return imgs, pids



