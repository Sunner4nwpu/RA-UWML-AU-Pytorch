
import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F

import os
import sys
import csv
import argparse
from datetime import datetime
import time
import json
import numpy as np
import random

import model
from params import ParamsControl
from dataset_disfa import get_data_loader
from loss import UncertainLoss
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
iter_num = 0

best_valid_score_ = 0.0
best_valid_mae_ = 100000.


def main(opts):      

    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    opts.save_dir = os.path.join(opts.save_dir, opts.snapshot)
    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    logging = utils.init_log(os.path.join(opts.save_dir, '{}.log'.format(opts.snapshot)))
    _print = logging.info
    _print(opts.snapshot)

    opts.train_num = '{:0>4}'.format(opts.train_num)
    opts.save_dir = os.path.join(opts.save_dir, opts.train_num)
    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
    _print(opts.train_num)

    with open(os.path.join(opts.save_dir, 'opts_setting.txt'), 'w') as f:
        json.dump(opts.__dict__, f, indent=2)

    train_loader, n_data_train = get_data_loader(opts, 'train', opts.fold)
    valid_loader, n_data_valid = get_data_loader(opts, 'valid', opts.fold)
    _print('Train Dataset Info')
    _print('train total samples:  {}'.format(n_data_train))
    _print('train batch_size:     {}'.format(opts.batch_size_train))
    _print('train batch_num:      {}'.format(len(train_loader)))
    _print('total {}*{} = {} iters'.format(len(train_loader), opts.last_epoch, len(train_loader) * opts.last_epoch))
    _print('Valid Dataset Info')
    _print('valid total samples:  {}'.format(n_data_valid))
    _print('valid batch_size:     {}'.format(opts.batch_size_valid))
    _print('valid batch_num:      {}'.format(len(valid_loader)))
    _print('total {}*{} = {} iters'.format(len(valid_loader), opts.last_epoch, len(valid_loader) * opts.last_epoch))

    net = model.AUResnet(opts).to(device)
    params = ParamsControl(opts, net)   
    criterion = UncertainLoss(opts.target_task, opts.init_sigma).to(device)

    AU_ID = ("AU1", "AU2", "AU4", "AU5", "AU6", "AU9", "AU12", "AU15", "AU17", "AU20", "AU25", "AU26")
    utils.save_loss(opts.save_dir, 'train', ("iter_num", "losses", "iccs", "maes"))
    utils.save_loss(opts.save_dir, 'valid', ("iter_num", "icc", "mae") + AU_ID * 2)

    for epoch in range(opts.start_epoch, opts.last_epoch):
        params.update(epoch)
        train(epoch, train_loader, valid_loader, net, criterion, params, opts)

    _print('valid best_valid_score:{}'.format(best_valid_score_))
    _print('valid best_valid_mae:{}'.format(best_valid_mae_))
    _print('finish...')
    _print('------' * 20)


def train(epoch, train_loader, valid_loader, net, criterion, params, opts):
    global iter_num

    losses = utils.AverageMeter()
    maes = utils.AverageMeter()
    iccs = utils.AverageMeter()

    net.train()

    for i, data in enumerate(train_loader):
        img, label, success = data[0].to(device), data[1].to(device), data[2].to(device)
        N, _, _, _ = img.shape

        pred_mean, pred_std, pred_logits  = net(img)
        loss = criterion(pred_mean, pred_std, pred_logits, label, success)
        MAE, _, ICC = utils.evaluate_au(pred_mean, label)

        params.zero_grad()
        loss.backward()
        params.back_grad()

        losses.update(loss.item(), N)
        maes.update(MAE.mean(), N)
        iccs.update(ICC.mean(), N)

        iter_num += 1
        if iter_num % opts.print_freq == 0:
            print('{0}_Iter: [{1}][{2}]\t'
                  'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  'ICC {iccs.val:.3f} ({iccs.avg:.3f})\t'
                  'MAE {maes.val:.3f} ({maes.avg:.3f})'.
                  format('Train', iter_num, epoch+1, losses=losses, iccs=iccs, maes=maes))
            utils.save_loss(opts.save_dir, 'train', (iter_num, losses.avg, iccs.avg, maes.avg))

        if iter_num % opts.valid_freq == 0:
            valid('valid', epoch, valid_loader, net, opts)



def valid(mode, epoch, data_loader, net, opts):
    global iter_num
    global best_valid_score_
    global best_valid_mae_

    net.eval()

    total_pred = []
    total_label = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            img, label, _ = data[0].to(device), data[1].to(device), data[2].to(device)

            pred_mean, _, _ = net(img)
            total_pred += pred_mean.detach().cpu().tolist()
            total_label += label.detach().cpu().tolist()

        total_MAE, _, total_ICC = utils.evaluate_au(total_pred, total_label)

    print('{0}_Iter: [{1}][{2}]\t'
        'ICC {icc:.3f}\t'
        'MAE {mae:.3f}'.
        format(mode, iter_num, epoch+1, icc=total_ICC.mean(), mae=total_MAE.mean()))

    if total_ICC.mean() > best_valid_score_:
        best_valid_score_ = total_ICC.mean()
        best_valid_mae_ = total_MAE.mean()

    if opts.save_checkpoint:
        utils.save_checkpoint(opts.save_dir, iter_num, net)

    utils.save_loss(opts.save_dir, mode, (iter_num, total_ICC.mean(), total_MAE.mean()) + tuple(total_ICC.tolist()) + tuple(total_MAE.tolist()))

    net.train()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AU_DISFA')
    parser.add_argument('--snapshot', type=str, default='DISFA_CCNN')
    parser.add_argument('--train_num', type=int, default=1)
    parser.add_argument('--batch_size_train', type=int, default=32)
    parser.add_argument('--batch_size_valid', type=int, default=1024)
    parser.add_argument('--lr_backbone', type=str, default='3e-4,3e-5')
    parser.add_argument('--lr_new', type=str, default='3e-4,3e-5')
    parser.add_argument('--lr_patch', type=str, default='0.0,0.0')
    parser.add_argument('--lr_attention', type=str, default='3e-4,3e-5')
    parser.add_argument('--use_Adam', type=int, default=1)
    parser.add_argument('--decay_backbone', type=str, default='10')
    parser.add_argument('--decay_new', type=str, default='10')
    parser.add_argument('--decay_patch', type=str, default='10')
    parser.add_argument('--decay_attention', type=str, default='10')
    parser.add_argument('--momentum', type=str, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=500)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--last_epoch', type=int, default=10)

    parser.add_argument('--json_dir', type=str, default='../preprocess/Disfa/Dataset')
    parser.add_argument('--json_name', type=str, default='Disfa_Right_112_0.55.json')
    parser.add_argument('--use_sampler', type=int, default=1)    
    parser.add_argument('--fold', type=str, default='CCNN')    

    parser.add_argument('--patch_number', type=int, default=3)
    parser.add_argument('--individual_featured', type=int, default=256)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--target_task', type=str, default='regress,uncertain')
    parser.add_argument('--strides', type=str, default='1,2,2,1')
    parser.add_argument('--pretrain_backbone', type=str, default='./models/pretrained_resnet18.ckpt')
    parser.add_argument('--block_num', type=int, default=2)
    parser.add_argument('--head_number', type=int, default=2)
    parser.add_argument('--init_sigma', type=float, default=-2.0)

    parser.add_argument('--resume_net', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='../results')
    parser.add_argument('--save_checkpoint', type=int, default=0)

    opts = parser.parse_args()

    main(opts)
