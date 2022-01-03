
from __future__ import print_function
import os
import sys
import time
import logging
import torch
import csv
import numpy as np

term_width = 10

TOTAL_BAR_LENGTH = 25.
last_time = time.time()
begin_time = last_time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _pre_process(y_hat, y_lab):
    y_hat = np.array(y_hat, dtype=np.float64).T
    y_lab = np.array(y_lab, dtype=np.float64).T

    assert np.all(y_hat.shape == y_lab.shape)
    if len(y_hat.shape) == 1:
        y_hat = np.expand_dims(y_hat, axis=0)
        y_lab = np.expand_dims(y_lab, axis=0)
    return y_hat, y_lab


def nMAE(y_hat, y_lab):
    y_hat, y_lab = _pre_process(y_hat, y_lab)
    return np.mean(np.abs(y_hat-y_lab), 1)


def nMSE(y_hat, y_lab):
    y_hat, y_lab = _pre_process(y_hat, y_lab)
    return np.mean((y_hat - y_lab)** 2, 1)


def nICC(y_hat, y_lab, cas=3, typ=1):
    y_hat, y_lab = _pre_process(y_hat, y_lab)
    Y = np.array((y_lab, y_hat))
    # number of targets
    n = Y.shape[2]
    # mean per target
    mpt = np.mean(Y, 0)
    # print mpt.eval()
    mpr = np.mean(Y, 2)
    # print mpr.eval()
    tm = np.mean(mpt, 1)
    # within target sum sqrs
    WSS = np.sum((Y[0]-mpt)**2 + (Y[1]-mpt)**2, 1)
    # within mean sqrs
    WMS = WSS/n
    # between rater sum sqrs
    RSS = np.sum((mpr - tm)**2, 0) * n
    # between rater mean sqrs
    RMS = RSS
    # between target sum sqrs
    TM = np.tile(tm, (y_hat.shape[1], 1)).T
    BSS = np.sum((mpt - TM)**2, 1) * 2
    # between targets mean squares
    BMS = BSS / (n - 1)
    # residual sum of squares
    ESS = WSS - RSS
    # residual mean sqrs
    EMS = ESS / (n - 1)
    if cas == 1:
        if typ == 1:
            res = (BMS - WMS) / (BMS + WMS)
        if typ == 2:
            res = (BMS - WMS) / BMS
    if cas == 2:
        if typ == 1:
            res = (BMS - EMS) / (BMS + EMS + 2 * (RMS - EMS) / n)
        if typ == 2:
            res = (BMS - EMS) / (BMS + (RMS - EMS) / n)
    if cas == 3:
        if typ == 1:
            res = (BMS - EMS) / (BMS + EMS)
        if typ == 2:
            res = (BMS - EMS) / BMS

    res[np.isnan(res)] = 0
    return res.astype('float32')


def evaluate_au(prediction, target):

    if isinstance(prediction, list):
        MAE = nMAE(prediction, target)
        MSE = nMSE(prediction, target)
        ICC = nICC(prediction, target)
    else:
        prediction = prediction.detach().cpu().data.numpy()
        target = target.detach().cpu().data.numpy()

        MAE = nMAE(prediction, target)
        MSE = nMSE(prediction, target)
        ICC = nICC(prediction, target)

    return MAE, MSE, ICC


def save_checkpoint(save_dir, iter_num, net):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state = {
        'net_state_dict': net.state_dict()
    }
    save_file = os.path.join(save_dir, 'ckpt_epoch_{}.pth'.format(iter_num))
    torch.save(state, save_file)


def save_loss(save_dir, valid_type, print_value):
    save_path = os.path.join(save_dir, valid_type + '.csv')

    csvfile = open(save_path, 'a', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(print_value)
    csvfile.close()


def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir),
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging


if __name__ == '__main__':
    pass
