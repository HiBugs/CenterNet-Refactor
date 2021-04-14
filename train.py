#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu@zju.edu.cn
@Date: 2020/11/27
@Description:
"""
import os
import argparse
import numpy as np
from progress.bar import Bar
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from networks.resnet import ResNet
from networks.dla_dcn import DLANet
from networks.losses import CtdetLoss, AverageMeter
# from datasets.face_dataset import FaceDataset
from datasets.coco import COCODataset


parser = argparse.ArgumentParser(description='Object Detection')
parser.add_argument('--data_dir', default="../datasets/COCO/coco2017", help='data dir')
parser.add_argument('--save_path', default="saved_models/COCO2017", help='saved models dir')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=16, help='testing batch size')
parser.add_argument('--num_epochs', type=int, default=140, help='the starting epoch count')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--pretrain', action='store_true', help='use fr pretrain?')
parser.add_argument('--seed', type=int, default=666, help='random seed to use. Default=666')
parser.add_argument('--cuda', type=str, default="0, 1", help='use cuda')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)


def adjust_learning_rate(optimizer, epoch):
    # if epoch in [int(opt.num_epochs * 0.4), int(opt.num_epochs * 0.7), int(opt.num_epochs * 0.9)]:
    if epoch in [90, 120]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def adjust_optimizer(model):
    params = []
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        params += [{'params': [value], 'lr': opt.lr}]
    # optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(params, lr=opt.lr)
    return optimizer


def train(train_loader, model, criterion,  optimizer, epoch):
    total_loss = AverageMeter()
    num_iters = len(train_loader)
    bar = Bar('{}/{}'.format(epoch, opt.num_epochs), max=num_iters)

    model.train()
    for i, sample in enumerate(train_loader):
        for k in sample:
            sample[k] = sample[k].to(device=device, non_blocking=True)
        pred = model(sample['input'])
        loss, hm_loss, wh_loss, off_loss = criterion(pred, sample)
        total_loss.update(loss.item(), sample['input'].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Bar.suffix = '{phase}: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            i + 1, num_iters, phase='train',
            total=bar.elapsed_td, eta=bar.eta_td)
        Bar.suffix = Bar.suffix + ('| total_loss: {:.4f}, hm_loss:{:.2f}, wh_loss:{:.2f}, off_loss:{:.2f}'
                                   .format(total_loss.avg, hm_loss.item(), wh_loss.item(), off_loss.item()))
        bar.next()

    bar.finish()


def validate(val_loader, model, criterion):
    total_loss = 0.

    model.eval()
    for i, sample in enumerate(val_loader):
        for k in sample:
            sample[k] = sample[k].to(device=device, non_blocking=True)

        pred = model(sample['input'])
        loss = criterion(pred, sample)
        total_loss += loss[0].item()

    return total_loss/len(val_loader)


# train_dataset = FaceDataset(data_dir=opt.data_dir, split='train')
train_dataset = COCODataset(data_dir=opt.data_dir, split='train')
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
# test_dataset = FaceDataset(data_dir=opt.data_dir, split='val')
test_dataset = COCODataset(data_dir=opt.data_dir, split='val')
test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=4)

# model = ResNet(18, heads={'hm': COCODataset.num_classes, 'wh': 2, 'reg': 2}, pretrained=True)
model = DLANet(34, heads={'hm': COCODataset.num_classes, 'wh': 2, 'reg': 2}, pretrained=True)
model = torch.nn.DataParallel(model).cuda()
# loss_weight = {'hm_weight': 1, 'wh_weight': 0.1, 'ang_weight': 0.1, 'reg_weight': 0.1}
loss_weight = {'hm_weight': 1, 'wh_weight': 0.1, 'reg_weight': 0.1}
criterion = CtdetLoss(loss_weight)
optimizer = adjust_optimizer(model)

best_test_loss = np.inf
for epoch in range(opt.num_epochs):
    adjust_learning_rate(optimizer, epoch)

    train(train_loader, model, criterion, optimizer, epoch)
    # torch.save(model.state_dict(), os.path.join(opt.save_path, 'last_dla34.pth'))
    test_loss = validate(test_loader, model, criterion)

    if best_test_loss > test_loss:
        best_test_loss = test_loss
        print('get best test loss %.5f' % best_test_loss)
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'coco2017_best_dla34.pth'))
    # else:
    #     torch.save(model.state_dict(), os.path.join(opt.save_path, 'last_resnet18.pth'))
