# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat
import torch.nn.functional as F
import numpy as np


def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1)
  neg_inds = gt.lt(1)

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

def _var_reg_loss(regr, gt_regr, mask):
    ''' var L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    pred = regr[:, :, :2]
    var = regr[:, :, 2:]
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    var = var * mask
    pred = pred * mask
    gt_regr = gt_regr * mask

    target = torch.abs(pred - gt_regr)
    inds = target.gt(1).float()
    regr = (target - 0.5) * inds + 0.5 * (1-inds) * torch.pow(target, 2)
    var_regr = torch.exp(-1*var) * regr + 0.5 * var
    regr_loss = var_regr.sum() / (num + 1e-4)
    # regr_loss = nn.functional.smooth_l1_loss(pred, gt_regr, size_average=False)
    # regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss


class NormRegLoss(nn.Module):

    def __init__(self):
        super(NormRegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred / (target + 1e-4), target * 0 + 1, mask)
        return loss

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss

class VarRegLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(VarRegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = _var_reg_loss(pred, target, mask)
        return loss

class VarRegL1Loss(nn.Module):
    def __init__(self):
        super(VarRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pass

class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


class NormRegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(NormRegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res

class ReducedFocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(ReducedFocalLoss, self).__init__()
    self.neg_loss = _kalyo_neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

def _kalyo_neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt[:, 2:3, :, :]
  neg_inds = gt[:, 1:2, :, :] - gt[:, 2:3, :, :]

  neg_weights = torch.pow(1 - gt[:, 0:1, :, :], 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss, neg_loss = pos_loss.sum(), neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

class GaussianwhLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self, num_scale):
        super(GaussianwhLoss, self).__init__()
        self.num_scale = num_scale

    def forward(self, output, target):
        sigma = torch.sigmoid(output[:, self.num_scale:, :, :])
        output = output[:, :self.num_scale, :, :]

        loss = - torch.log(self._gaussian_dist_pdf(output, target[:, :self.num_scale, :, :], sigma) + 1e-9) / 2.0
        loss *= target[:, self.num_scale:, :, :]

        assigned_boxes = target[:, self.num_scale:, :, :].sum()
        # class_loss = 0. * tf.reduce_sum(l1_loss) / tf.maximum(1.0, assigned_boxes)
        class_loss = loss.sum() / max(1.0, assigned_boxes)

        return class_loss

    def _gaussian_dist_pdf(self, val, mean, var):
        return torch.exp(- (val - mean) ** 2.0 / (var + 1e-9) / 2.0) / torch.sqrt(2.0 * np.pi * (var + 1e-9))

class CaltechwhLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self, num_scale):
        super(CaltechwhLoss, self).__init__()
        self.num_scale = num_scale

    def forward(self, output, target):
        absolute_loss = torch.abs(target[:, 0:self.num_scale, :, :] - output) / (target[:, 0:self.num_scale, :, :] + 1e-10)
        square_loss = 0.5 * ((target[:, 0:self.num_scale, :, :] - output[:, 0:self.num_scale, :, :]) / (target[:, 0:self.num_scale, :, :] + 1e-10)) ** 2

        inds = absolute_loss.lt(1).float()
        l1_loss = target[:, self.num_scale:, :, :] * (inds * square_loss + (1 - inds) * (absolute_loss - 0.5))

        assigned_boxes = target[:, self.num_scale:, :, :].sum()
        # class_loss = 0. * tf.reduce_sum(l1_loss) / tf.maximum(1.0, assigned_boxes)
        class_loss = l1_loss.sum() / max(1.0, assigned_boxes)

        return class_loss

class CaltechRegLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(CaltechRegLoss, self).__init__()

    def forward(self, output, target):
        absolute_loss = torch.abs(target[:, :2, :, :] - output[:, :, :, :])
        square_loss = 0.5 * (target[:, :2, :, :] - output[:, :, :, :]) ** 2

        inds = absolute_loss.lt(1).float()
        l1_loss = target[:, 2:3, :, :] * (inds * square_loss + (1 - inds) * (absolute_loss - 0.5))
        # l1_loss = y_true[:, :, :, 2] * tf.reduce_sum(absolute_loss, axis=-1)

        assigned_boxes = target[:, 2:3, :, :].sum()
        # class_loss = 0.1*tf.reduce_sum(l1_loss) / tf.maximum(1.0, assigned_boxes)
        class_loss = l1_loss.sum() / max(1.0, assigned_boxes)

        return class_loss

class densityLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(densityLoss, self).__init__()

    def forward(self, output, target):
        if output.shape[1] != 1:
            output = output.pow(2).sum(dim=1, keepdim=True).sqrt()
        pos_inds = target[:, 2:3, :, :]
        neg_inds = target[:, 1:2, :, :] - target[:, 2:3, :, :]

        neg_weights = torch.pow(1 - target[:, 0:1, :, :], 4)

        absolute_loss = torch.abs(target[:, 3:4, :, :] - output)
        square_loss = 0.5 * (target[:, 3:4, :, :] - output) ** 2

        inds = absolute_loss.lt(1).float()
        l1_loss = (inds * square_loss + (1 - inds) * (absolute_loss - 0.5))

        # pos_loss here
        # pos_loss = l1_loss * torch.pow(1-output, 2) * pos_inds
        pos_loss = l1_loss * pos_inds
        neg_loss = l1_loss * torch.pow(output, 2) * neg_weights * neg_inds


        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        # initialize should be also modified
        # loss = (neg_loss + pos_loss) / max(1.0, num_pos)
        loss = pos_loss / max(1.0, num_pos)
        return loss

class AELoss(nn.Module):
    def __init__(self):
        super(AELoss, self).__init__()
        self.eps = 1e-10

    def forward(self, output, tag_pull, tag_push, mask_pull, mask_push):
        # output: Bx1xHxW
        # pull_num = mask_pull.sum(dim=1, keepdim=True).float()
        if output.shape[1] != 1:
            norm = output.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
            output = torch.div(output, norm)

        pull_num = mask_pull.sum().float()
        pull_0 = _tranpose_and_gather_feat(output, tag_pull[:, :, 0])
        pull_1 = _tranpose_and_gather_feat(output, tag_pull[:, :, 1])
        pull_loss = (pull_0 - pull_1).pow(2).sum(dim=2) / (pull_num + 1e-4)
        # pull_loss = torch.pow(pull_0 - pull_1, 2) / (pull_num + 1e-4)
        pull_loss = pull_loss[mask_pull].sum()

        # push_num = mask_push.sum(dim=1, keepdim=True).float()
        push_num = mask_push.sum().float()
        push_0 = _tranpose_and_gather_feat(output, tag_push[:, :, 0])
        push_1 = _tranpose_and_gather_feat(output, tag_push[:, :, 1])
        push_loss = (1 - (push_0 - push_1).abs().sum(dim=2)) / (push_num + 1e-4)
        push_loss = nn.functional.relu(push_loss, inplace=True)
        push_loss = push_loss[mask_push].sum()

        # print('pull_num: {}, push_num: {}'.format(pull_num, push_num))
        # print('pull_loss: {}, push_loss: {}'.format(pull_loss, push_loss))

        return pull_loss + push_loss

class GHMCLoss(nn.Module):
    def __init__(self, opt):
        super(GHMCLoss, self).__init__()
        self.bins = opt.bins
        self.momentum = opt.momentum
        self.edges = [float(x) / self.bins for x in range(self.bins+1)]
        self.edges[-1] += 1e-6
        if self.momentum > 0:
            self.acc_sum = [0.0 for _ in range(self.bins)]

    def forward(self, pred, target):
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        pos_inds = target[:, 2:3, :, :]
        neg_inds = target[:, 1:2, :, :] - target[:, 2:3, :, :]

        neg_weights = torch.pow(1 - target[:, 0:1, :, :], 4)

        # g = torch.abs((pred.detach() - 1) * pos_inds + pred.detach() * neg_inds)
        g = torch.abs((pred.detach() * target[:, 1:2, :, :] - pos_inds))

        valid = target[:, 1:2, :, :] > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0 # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        if self.bins == 0:
            weights = torch.zeros_like(pred)

        loss = 0
        pos_loss = torch.log(pred) * pos_inds * weights
        neg_loss = torch.log(1 - pred) * neg_inds * neg_weights * weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss / tot
        else:
            loss = loss - (pos_loss + neg_loss) / tot
        return loss


class GHMRLoss(nn.Module):
    def __init__(self, opt):
        super(GHMRLoss, self).__init__()
        self.mu = opt.mu
        self.bins = opt.bins
        if self.bins == 0:
            self.edges = [1.]
        else:
            self.edges = [float(x) / self.bins for x in range(self.bins+1)]
        self.edges[-1] = 1e3
        self.momentum = opt.momentum
        if self.momentum > 0:
            self.acc_sum = [0.0 for _ in range(self.bins)]

    def forward(self, pred, target):
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = (pred - target[:, 0:1, :, :]) / (target[:, 0:1, :, :] + 1e-4)
        temp = torch.sqrt(diff * diff + mu * mu)
        loss = temp - mu

        # gradient length
        g = torch.abs(diff / temp).detach()
        weights = torch.zeros_like(g)

        valid = target[:, 1:2, :, :] > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0

        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1-mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / self.acc_sum[i]
                n += 1
        if n > 0:
            weights /= n

        if self.bins == 0:
            weights = torch.ones_like(g)

        loss = loss * weights * target[:, 1:2, :, :]
        loss = loss.sum() / tot

        return loss

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        # different from data_generator since here is y,x not x,y
        # self.Off = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32).view(1, 1, 4, 2).to('cuda')

    def forward(self, output_h, output_off, target_h, target_off, attract, repel, mask_attract, mask_repel, pre_off):
        attract_num = mask_attract.sum().float()

        attract_h = _tranpose_and_gather_feat(output_h, attract.view(attract.size(0), -1)).view(attract.size(0), attract.size(1), attract.size(2), 1) # B x 512 x 4 x 1
        # attract_h = _tranpose_and_gather_feat(target_h[:, :1, :, :], attract.view(attract.size(0), -1)).view(attract.size(0), attract.size(1), attract.size(2), 1) # B x 512 x 4 x 1
        attract_h_mean = attract_h.mean(2, keepdim=True) # B x 512 x 1 x 1
        attract_off = _tranpose_and_gather_feat(output_off, attract.view(attract.size(0), -1)).view(attract.size(0), attract.size(1), attract.size(2), -1) # B x 512 x 4 x 2
        # attract_off = _tranpose_and_gather_feat(target_off[:, :2, :, :], attract.view(attract.size(0), -1)).view(attract.size(0), attract.size(1), attract.size(2), -1) # B x 512 x 4 x 2
        # attract_off *=  torch.tensor([[1, 1], [-1, -1], [1, 1], [-1, -1]], dtype=torch.float32).view(1, 1, 4, 2).to('cuda')
        attract_off +=  torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32).view(1, 1, 4, 2).to('cuda')
        attract_off_mean = attract_off.mean(2, keepdim=True) # B x 512 x 1 x 2

        attract_iou = iou(torch.cat((torch.exp(attract_h), attract_off), dim=-1), torch.cat((torch.exp(attract_h_mean), attract_off_mean), dim=-1))
        attract_loss = (1 - attract_iou) / (attract_num + 1e-4)
        attract_loss = attract_loss[mask_attract].sum()

        repel_num = mask_repel.sum().float()
        repel_h = _tranpose_and_gather_feat(output_h, repel.view(repel.size(0), -1)).view(repel.size(0),
                                                                                          repel.size(1),
                                                                                          repel.size(2),
                                                                                          repel.size(3),
                                                                                          1) # B x 512 x 2 x 4 x 1
        repel_h_mean = repel_h.mean(3, keepdim=True)

        repel_off = _tranpose_and_gather_feat(output_off, repel.view(repel.size(0), -1)).view(
            repel.size(0), repel.size(1), repel.size(2), repel.size(3), -1)
        repel_off += torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32).view(1, 1, 1, 4, 2).to('cuda')

        # pre_off should be y,x order
        repel_off_mean = repel_off.mean(3, keepdim=True) # B x 512 x 2 x 1 x 2
        repel_off_mean[:, :, 1:2, :, :] += pre_off.view(pre_off.size(0), pre_off.size(1), 1, 1, pre_off.size(2))

        box = torch.cat((torch.exp(repel_h_mean), repel_off_mean), dim=-1)
        repel_iou = iou(box[:, :, 0, :, :], box[:, :, 1, :, :])
        repel_loss = repel_iou / (repel_num + 1e-4)
        repel_loss = repel_loss[mask_repel].sum()

        print('attract_loss: {}'.format(attract_loss))
        print('repel_loss: {}'.format(repel_loss))

        return attract_loss + repel_loss

def iou(BoxA, BoxB):
    """
    input shape: B x 512 x 4 x 3
    :param BoxA: h, o_y, o_x
    :param BoxB: h, o_y, o_x
    :return: iou: B x 512 x 4
    """
    areaA = BoxA[:, :, :, 0] * BoxA[:, :, :, 0] * 0.41
    areaB = BoxB[:, :, :, 0] * BoxB[:, :, :, 0] * 0.41

    BoxA_y_min = BoxA[:, :, :, 1] - BoxA[:, :, :, 0] / 2
    BoxA_x_min = BoxA[:, :, :, 2] - 0.41 * BoxA[:, :, :, 0] / 2
    BoxA_y_max = BoxA[:, :, :, 1] + BoxA[:, :, :, 0] / 2
    BoxA_x_max = BoxA[:, :, :, 2] + 0.41 * BoxA[:, :, :, 0] / 2

    BoxB_y_min = BoxB[:, :, :, 1] - BoxB[:, :, :, 0] / 2
    BoxB_x_min = BoxB[:, :, :, 2] - 0.41 * BoxB[:, :, :, 0] / 2
    BoxB_y_max = BoxB[:, :, :, 1] + BoxB[:, :, :, 0] / 2
    BoxB_x_max = BoxB[:, :, :, 2] + 0.41 * BoxB[:, :, :, 0] / 2

    y_min_max = torch.max(BoxA_y_min, BoxB_y_min)
    x_min_max = torch.max(BoxA_x_min, BoxB_x_min)
    y_max_min = torch.min(BoxA_y_max, BoxB_y_max)
    x_max_min = torch.min(BoxA_x_max, BoxB_x_max)

    I = torch.clamp((y_max_min - y_min_max), min=0) * torch.clamp((x_max_min - x_min_max), min=0)
    U = areaA + areaB - I
    iou = I / (U + 1e-6)

    return iou