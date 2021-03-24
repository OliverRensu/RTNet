import pytorch_ssim
import pytorch_iou
import torch.nn as nn
import torch
bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def bce_ssim_loss(pred,target):
    loss=0
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss += bce_out + ssim_out + iou_out
    return loss


def muti_bce_loss_fusion(pred, labels_v):
    S = len(pred)
    loss=0
    loss0 = bce_ssim_loss(pred[0], labels_v)
    for i in range(1, S):
        loss += bce_ssim_loss(pred[i], labels_v)
    loss += loss0
    return loss0, loss
