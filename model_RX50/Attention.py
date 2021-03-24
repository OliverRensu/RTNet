import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RScaling(nn.Module):
    def __init__(self, in_planes, ratio):
        super(RScaling, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes*2, in_planes // ratio, 1)
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes*2, 1)
        self.in_planes=in_planes
    def forward(self, img, flow):
        f = self.fc2(self.relu1(self.fc1(self.avg_pool(torch.cat([img, flow], 1)))))#B 2C 1 1
        return img.mul(F.sigmoid(f[:, :self.in_planes, :, :]))+img, \
               flow.mul(F.sigmoid(f[:, self.in_planes:, :, :]))+flow

class RGating(nn.Module):
    def __init__(self, in_planes):
        super(RGating, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_planes*2, 2, kernel_size=1), nn.Sigmoid())
    def forward(self, img, flow):
        gate = self.conv(torch.cat([img, flow], 1))
        return img.mul(gate[:,:1,:,:])+img, flow.mul(gate[:,1:,:,:])+flow


class SelfScaling(nn.Module):
    def __init__(self, in_planes, ratio):
        super(SelfScaling, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1)
        self.in_planes=in_planes
    def forward(self, img):
        f = self.fc2(self.relu1(self.fc1(self.avg_pool(img))))#B 2C 1 1
        return img.mul(F.sigmoid(f))+img

class SelfGating(nn.Module):
    def __init__(self, in_planes):
        super(SelfGating, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_planes, 1, kernel_size=1), nn.Sigmoid())
    def forward(self, img):
        gate = self.conv(img)
        return img.mul(gate)+img


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(4, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, flow):
        x = torch.cat([torch.mean(img, dim=1, keepdim=True),
                       torch.max(img, dim=1, keepdim=True)[0],
                       torch.mean(flow, dim=1, keepdim=True),
                       torch.max(flow, dim=1, keepdim=True)[0]], dim=1)
        x = self.sigmoid(self.conv1(x))
        return img.mul(x)+flow.mul(1-x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes*2, in_planes // ratio, 1)
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1)
        self.in_planes=in_planes
    def forward(self, img, flow):
        f = self.fc2(self.relu1(self.fc1(self.avg_pool(torch.cat([img, flow], 1)))))#B 2C 1 1
        f=F.sigmoid(f)
        return img.mul(f), flow.mul(1-f)


class STAFM(nn.Module):
    def __init__(self, inplanes, ratio=2):
        super(STAFM, self).__init__()
        self.channel = ChannelAttention(inplanes, ratio=ratio)
        self.spatial = SpatialAttention()
    def forward(self, spatial, temporal):
        spatial, temporal = self.channel(spatial, temporal)
        feature = self.spatial(spatial, temporal)
        return feature

class RTrans(nn.Module):
    def __init__(self, d_model, h=4):
        super(RTrans, self).__init__()
        self.d_model = d_model
        self.d_k = d_model//h
        self.h = h
        self.query = nn.Conv2d(self.d_model, self.d_model, 1)
        self.key = nn.Conv2d(self.d_model, self.d_model, 1)
        self.value = nn.Conv2d(self.d_model, self.d_model, 1)
        self.softmax = nn.Softmax(-1)

    def self_attention(self, q, k, v):
        attention = self.softmax(torch.matmul(q, k)/math.sqrt(self.d_k))
        v = torch.matmul(v, attention)
        return v

    def forward(self, q, k, v):
        B, C, H, W = q.size()
        q, k, v = self.query(q).view(B, self.d_model, H*W).view(B, self.h, self.d_k, H*W).transpose(2,3).contiguous(),\
                  self.key(k).view(B, self.d_model, H*W).view(B, self.h, self.d_k, H*W),\
                  self.value(v).view(B, self.d_model, H*W).view(B, self.h, self.d_k, H*W)
        x = self.self_attention(q, k, v).view(B, self.d_model, H*W).view(B, self.d_model, H, W)
        return x

class OldRTrans(nn.Module):
    def __init__(self, d_model, h=2):
        super(OldRTrans, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        self.query = nn.Conv2d(self.d_model, self.d_model, 1)
        self.key = nn.Conv2d(self.d_model, self.d_model, 1)
        self.value = nn.Conv2d(self.d_model, self.d_model, 1)
        self.softmax = nn.Softmax(-1)

    def self_attention(self, q, k, v):
        attention = self.softmax(torch.matmul(q, k) / math.sqrt(self.d_k))
        v = torch.matmul(v, attention)
        return v

    def forward(self, q, k, v):
        B, C, H_org, W_org = q.size()
        q, k, v = self.maxpool(q), self.maxpool(k), self.maxpool(v)
        B, C, H, W = q.size()
        q, k, v = self.query(q).view(B, self.d_model, H*W).view(B, self.h, self.d_k, H*W).transpose(2,3).contiguous(), \
                  self.key(k).view(B, self.d_model, H*W).view(B, self.h, self.d_k, H*W), \
                  self.value(v).view(B, self.d_model, H*W).view(B, self.h, self.d_k, H*W)
        x = self.self_attention(q, k, v).view(B, self.d_model, H, W)
        return F.interpolate(x, (H_org, W_org), mode='bilinear', align_corners=True)


class RTModule(nn.Module):
    def __init__(self, channel, h, ST, redunction=False):
        super(RTModule, self).__init__()
        self.ST=ST
        self.scaling = RScaling(channel, h)
        if redunction:
            self.transformation1 = OldRTrans(channel, h)
        else:
            self.transformation1 = RTrans(channel, h)
        if self.ST:
            if redunction:
                self.transformation2 = OldRTrans(channel, h)
            else:
                self.transformation2 = RTrans(channel, h)
        self.gating = RGating(channel)
    def forward(self, x, y):
        x, y = self.scaling(x, y)
        x = self.transformation1(x, y, y)
        if self.ST:
            y = self.transformation2(y, x, x)
        else:
            y = self.transformation1(y, x, x)
        x, y = self.gating(x, y)
        return x, y


class SelfModule(nn.Module):
    def __init__(self, channel, h, redunction=False):
        super(SelfModule, self).__init__()
        self.scaling = SelfScaling(channel, h)
        if redunction:
            self.transformation = OldRTrans(channel, h)
        else:
            self.transformation = RTrans(channel, h)
        self.gating = SelfGating(channel)

    def forward(self, x):
        x = self.scaling(x)
        x = self.transformation(x,x,x)
        x = self.gating(x)
        return x


