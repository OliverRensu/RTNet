import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

class out_block(nn.Module):
    def __init__(self, infilter):
        super(out_block, self).__init__()
        self.conv1 = nn.Sequential(*[nn.Conv2d(infilter, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)])
        self.conv2 = nn.Conv2d(64, 1, 1)
    def forward(self, x, H, W):
        x= F.interpolate(self.conv1(x), (H, W), mode='bilinear', align_corners=True)
        return self.conv2(x)

class decoder_stage(nn.Module):
    def __init__(self, infilter, midfilter, outfilter):
        super(decoder_stage, self).__init__()
        self.layer = nn.Sequential(*[nn.Conv2d(infilter, midfilter, 3, padding=1, bias=False), nn.BatchNorm2d(midfilter), nn.ReLU(inplace=True),
                                nn.Conv2d(midfilter, midfilter, 3, padding=1, bias=False), nn.BatchNorm2d(midfilter), nn.ReLU(inplace=True),
                                nn.Conv2d(midfilter, outfilter, 3, padding=1, bias=False), nn.BatchNorm2d(outfilter),
                                nn.ReLU(inplace=True)])
    def forward(self, x):
        return self.layer(x)

class Spatial(nn.Module):
    def __init__(self):
        super(Spatial, self).__init__()
        # -------------Encoder--------------
        resnet = models.resnet34(pretrained=False)
        # -------------Encoder--------------
        self.inconv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder1 = resnet.layer1  # 256
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 64
        self.encoder4 = resnet.layer4  # 32
        # ------------dilation---------------#
        self.dilation1 = nn.Sequential(nn.Conv2d(512, 512, 3, dilation=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.dilation2 = nn.Sequential(nn.Conv2d(512, 512, 3, dilation=2, padding=2, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.dilation3 = nn.Sequential(nn.Conv2d(512, 512, 3, dilation=4, padding=4, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.dilation4 = nn.Sequential(nn.Conv2d(512, 512, 3, dilation=8, padding=8, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        # ------------decoder----------------#
        self.decoder4 = decoder_stage(512*5, 512, 256)  # 32
        self.decoder3 = decoder_stage(512, 256, 128)  # 128
        self.decoder2 = decoder_stage(256, 128, 64)  # 256
        self.decoder1 = decoder_stage(128, 64, 64)  # 256

        self.out4 = out_block(256)
        self.out3 = out_block(128)
        self.out2 = out_block(64)
        self.out1 = out_block(64)

    def encoder(self, x):
        hx = self.inconv(x)
        h1 = self.encoder1(hx)  # 256
        h2 = self.encoder2(h1)  # 128
        h3 = self.encoder3(h2)  # 64
        h4 = self.encoder4(h3)  # 32
        h5 = torch.cat([h4, self.dilation1(h4), self.dilation2(h4), self.dilation3(h4), self.dilation4(h4)], 1)
        return [h1, h2, h3, h4], h5

    def decoder(self, x, f):
        feature4 = self.decoder4(x)
        B, C, H, W = f[2].size()
        feature3 = self.decoder3(torch.cat((F.interpolate(feature4, (H, W), mode='bilinear', align_corners=True), f[2]), 1))
        B, C, H, W = f[1].size()
        feature2 = self.decoder2(torch.cat((F.interpolate(feature3, (H, W), mode='bilinear', align_corners=True), f[1]), 1))
        B, C, H, W = f[0].size()
        feature1 = self.decoder1(torch.cat((F.interpolate(feature2, (H, W), mode='bilinear', align_corners=True), f[0]), 1))

        out4 = torch.sigmoid(self.out4(feature4, H*4, W*4))
        out3 = torch.sigmoid(self.out3(feature3, H*4, W*4))
        out2 = torch.sigmoid(self.out2(feature2, H*4, W*4))
        out1 = torch.sigmoid(self.out1(feature1, H*4, W*4))

        return [out1, out2, out3, out4]

    def forward(self, x):
        feature, dilation = self.encoder(x)
        return self.decoder(dilation, feature)
