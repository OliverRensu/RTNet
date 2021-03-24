import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .spatial import Spatial
from .temporal import Temporal
from .Attention import RTModule, SelfModule, STAFM
class Interactive(nn.Module):
    def __init__(self, spatial_ckpt=None, temporal_ckpt=None):
        super(Interactive, self).__init__()
        self.spatial_net = Spatial()
        self.temporal_net = Temporal()
        if spatial_ckpt is not None:
            self.spatial_net.load_state_dict(torch.load(spatial_ckpt, map_location='cpu')['state_dict'])
            print("Successfully load spatial:{}".format(spatial_ckpt))
        if temporal_ckpt is not None:
            self.temporal_net.load_state_dict(torch.load(temporal_ckpt, map_location='cpu')['state_dict'])
            print("Successfully load temporal:{}".format(temporal_ckpt))
        self.SSTransformer1 = RTModule(64, h=1, ST=False, redunction=True)
        self.SSTransformer2 = RTModule(128, h=2, ST=False)
        self.SSTransformer3 = RTModule(256, h=4, ST=False)
        self.SSTransformer4 = RTModule(512, h=8, ST=False)
        self.STTransformer1 = RTModule(64, h=1, ST=True, redunction=True)
        self.STTransformer2 = RTModule(128, h=2, ST=True)
        self.STTransformer3 = RTModule(256, h=4, ST=True)
        self.STTransformer4 = RTModule(512, h=8, ST=True)
        self.SselfTransformer1 = SelfModule(64, h=1, redunction=True)
        self.SselfTransformer2 = SelfModule(128, h=2)
        self.SselfTransformer3 = SelfModule(256, h=4)
        self.SselfTransformer4 = SelfModule(512, h=8)

        self.TTTransformer1 = RTModule(64, h=1, ST=False, redunction=True)
        self.TTTransformer2 = RTModule(128, h=2, ST=False)
        self.TTTransformer3 = RTModule(256, h=4, ST=False)
        self.TTTransformer4 = RTModule(512, h=8, ST=False)
        self.TselfTransformer1 = SelfModule(64, h=1, redunction=True)
        self.TselfTransformer2 = SelfModule(128, h=2)
        self.TselfTransformer3 = SelfModule(256, h=4)
        self.TselfTransformer4 = SelfModule(512, h=8)
        self.Transformerstafm3 = STAFM(256, 4)
        self.Transformerstafm2 = STAFM(128, 2)
        self.Transformerstafm1 = STAFM(64, 1)


    def spatial_decoder(self, x, spatial_f, temporal_f):
        feature4 = self.spatial_net.decoder4(x)
        B, C, H, W = spatial_f[2].size()
        feature3 = self.spatial_net.decoder3(
            torch.cat((F.interpolate(feature4, (H, W), mode='bilinear', align_corners=True),
                       self.Transformerstafm3(spatial_f[2], temporal_f[2])), 1))
        B, C, H, W = spatial_f[1].size()
        feature2 = self.spatial_net.decoder2(
            torch.cat((F.interpolate(feature3, (H, W), mode='bilinear', align_corners=True),
                       self.Transformerstafm2(spatial_f[1], temporal_f[1])), 1))
        B, C, H, W = spatial_f[0].size()
        feature1 = self.spatial_net.decoder1(
            torch.cat((F.interpolate(feature2, (H, W), mode='bilinear', align_corners=True),
                       self.Transformerstafm1(spatial_f[0], temporal_f[0])), 1))

        out4 = torch.sigmoid(self.spatial_net.out4(feature4, H * 4, W * 4))
        out3 = torch.sigmoid(self.spatial_net.out3(feature3, H * 4, W * 4))
        out2 = torch.sigmoid(self.spatial_net.out2(feature2, H * 4, W * 4))
        out1 = torch.sigmoid(self.spatial_net.out1(feature1, H * 4, W * 4))
        return [out1, out2, out3, out4]

    def temporal_decoder(self, x, f):
        feature4 = self.temporal_net.decoder4(x)
        B, C, H, W = f[2].size()
        feature3 = self.temporal_net.decoder3(torch.cat((F.interpolate(feature4, (H, W), mode='bilinear', align_corners=True), f[2]), 1))
        B, C, H, W = f[1].size()
        feature2 = self.temporal_net.decoder2(torch.cat((F.interpolate(feature3, (H, W), mode='bilinear', align_corners=True), f[1]), 1))
        B, C, H, W = f[0].size()
        feature1 = self.temporal_net.decoder1(torch.cat((F.interpolate(feature2, (H, W), mode='bilinear', align_corners=True), f[0]), 1))

        out4 = torch.sigmoid(self.temporal_net.out4(feature4, H * 4, W * 4))
        out3 = torch.sigmoid(self.temporal_net.out3(feature3, H * 4, W * 4))
        out2 = torch.sigmoid(self.temporal_net.out2(feature2, H * 4, W * 4))
        out1 = torch.sigmoid(self.temporal_net.out1(feature1, H * 4, W * 4))
        return [out1, out2, out3, out4]

    def encoder(self, x, flow):
        B, Seq, C, H, W = x.size()
        spatial0 = self.spatial_net.inconv(x.view(B*Seq, C, H, W))
        temporal0 = self.temporal_net.inconv(flow.view(B*Seq, C*2, H, W))
        _, C, H, W = spatial0.size()

        spatial1 = self.spatial_net.encoder1(spatial0)#64 128
        temporal1 = self.temporal_net.encoder1(temporal0)
        _, C, H, W = spatial1.size()
        spatial1_1, spatial1_2 = spatial1.view(B, Seq, C, H, W).chunk(2, dim=1)
        spatial1_1, spatial1_2 = spatial1_1.squeeze(1), spatial1_2.squeeze(1)
        temporal1_1, temporal1_2 = temporal1.view(B, Seq, C, H , W).chunk(2, dim=1)
        temporal1_1, temporal1_2 = temporal1_1.squeeze(1), temporal1_2.squeeze(1)

        S1_1S, S1_2S = self.SSTransformer1(spatial1_1, spatial1_2)
        S1_1self, S1_2self = self.SselfTransformer1(spatial1_1), self.SselfTransformer1(spatial1_2)
        S1_1T, T1_1S = self.STTransformer1(spatial1_1, temporal1_1)
        S1_2T, T1_2S = self.STTransformer1(spatial1_2, temporal1_2)
        T1_1T, T1_2T = self.TTTransformer1(temporal1_1, temporal1_2)
        T1_1self, T1_2self = self.TselfTransformer1(temporal1_1), self.TselfTransformer1(temporal1_2)
        S1_1 = S1_1S + S1_1self + S1_1T + spatial1_1
        S1_2 = S1_2S + S1_2self + S1_2T + spatial1_2
        T1_1 = T1_1S + T1_1self + T1_1T + temporal1_1
        T1_2 = T1_2S + T1_2self + T1_2T + temporal1_2
        h1_spatial = torch.cat([S1_1, S1_2], 0)
        h1_temporal = torch.cat([T1_1, T1_2], 0)

        spatial2 = self.spatial_net.encoder2(h1_spatial)#128 64
        temporal2 = self.temporal_net.encoder2(h1_temporal)
        _, C, H, W = spatial2.size()
        spatial2_1, spatial2_2 = spatial2.view(B, Seq, C, H, W).chunk(2, dim=1)
        spatial2_1, spatial2_2 = spatial2_1.squeeze(1), spatial2_2.squeeze(1)
        temporal2_1, temporal2_2 = temporal2.view(B, Seq, C, H, W).chunk(2, dim=1)
        temporal2_1, temporal2_2 = temporal2_1.squeeze(1), temporal2_2.squeeze(1)

        S2_1S, S2_2S = self.SSTransformer2(spatial2_1, spatial2_2)
        S2_1self, S2_2self = self.SselfTransformer2(spatial2_1), self.SselfTransformer2(spatial2_2)
        S2_1T, T2_1S = self.STTransformer2(spatial2_1, temporal2_1)
        S2_2T, T2_2S = self.STTransformer2(spatial2_2, temporal2_2)
        T2_1T, T2_2T = self.TTTransformer2(temporal2_1, temporal2_2)
        T2_1self, T2_2self = self.TselfTransformer2(temporal2_1), self.TselfTransformer2(temporal2_2)
        S2_1 = S2_1S + S2_1self + S2_1T + spatial2_1
        S2_2 = S2_2S + S2_2self + S2_2T + spatial2_2
        T2_1 = T2_1S + T2_1self + T2_1T + temporal2_1
        T2_2 = T2_2S + T2_2self + T2_2T + temporal2_2
        h2_spatial = torch.cat([S2_1, S2_2], 0)
        h2_temporal = torch.cat([T2_1, T2_2], 0)

        spatial3 = self.spatial_net.encoder3(h2_spatial)#256 32
        temporal3 = self.temporal_net.encoder3(h2_temporal)
        _, C, H, W = spatial3.size()
        spatial3_1, spatial3_2 = spatial3.view(B, Seq, C, H, W).chunk(2, dim=1)
        temporal3_1, temporal3_2 = temporal3.view(B, Seq, C, H, W).chunk(2, dim=1)
        spatial3_1, spatial3_2 = spatial3_1.squeeze(1), spatial3_2.squeeze(1)
        temporal3_1, temporal3_2 = temporal3_1.squeeze(1), temporal3_2.squeeze(1)

        S3_1S, S3_2S = self.SSTransformer3(spatial3_1, spatial3_2)
        S3_1self, S3_2self = self.SselfTransformer3(spatial3_1), self.SselfTransformer3(spatial3_2)
        S3_1T, T3_1S = self.STTransformer3(spatial3_1, temporal3_1)
        S3_2T, T3_2S = self.STTransformer3(spatial3_2, temporal3_2)
        T3_1T, T3_2T = self.TTTransformer3(temporal3_1, temporal3_2)
        T3_1self, T3_2self = self.TselfTransformer3(temporal3_1), self.TselfTransformer3(temporal3_2)
        S3_1 = S3_1S + S3_1self + S3_1T + spatial3_1
        S3_2 = S3_2S + S3_2self + S3_2T + spatial3_2
        T3_1 = T3_1S + T3_1self + T3_1T + temporal3_1
        T3_2 = T3_2S + T3_2self + T3_2T + temporal3_2
        h3_spatial = torch.cat([S3_1, S3_2], 0)
        h3_temporal = torch.cat([T3_1, T3_2], 0)

        spatial4 = self.spatial_net.encoder4(h3_spatial)#512 16
        temporal4 = self.temporal_net.encoder4(h3_temporal)
        _, C, H, W = spatial4.size()
        spatial4_1, spatial4_2 = spatial4.view(B, Seq, C, H, W).chunk(2, dim=1)
        temporal4_1, temporal4_2 = temporal4.view(B, Seq, C, H, W).chunk(2, dim=1)
        spatial4_1, spatial4_2 = spatial4_1.squeeze(1), spatial4_2.squeeze(1)
        temporal4_1, temporal4_2 = temporal4_1.squeeze(1), temporal4_2.squeeze(1)

        S4_1S, S4_2S = self.SSTransformer4(spatial4_1, spatial4_2)
        S4_1self, S4_2self = self.SselfTransformer4(spatial4_1), self.SselfTransformer4(spatial4_2)
        S4_1T, T4_1S = self.STTransformer4(spatial4_1, temporal4_1)
        S4_2T, T4_2S = self.STTransformer4(spatial4_2, temporal4_2)
        T4_1T, T4_2T = self.TTTransformer4(temporal4_1, temporal4_2)
        T4_1self, T4_2self = self.TselfTransformer4(temporal4_1), self.TselfTransformer4(temporal4_2)
        S4_1 = S4_1S + S4_1self + S4_1T + spatial4_1
        S4_2 = S4_2S + S4_2self + S4_2T + spatial4_2
        T4_1 = T4_1S + T4_1self + T4_1T + temporal4_1
        T4_2 = T4_2S + T4_2self + T4_2T + temporal4_2
        h4_spatial = torch.cat([S4_1, S4_2], 0)
        h4_temporal = torch.cat([T4_1, T4_2], 0)

        spatial5 = torch.cat((h4_spatial,
            self.spatial_net.dilation1(h4_spatial),
            self.spatial_net.dilation2(h4_spatial),
            self.spatial_net.dilation3(h4_spatial),
            self.spatial_net.dilation4(h4_spatial)), 1)

        temporal5 = torch.cat((h4_temporal,
            self.temporal_net.dilation1(h4_temporal),
            self.temporal_net.dilation2(h4_temporal),
            self.temporal_net.dilation3(h4_temporal),
            self.temporal_net.dilation4(h4_temporal)), 1)
        return spatial5, [h1_spatial, h2_spatial, h3_spatial], temporal5, [h1_temporal, h2_temporal, h3_temporal]

    def forward(self, img, flow, train_flow=True):
        if train_flow:
            spatial, spatial_feature, temporal, temporal_feature = self.encoder(img, flow)
            return self.spatial_decoder(spatial, spatial_feature, temporal_feature), self.temporal_decoder(temporal, temporal_feature)
        else:
            spatial, spatial_feature, temporal, temporal_feature = self.encoder(img, flow)
            return self.spatial_decoder(spatial, spatial_feature, temporal_feature)