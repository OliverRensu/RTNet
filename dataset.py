import torch
from imageio import imread
import numpy as np
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
def img_normalize(image):
    if len(image.shape)==2:
        channel = (image[:, :, np.newaxis] - 0.485) / 0.229
        image = np.concatenate([channel,channel,channel], axis=2)
    else:
        image = (image-np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3)))\
                /np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    return image


class RTDataset(Dataset):
    def __init__(self, pathes, T, size):
        self.fwflow_list=[]
        self.bwflow_list=[]
        self.img_list=[]
        self.label_list=[]
        self.size=size
        self.T = T
        for path in pathes:
            file = sorted(os.listdir(os.path.join(path, "img_flip")))
            for i in file:
                self.img_list.append(os.path.join(path, "img_flip", i))
                self.label_list.append(os.path.join(path, "label_flip", i))
                self.fwflow_list.append(os.path.join(path, "flow_img_flip", "fw_"+i))
                self.bwflow_list.append(os.path.join(path, "flow_img_flip", "bw_"+i))
        self.dataset_len = len(self.img_list)
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, item):
        frame=[item]
        scope = 40
        other = np.random.randint(-scope, scope)
        while item + other >=self.dataset_len or item+other<0 or other==0:
            other = np.random.randint(-scope, scope)
        name1 = self.img_list[item]
        name2 = self.img_list[item+other]
        while name1.split('/')[-1].split("_")[0] != name2.split('/')[-1].split("_")[0] or name1.split('/')[-1].split("_")[-1] != name2.split('/')[-1].split("_")[-1]:
            other = np.random.randint(-scope, scope)
            while item + other >= self.dataset_len or item + other < 0 or other == 0:
                other = np.random.randint(-scope, scope)
            name2 = self.img_list[item + other]
        frame.append(item+other)
        videos, labels, fwflows, bwflows=[],[],[],[]
        for i in frame:
            video = imread(self.img_list[i])
            fw = imread(self.fwflow_list[i])
            bw = imread(self.bwflow_list[i])
            label = imread(self.label_list[i])
            if len(label.shape)==3:
                label=label[:,:,0]
            label=label[:, :, np.newaxis]
            videos.append(img_normalize(video.astype(np.float32)/255.))
            labels.append(label.astype(np.float32)/255.)
            fwflows.append(img_normalize(fw.astype(np.float32)/255.))
            bwflows.append(img_normalize(bw.astype(np.float32)/ 255.))
        video = torch.from_numpy(np.stack(videos, 0)).permute(0,3,1,2)
        label = torch.from_numpy(np.stack(labels, 0)).permute(0,3,1,2)
        fwflow = torch.from_numpy(np.stack(fwflows, 0)).permute(0,3,1,2)
        bwflow = torch.from_numpy(np.stack(bwflows, 0)).permute(0,3,1,2)
        if self.size is None:
            return {'video': video,
                    'label': label,
                    'fwflow': fwflow,
                    'bwflow': bwflow}
        else:
            return {'video': F.interpolate(video, (self.size, int(self.size*1.75)), mode='bilinear', align_corners=True),
                    'label': F.interpolate(label, (self.size, int(self.size*1.75)), mode='bilinear', align_corners=True),
                    'fwflow': F.interpolate(fwflow, (self.size, int(self.size*1.75)), mode='bilinear', align_corners=True),
                    'bwflow': F.interpolate(bwflow, (self.size, int(self.size*1.75)), mode='bilinear', align_corners=True),}


class RTTestDataset(Dataset):
    def __init__(self, pathes, T, H, W):
        self.fwflow_list = []
        self.bwflow_list = []
        self.img_list = []
        self.label_list = []
        self.T = T
        self.H, self.W = H, W
        for path in pathes:
            file = sorted(os.listdir(os.path.join(path, "img")))
            for i in file:
                self.img_list.append(os.path.join(path, "img", i))
                self.label_list.append(os.path.join(path, "label", i[:-3] + "png"))
                self.fwflow_list.append(os.path.join(path, "flow_img", "fw_" + i[:-3] + "png"))
                self.bwflow_list.append(os.path.join(path, "flow_img", "bw_" + i[:-3] + "png"))
        self.dataset_len = len(self.img_list)
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        frame = [item]
        scope = 10
        other = np.random.randint(-scope, scope)
        while item + other >= self.dataset_len or item + other < 0 or other == 0:
            other = np.random.randint(-scope, scope)
        name1 = self.img_list[item]
        name2 = self.img_list[item + other]
        while name1.split('/')[-1].split("_")[0] != name2.split('/')[-1].split("_")[0]:
            other = np.random.randint(-scope, scope)
            while item + other >= self.dataset_len or item + other < 0 or other == 0:
                other = np.random.randint(-scope, scope)
            name2 = self.img_list[item + other]
        frame.append(item + other)
        videos, labels, fwflows, bwflows = [], [], [], []
        for i in frame:
            video = imread(self.img_list[i])
            fw = imread(self.fwflow_list[i])
            bw = imread(self.bwflow_list[i])
            label = imread(self.label_list[i])
            if len(label.shape) == 3:
                label = label[:, :, 0]
            label = label[:, :, np.newaxis]
            videos.append(img_normalize(video.astype(np.float32) / 255.))
            labels.append(label.astype(np.float32) / 255.)
            fwflows.append(img_normalize(fw.astype(np.float32) / 255.))
            bwflows.append(img_normalize(bw.astype(np.float32) / 255.))
            H, W = labels[0].shape[0], labels[0].shape[1]
        return {'video': F.interpolate(torch.from_numpy(np.stack(videos, 0)).permute(0, 3, 1, 2), (self.H, self.W), mode='bilinear', align_corners=True),
                'fwflow': F.interpolate(torch.from_numpy(np.stack(fwflows, 0)).permute(0, 3, 1, 2), (self.H, self.W), mode='bilinear', align_corners=True),
                'bwflow': F.interpolate(torch.from_numpy(np.stack(bwflows, 0)).permute(0, 3, 1, 2), (self.H, self.W), mode='bilinear', align_corners=True),
                "label_org":torch.from_numpy(np.stack([labels[0]], 0)).permute(0, 3, 1, 2),
                "H":H, "W":W, 'name': self.img_list[item].split("/")[-1]}

