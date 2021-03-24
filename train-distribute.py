import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import glob
from dataset import RTDataset
from model_R34 import Interactive
import os
import train_loss
import torch.nn.functional as F
import numpy as np
from Measurement import SMeasure
from imageio import imwrite
import argparse
import torch.distributed as dist
import random
def my_collate_fn(batch):
    size = [256, 320, 384]
    H = size[np.random.randint(0, 3)]
    W = int(1.75*H)
    img = []
    label = []
    fw_flow= []
    bw_flow=[]
    for item in batch:
        img.append(F.interpolate(item['video'], (H, W), mode='bilinear', align_corners=True))
        label.append(F.interpolate(item['label'], (H, W), mode='bilinear', align_corners=True))
        bw_flow.append(F.interpolate(item['bwflow'], (H, W), mode='bilinear', align_corners=True))
        fw_flow.append(F.interpolate(item['fwflow'], (H, W), mode='bilinear', align_corners=True))
    return {'video': torch.stack(img, 0), 'label': torch.stack(label, 0),
            "bwflow": torch.stack(bw_flow, 0), "fwflow": torch.stack(fw_flow, 0)}

def adjust_learning_rate(optimizer, decay_count, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(1e-5, 5e-4 * pow(decay_rate, decay_count))
        print(param_group['lr'])

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    set_seed(1024)
    lr = 1e-3
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()
    print("local_rank", args.local_rank)
    world_size = int(os.environ['WORLD_SIZE'])
    print("world size", world_size)
    dist.init_process_group(backend='nccl')
    # ------- 2. set the directory of training dataset --------
    model_dir = "./saved_model/"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    epoch_num = 100
    batch_size_train = 1

    # Training Data
    dataset = RTDataset(["../../DataSet/DAVIS/train"], 2, None)
    datasampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=args.local_rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, sampler=datasampler, num_workers=8, collate_fn=my_collate_fn)
    print("Training Set, DataSet Size:{}, DataLoader Size:{}".format(len(dataset), len(dataloader)))

    # ------- 3. define model --------
    # define the net
    spatial_ckpt = './models/spatial_RX50.pth'
    temporal_ckpt = './models/temporal_RX50.pth'
    torch.cuda.set_device(args.local_rank)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Interactive(spatial_ckpt, temporal_ckpt))
    net = torch.nn.parallel.DistributedDataParallel(net.cuda(args.local_rank), device_ids=[args.local_rank], find_unused_parameters=True)
    pretrained_net=[]
    transformer=[]
    for name, param in net.named_parameters():
        if "Transformer" in name:
                transformer.append(param)
        elif "spatial_net" in name or "temporal_net" in name:
            param.requires_grad = False
            pretrained_net.append(param)
    param_group = [{"params": transformer, 'lr': lr},
                   {"params": pretrained_net, "lr": 0}]
    optimizer = optim.SGD(param_group, lr=lr, momentum=0.9, weight_decay=0.0005)
    re_load = False
    model_name = ""
    if re_load:
        model_CKPT = torch.load(model_dir + model_name, map_location='cpu')
        optimizer.load_state_dict(model_CKPT['optimizer'])
        net.load_state_dict(model_CKPT['state_dict'])
        print("Successfully load: {}".format(model_name))

    optimizer.zero_grad()
    tag=True
    for epoch in range(1, epoch_num):
        running_loss = 0.0
        running_spatial_loss = 0.0
        running_temporal_loss = 0.0
        ite_num_per = 0
        iter_num = 0
        datasampler.set_epoch(epoch)
        net.train()
        i=0
        if tag and epoch>4:
            lr = 5e-4
            for name, param in net.named_parameters():
                param.requires_grad=True
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            tagx = False
        if epoch>15:
            adjust_learning_rate(optimizer, (epoch-20))
        for data in dataloader:
            ite_num_per = ite_num_per + 1
            i+=1
            iter_num = iter_num + 1
            img, fw_flow, bw_flow, label = data['video'].cuda(args.local_rank), \
                                           data['fwflow'].cuda(args.local_rank),\
                                           data['bwflow'].cuda(args.local_rank),\
                                           data['label'].cuda(args.local_rank)
            B, Seq, C, H, W = img.size()
            spatial_out, temporal_out = net(img, torch.cat((fw_flow, bw_flow), 2))
            spatial_loss0, spatial_loss = train_loss.muti_bce_loss_fusion(spatial_out, label.view(B * Seq, 1, H, W))
            temporal_loss0, temporal_loss = train_loss.muti_bce_loss_fusion(temporal_out, label.view(B * Seq, 1, H, W))
            loss = spatial_loss + temporal_loss
            running_loss += loss.item()  # total loss
            running_spatial_loss += spatial_loss0.item()
            running_temporal_loss += temporal_loss0.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if iter_num%200==0:
                print(
                    "[epoch: {}/{}, iter: {}/{}, iter: {}] train loss: {:.5f}, spatial: {:.5f}, temporal:{:5f}".format(
                        epoch, epoch_num, i, len(dataloader), iter_num,
                        running_loss / ite_num_per, running_spatial_loss / ite_num_per, running_temporal_loss / ite_num_per))
        if args.local_rank==0:
            torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
                       model_dir + "epoch_{}_loss_{:.5f}_spatial_{:.5f}_temporal_{:.5f}.pth".format(
                           epoch, running_loss / ite_num_per, running_spatial_loss / ite_num_per, running_temporal_loss / ite_num_per))