import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import RTTestDataset
from model import Interactive
import torch.nn.functional as F
from imageio import imwrite
import random
def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
if __name__ == '__main__':
	setup_seed(1024)
	model_dir = "./saved_model/"
	batch_size_val = 1
	dataset = "../../DataSet/DAVIS/val"
	DAVIS_dataset = RTTestDataset(dataset, 2, 384, int(384 * 1.75))
	DAVIS_dataloader = DataLoader(DAVIS_dataset, batch_size=1, shuffle=False, num_workers=4)
	net = Interactive().cuda()
	model_name = 'model_RX50.pth'
	ckpt = torch.load(model_dir + model_name)['state_dict']
	model_dict = net.state_dict()
	pretrained_dict = {k[7:]: v for k, v in ckpt.items() if k[7:] in model_dict}
	model_dict.update(pretrained_dict)
	net.load_state_dict(model_dict)
	net.eval()
	for data in DAVIS_dataloader:
		img, fw_flow, bw_flow, label_org = data['video'].cuda(), data['fwflow'].cuda(), data['bwflow'].cuda(), data['label_org'].cuda()
		_, _, _, H, W = label_org.size()
		flow = torch.cat((fw_flow, bw_flow), 2)
		with torch.no_grad():
			out, _ = net(img, flow)
		out = F.interpolate(out[0], (H, W), mode='bilinear', align_corners=True)
		out = out[0, 0].cpu().numpy()
		out = (out - np.min(out) + 1e-12) / (np.max(out) - np.min(out) + 1e-12) * 255.
		out = out.astype(np.uint8)
		imwrite("./results/"+data['name'][0], out)