import os.path
import sys
import torch
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
import pyiqa
from os import path as osp
from basicsr.utils.options import parse_options
import glob, cv2
import tqdm
from argparse import ArgumentParser 
import yaml


root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
# opt, args = parse_options(root_path, is_train=False)
parser = ArgumentParser()
parser.add_argument('--gt_dir', '-g', type=str)
parser.add_argument('--lq_dir', '-l', type=str)
parser.add_argument('--option','-opt', type=str)

args = parser.parse_args()

hr_img_dir = args.gt_dir
sr_img_dir = args.lq_dir
with open(args.option, 'r', encoding='utf-8') as f:
    opt = yaml.load(f.read(), Loader=yaml.FullLoader)


sr_img_paths = sorted(glob.glob(os.path.join(sr_img_dir, "*.png")))
hr_img_paths = sorted(glob.glob(os.path.join(hr_img_dir, "*.png")))

with_metrics = opt['val'].get('metrics') is not None
device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
metric_results = {
                metric: 0
                for metric in opt['val']['metrics'].keys()
            }

if opt['val'].get('metrics') is not None:
    metric_funcs = {}
    for _, nopt in opt['val']['metrics'].items():
        mopt = nopt.copy()
        name = mopt.pop('type', None)
        mopt.pop('better', None)
        metric_funcs[name] = pyiqa.create_metric(name, device=device, **mopt)

print(len(sr_img_paths))
for idx, (sr_path, hr_path) in tqdm.tqdm(enumerate(zip(sr_img_paths, hr_img_paths))):
    #print(sr_path)
    #print(hr_path)
    sr_img = cv2.imread(sr_path)
    gt = cv2.imread(hr_path)
    metric_data = [img2tensor(sr_img).unsqueeze(0)/255, img2tensor(gt).unsqueeze(0)/255]

    if with_metrics:
        # calculate metrics
        for name, opt_ in opt['val']['metrics'].items():
            tmp_result = metric_funcs[name](*metric_data)
            metric_results[name] += tmp_result.item()

if with_metrics:
    # calculate average metric
    for metric in metric_results.keys():
        metric_results[metric] /= (idx + 1)

# print(metric_results)
metrics = ("".join([k+' | ' for k,v in metric_results.items()]))[:-1]+":"
values = ("".join(["{:.4f}".format(v) + ' | ' if v < 1 else "{:.2f}".format(v) +' | ' for k,v in metric_results.items()]))[:-2]
print(metrics, values)
