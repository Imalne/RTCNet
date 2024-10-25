import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pyiqa
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.utils import img2tensor

lpips_metric = pyiqa.create_metric("lpips")

@METRIC_REGISTRY.register()
def pyiqa_psnr(img, img2, crop_border, test_y_channel=False, bgr2rgb=True, **kwargs):
    if len(img.shape)<3:
        img = np.expand_dims(img, axis=-1)
        img2 = np.expand_dims(img2, axis=-1)
    psnr_metric = pyiqa.create_metric("psnr", crop_border=crop_border, test_y_channel=test_y_channel)
    return psnr_metric(*[img2tensor(img, bgr2rgb=bgr2rgb).unsqueeze(0)/255, img2tensor(img2, bgr2rgb=bgr2rgb).unsqueeze(0)/255]).item()

@METRIC_REGISTRY.register()
def pyiqa_ssim(img, img2, crop_border, test_y_channel=False, bgr2rgb=True, **kwargs):
    if len(img.shape)<3:
        img = np.expand_dims(img, axis=-1)
        img2 = np.expand_dims(img2, axis=-1)
    ssim_metric = pyiqa.create_metric("ssim", crop_border=crop_border, test_y_channel=test_y_channel)
    return ssim_metric(*[img2tensor(img, bgr2rgb=bgr2rgb).unsqueeze(0)/255, img2tensor(img2, bgr2rgb=bgr2rgb).unsqueeze(0)/255]).item()

@METRIC_REGISTRY.register()
def pyiqa_lpips(img, img2, bgr2rgb=True, **kwargs):
    if len(img.shape)<3:
        img = np.expand_dims(img, axis=-1)
        img2 = np.expand_dims(img2, axis=-1)
    return lpips_metric(*[img2tensor(img, bgr2rgb=bgr2rgb).unsqueeze(0)/255, img2tensor(img2, bgr2rgb=bgr2rgb).unsqueeze(0)/255]).item()


