from distutils.command.build import build
import numpy as np

import torch, tqdm
from os import path as osp
from collections import OrderedDict

from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel, BaseModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.metrics import calculate_metric


@MODEL_REGISTRY.register()
class TestModel(BaseModel):
    """Example model based on the SRModel class.

    In this example model, we want to implement a new model that trains with both L1 and L2 loss.

    New defined functions:
        init_training_settings(self)
        feed_data(self, data)
        optimize_parameters(self, current_iter)

    Inherited functions:
        __init__(self, opt)
        setup_optimizers(self)
        test(self)
        dist_validation(self, dataloader, current_iter, tb_logger, save_img)
        nondist_validation(self, dataloader, current_iter, tb_logger, save_img)
        _log_validation_metric_values(self, current_iter, dataset_name, tb_logger)
        get_current_visuals(self)
        save(self, epoch, current_iter)
    """
    def __init__(self, opt):
        super(TestModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.lr_gt = torch.nn.functional.interpolate(self.lq, scale_factor=4)

    def test(self):
        self.h_l,self.w_l = self.lq.shape[-2:]
        self.h,self.w = self.gt.shape[-2:]
        self.padding_h = (256 - self.h % 256)%256
        self.padding_w = (256 - self.w % 256)%256
        self.padding_h_l = (64 - self.h_l % 64)%64
        self.padding_w_l = (64 - self.w_l % 64)%64

        self.lq = torch.nn.functional.pad(self.lq, (self.padding_w_l//2, self.padding_w_l - self.padding_w_l//2, self.padding_h_l//2, self.padding_h_l-self.padding_h_l//2), mode="reflect")
        self.gt = torch.nn.functional.pad(self.gt, (self.padding_w//2, self.padding_w - self.padding_w//2, self.padding_h//2, self.padding_h-self.padding_h//2), mode="reflect")

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq, hr = torch.zeros_like(self.gt))
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(lr=self.lq, hr = torch.zeros_like(self.gt))

        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        metrics_indexs = ["sr"]

        if with_metrics:
            if not hasattr(self, 'metric_results_dict'):  # only execute in the first run
                self.metric_results_dict={}
                for key in metrics_indexs:
                    self.metric_results_dict[key]={metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results

        self.metric_results_dict = {}
        for key in metrics_indexs:
            self.metric_results_dict[key]={metric: 0 for metric in self.opt['val']['metrics'].keys()}

        metric_datas = dict()
        if use_pbar:
            pbar = tqdm.tqdm(total=len(dataloader), unit='image')

        sample_num = 25
        save_samples=({
            "sr": [],
            "lr": [],
            "hr": [],
        })


        for idx, val_data in tqdm.tqdm(enumerate(dataloader)):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()

            metric_datas["sr"] = {'img': tensor2img([visuals['sr']]), "img2": tensor2img([visuals['gt']])}

            if len(save_samples["sr"]) < sample_num:
                save_samples["sr"].append(tensor2img([visuals['sr']]))
                save_samples["lr"].append(tensor2img([self.lq]))
                save_samples["hr"].append(tensor2img([visuals['gt']]))


            # tentative for out of GPU memory
            del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                            f'{img_name}_{self.opt["name"]}.png')
                    
                imwrite(metric_datas["sr"]["img"], save_img_path)


            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    for key in self.metric_results_dict.keys():
                        self.metric_results_dict[key][name] += calculate_metric(metric_datas[key], opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()


        if with_metrics:
            for setting_name in self.metric_results_dict.keys():
                    for metric in self.metric_results_dict[setting_name].keys():
                        self.metric_results_dict[setting_name][metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for setting_name in self.metric_results_dict:
            for metric, value in self.metric_results_dict[setting_name].items():
                log_str += f'\t # {setting_name}: \t {metric}: {value:.4f}'
                if hasattr(self, 'best_metric_results') and setting_name == "sr":
                    log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                        f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
                log_str += '\n'
        print(log_str)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        hc_hr_recon, hc_lr_recon, lc_hr_recon, lc_lr_recon, latent_loss_hc_hr, latent_loss_hc_lr, latent_loss_lc_hr, latent_loss_lc_lr, id_hc_hr, id_hc_lr, id_lc_hr, id_lc_lr, dist_hc_hr, dist_hc_lr, dist_lc_hr, dist_lc_lr, enc_constraint = self.output
        out_dict['lq'] = self.lq.detach().cpu()[:,:,self.padding_h_l//2:self.h_l+self.padding_h_l//2,self.padding_w_l//2:self.w_l+self.padding_w_l//2]
        out_dict['sr'] = hc_lr_recon.detach().cpu()[:,:,self.padding_h//2:self.h+self.padding_h//2,self.padding_w//2:self.w+self.padding_w//2]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:,:,self.padding_h//2:self.h+self.padding_h//2,self.padding_w//2:self.w+self.padding_w//2]
        return out_dict

    def _initialize_best_metric_results(self, dataset_name):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric, content in self.opt['val']['metrics'].items():
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record