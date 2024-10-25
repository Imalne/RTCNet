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
class MultiScaleModel(BaseModel):
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
        super(MultiScaleModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self.net_d = build_network(opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        load_path_d = self.opt['path'].get('pretrain_network_d', None)
        if load_path_d is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path_d, self.opt['path'].get('strict_load_d', True), param_key)

        if self.is_train:
            self.net_d.train()
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        self.l2_pix = build_loss(train_opt['l2_opt']).to(self.device)
        self.l_grad = build_loss(train_opt['l_grad']).to(self.device)
        self.l_percep = build_loss(train_opt["perceptual_opt"]).to(self.device)
        self.model_to_device(self.l_percep)
        self.l_gan = build_loss(train_opt['gan_opt']).to(self.device)


        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.lr_gt = torch.nn.functional.interpolate(self.lq, scale_factor=4)


    def optimize_parameters(self, current_iter, tb_logger):
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        self.output = self.net_g(self.lq, self.gt)
        hc_hr_recon, hc_hr_recon2, hc_lr_recon, hc_lr_recon2, lc_hr_recon, lc_hr_recon2, lc_lr_recon, lc_lr_recon2, latent_loss_hc_hr, latent_loss_hc_lr, latent_loss_lc_hr, latent_loss_lc_lr, id_hc_hr, id_hc_lr, id_lc_hr, id_lc_lr, dist_hc_hr, dist_hc_lr, dist_lc_hr, dist_lc_lr, enc_constraint  = self.output

        if current_iter % self.opt['logger']['print_freq'] == 0:
            tb_logger.add_scalar("dist/dist_hc_hr", torch.mean(dist_hc_hr[0]), current_iter)
            tb_logger.add_scalar("dist/dist_lc_hr", torch.mean(dist_lc_hr[0]), current_iter)
            tb_logger.add_scalar("dist/dist_hc_lr", torch.mean(dist_hc_lr[0]), current_iter)
            tb_logger.add_scalar("dist/dist_lc_lr", torch.mean(dist_lc_lr[0]), current_iter)
        if current_iter % self.opt['logger']['visuak_freq'] == 0:
            tb_logger.add_image("train/lr", torch.from_numpy(tensor2img([self.lq]).astype(np.float)/255).permute(2,0,1), current_iter)
            tb_logger.add_image("train/hr", torch.from_numpy(tensor2img([self.gt]).astype(np.float)/255).permute(2,0,1), current_iter)
            tb_logger.add_image("train/hc_hr_recon", torch.from_numpy(tensor2img([hc_hr_recon]).astype(np.float)/255).permute(2,0,1), current_iter)
            tb_logger.add_image("train/hc_lr_recon", torch.from_numpy(tensor2img([hc_lr_recon]).astype(np.float)/255).permute(2,0,1), current_iter)
            tb_logger.add_image("train/lc_hr_recon", torch.from_numpy(tensor2img([lc_hr_recon]).astype(np.float)/255).permute(2,0,1), current_iter)
            tb_logger.add_image("train/lc_lr_recon", torch.from_numpy(tensor2img([lc_lr_recon]).astype(np.float)/255).permute(2,0,1), current_iter)

        l_total = 0
        loss_dict = OrderedDict()
        
        # l2 loss
        l_l2_hc_hr = self.l2_pix(hc_hr_recon, self.gt) + self.l2_pix(hc_hr_recon2, self.gt)
        l_l2_hc_lr = self.l2_pix(hc_lr_recon, self.gt) + self.l2_pix(hc_lr_recon2, self.gt)
        l_l2_lc_hr = self.l2_pix(lc_hr_recon, self.lr_gt) + self.l2_pix(lc_hr_recon2, self.lr_gt)
        l_l2_lc_lr = self.l2_pix(lc_lr_recon, self.lr_gt) + self.l2_pix(lc_lr_recon2, self.lr_gt)
        l_total += l_l2_hc_hr
        l_total += l_l2_lc_hr
        l_total += l_l2_hc_lr
        l_total += l_l2_lc_lr
        loss_dict['l_recon_hc_hr'] = l_l2_hc_hr
        loss_dict['l_recon_hc_lr'] = l_l2_hc_lr
        loss_dict['l_recon_lc_hr'] = l_l2_lc_hr
        loss_dict['l_recon_lc_lr'] = l_l2_lc_lr

        #laten loss
        l_total += latent_loss_hc_hr.mean() * self.opt["train"]["latent_loss_opt"]["loss_weight"]
        l_total += latent_loss_hc_lr.mean() * self.opt["train"]["latent_loss_opt"]["loss_weight"]
        l_total += latent_loss_lc_hr.mean() * self.opt["train"]["latent_loss_opt"]["loss_weight"]
        l_total += latent_loss_lc_lr.mean() * self.opt["train"]["latent_loss_opt"]["loss_weight"]
        loss_dict['l_latent_hc_hr'] = latent_loss_hc_hr
        loss_dict['l_latent_hc_lr'] = latent_loss_hc_lr
        loss_dict['l_latent_lc_hr'] = latent_loss_lc_hr
        loss_dict['l_latent_lc_lr'] = latent_loss_lc_lr

        # enc consistent constraint
        for i in range(len(enc_constraint)):
            l_enc_cons = enc_constraint[i].mean()
            l_total += l_enc_cons * self.opt["train"]["enc_constraint_opt"]["loss_weight"]
            loss_dict[f"enc_constraint_{i}"] = l_enc_cons

        # perceptual loss
        # l_hr_percep, l_hr_style = self.l_percep(hc_hr_recon, self.gt)
        l_lr_percep, l_lr_style = self.l_percep(hc_lr_recon, self.gt)
        l_lr_percep2, l_lr_style2 = self.l_percep(hc_lr_recon2, self.gt)
        if l_lr_percep is not None:
            l_total += l_lr_percep.mean()
            l_total += l_lr_percep2.mean()
            loss_dict['l_lr_percep'] = l_lr_percep.mean()
        if l_lr_style is not None:
            l_total += l_lr_style
            l_total += l_lr_style2
            loss_dict['l_lr_style'] = l_lr_style

        
        # gan loss
        fake_g_pred_sr = self.net_d(hc_lr_recon)
        l_g_gan_sr = self.l_gan(fake_g_pred_sr, True, is_disc=False)
        l_total += l_g_gan_sr
        loss_dict['l_g_gan_sr'] = l_g_gan_sr



        loss_dict["l_total"] = l_total
        #id_acc
        for i in range(len(id_hc_hr)):
            tb_logger.add_scalar(f'acc_{i}/id_lc_lr_hc_lr', torch.sum(id_lc_lr[i] == id_hc_lr[i])/id_lc_lr[i].flatten().size(0), current_iter)
            tb_logger.add_scalar(f'acc_{i}/id_hc_lr_hc_hr', torch.sum(id_hc_lr[i] == id_hc_hr[i])/id_hc_lr[i].flatten().size(0), current_iter)
            tb_logger.add_scalar(f'acc_{i}/id_lc_lr_hc_hr', torch.sum(id_lc_lr[i] == id_hc_hr[i])/id_lc_lr[i].flatten().size(0), current_iter)

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)


        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        

        for p in self.net_d.parameters():
            p.requires_grad = True
        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.l_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(hc_lr_recon.detach())
        l_d_fake = self.l_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        

    def test(self):
        self.h_l,self.w_l = self.lq.shape[-2:]
        self.h,self.w = self.gt.shape[-2:]
        self.padding_h = (256 - self.h % 256)%256
        self.padding_w = (256 - self.w % 256)%256
        self.padding_h_l = (256 - self.h_l % 64)%64
        self.padding_w_l = (256 - self.w_l % 64)%64

        self.lq = torch.nn.functional.pad(self.lq, (self.padding_w_l//2, self.padding_w_l - self.padding_w_l//2, self.padding_h_l//2, self.padding_h_l-self.padding_h_l//2), mode="reflect")
        self.gt = torch.nn.functional.pad(self.gt, (self.padding_w//2, self.padding_w - self.padding_w//2, self.padding_h//2, self.padding_h-self.padding_h//2), mode="reflect")

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq, self.gt)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(lr=self.lq, hr = self.gt)

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


        min_size=[100000,100000]
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
                min_size = [min(save_samples["hr"][-1].shape[0], min_size[0]), min(save_samples["hr"][-1].shape[1], min_size[1])]


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
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    for key in self.metric_results_dict.keys():
                        self.metric_results_dict[key][name] += calculate_metric(metric_datas[key], opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        save_samples['lr'] = [i[:min_size[0]//4,:min_size[1]//4] for i in save_samples['lr']]
        save_samples['hr'] = [i[:min_size[0],:min_size[1]] for i in save_samples['hr']]
        save_samples['sr'] = [i[:min_size[0],:min_size[1]] for i in save_samples['sr']]

        if not save_img and tb_logger is not None:
            tb_logger.add_image("valid/lr", torch.from_numpy(np.vstack(save_samples['lr']).astype(np.float) / 255).permute(2,0,1), current_iter)
            tb_logger.add_image("valid/hr", torch.from_numpy(np.vstack(save_samples['hr']).astype(np.float) / 255).permute(2,0,1), current_iter)
            tb_logger.add_image("valid/sr", torch.from_numpy(np.vstack(save_samples['sr']).astype(np.float) / 255).permute(2,0,1), current_iter)

        if use_pbar:
            pbar.close()


        if with_metrics:
            for setting_name in self.metric_results_dict.keys():
                # update the best metric result
                if setting_name == "sr":
                    for metric in self.metric_results_dict[setting_name].keys():
                        self.metric_results_dict[setting_name][metric] /= (idx + 1)
                        self._update_best_metric_result(dataset_name, metric, self.metric_results_dict[setting_name][metric], current_iter)

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

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for setting_name in self.metric_results_dict:
                for metric, value in self.metric_results_dict[setting_name].items():
                    tb_logger.add_scalar(f'metrics/{dataset_name}/{setting_name}_{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        hc_hr_recon, hc_lr_recon, lc_hr_recon, lc_lr_recon, latent_loss_hc_hr, latent_loss_hc_lr, latent_loss_lc_hr, latent_loss_lc_lr, id_hc_hr, id_hc_lr, id_lc_hr, id_lc_lr, dist_hc_hr, dist_hc_lr, dist_lc_hr, dist_lc_lr, enc_constraint = self.output
        out_dict['lq'] = self.lq.detach().cpu()[:,:,self.padding_h_l//2:self.h_l+self.padding_h_l//2,self.padding_w_l//2:self.w_l+self.padding_w_l//2]
        out_dict['sr'] = hc_lr_recon.detach().cpu()[:,:,self.padding_h//2:self.h+self.padding_h//2,self.padding_w//2:self.w+self.padding_w//2]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:,:,self.padding_h//2:self.h+self.padding_h//2,self.padding_w//2:self.w+self.padding_w//2]
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
            self.save_network(self.net_d, 'net_d', current_iter)
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

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

    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
            if val >= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
        else:
            if val <= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter