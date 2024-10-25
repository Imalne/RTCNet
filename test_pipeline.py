import datetime
import logging
import math
import time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options
import os


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders


    val_loaders = []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            continue
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return val_loaders


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=False)
    opt['root_path'] = root_path

    opt['path']['pretrain_network_g']=opt['path']['test_ckpt_path']

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # copy the yml file to the experiment root
    # print(opt['path']['experiments_root'])
    if not os.path.exists(opt['path']['results_root']):
        os.makedirs(opt['path']['results_root'])
    copy_opt_file(args.opt, opt['path']['results_root'])
    

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    # log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = None

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    val_loaders = result

    # create model
    model = build_model(opt)

    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, 0, tb_logger, True)
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
