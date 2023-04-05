import torch
import numpy as np
import sys
import random
import shutil
import os
from tqdm import tqdm

## Adopted from https://github.com/clovaai/aasist

AUDIO_FILE_FIELD = 1

def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def seed_worker(worker_id):
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SGDRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """SGD with restarts scheduler"""
    def __init__(self, optimizer, T0, T_mul, eta_min, last_epoch=-1):
        self.Ti = T0
        self.T_mul = T_mul
        self.eta_min = eta_min

        self.last_restart = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        T_cur = self.last_epoch - self.last_restart
        if T_cur >= self.Ti:
            self.last_restart = self.last_epoch
            self.Ti = self.Ti * self.T_mul
            T_cur = 0

        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + np.cos(np.pi * T_cur / self.Ti)) / 2
            for base_lr in self.base_lrs
        ]


def _get_optimizer(model_parameters, optim_config):
    """Defines optimizer according to the given config"""
    optimizer_name = optim_config['optimizer']

    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_parameters,
                                    lr=optim_config['base_lr'],
                                    momentum=optim_config['momentum'],
                                    weight_decay=optim_config['weight_decay'],
                                    nesterov=optim_config['nesterov'])
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_parameters,
                                     lr=optim_config['base_lr'],
                                     betas=optim_config['betas'],
                                     weight_decay=optim_config['weight_decay'],
                                     amsgrad=optim_config['amsgrad'])
    else:
        print('Un-known optimizer', optimizer_name)
        sys.exit()

    return optimizer


def _get_scheduler(optimizer, optim_config):
    """
    Defines learning rate scheduler according to the given config
    """
    if optim_config['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=optim_config['milestones'],
            gamma=optim_config['lr_decay'])

    elif optim_config['scheduler'] == 'sgdr':
        scheduler = SGDRScheduler(optimizer, optim_config['T0'],
                                  optim_config['Tmult'],
                                  optim_config['lr_min'])

    elif optim_config['scheduler'] == 'cosine':
        total_steps = optim_config['epochs'] * optim_config['steps_per_epoch']

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                optim_config['lr_min'] / optim_config['base_lr']))

    else:
        scheduler = None
    return scheduler


def create_optimizer(model_parameters, optim_config):
    """Defines an optimizer and a scheduler"""
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler

def generate_sub_sample(config: dict):
    print("generate sub sample - start")
    sub_sample_folder = config['sub_sample_folder']

    if os.path.exists(sub_sample_folder):
        shutil.rmtree(sub_sample_folder)
    os.makedirs(sub_sample_folder)
    protocols = [config['train_protocol'], config['dev_protocol'], config['eval_protocol']]
    audio_folders = [config['train_audio_folder'], config['dev_audio_folder'], config['eval_audio_folder']]
    fractions = [config['train_fraction'], config['dev_fraction'], config['eval_fraction']]
    sample_names = ['train', 'dev', 'eval']
    for protocol, audio_folder, fraction, name in zip(protocols, audio_folders, fractions, sample_names):
        handle_sample(protocol, audio_folder, fraction, name, sub_sample_folder)
    print("generate sub sample - end")

def handle_sample(protocol: str, audio_folder: str, fraction: float, sample_name: str, sub_sample_folder: str):
    print("handle sample: " + sample_name)
    with open(protocol, "r", encoding="utf-8") as f:
        items = [x.strip() for x in f.readlines()]

    sub_sample = [x for x in items if np.random.random() < fraction]
    new_audio_folder = sub_sample_folder + sample_name + "/"
    os.makedirs(sub_sample_folder + sample_name)
    with open(sub_sample_folder + sample_name + "_protocol.txt", "w", encoding='utf-8') as f:
        for sample in tqdm(sub_sample):
            f.write(sample + "\n")
            f.flush()
            audio_file = sample.split()[AUDIO_FILE_FIELD]
            shutil.copy2(audio_folder + audio_file + ".flac", new_audio_folder)
