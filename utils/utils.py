import os
import random

import torch
import numpy as np

import yaml


def get_config(filename: str):
    with open(filename, 'r', encoding='utf8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def initialize_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def initialize_log_directory():
    if not os.path.exists(os.path.join(os.getcwd(), 'logs')):
        os.mkdir(os.path.join(os.getcwd(), 'logs'))
    if not os.path.exists(os.path.join(os.getcwd(), 'logs/ALL')):
        os.mkdir(os.path.join(os.getcwd(), 'logs/ALL'))
    if not os.path.exists(os.path.join(os.getcwd(), 'logs/DIAG')):
        os.mkdir(os.path.join(os.getcwd(), 'logs/DIAG'))
    if not os.path.exists(os.path.join(os.getcwd(), 'logs/SUB')):
        os.mkdir(os.path.join(os.getcwd(), 'logs/SUB'))
    if not os.path.exists(os.path.join(os.getcwd(), 'logs/SUPER')):
        os.mkdir(os.path.join(os.getcwd(), 'logs/SUPER'))
    if not os.path.exists(os.path.join(os.getcwd(), 'logs/FORM')):
        os.mkdir(os.path.join(os.getcwd(), 'logs/FORM'))
    if not os.path.exists(os.path.join(os.getcwd(), 'logs/RHYTHM')):
        os.mkdir(os.path.join(os.getcwd(), 'logs/RHYTHM'))
    if not os.path.exists(os.path.join(os.getcwd(), 'logs/CPSC')):
        os.mkdir(os.path.join(os.getcwd(), 'logs/CPSC'))


def init_dir(save_model_path):
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    if not os.path.exists(os.path.join(save_model_path, 'checkpoints')):
        os.mkdir(os.path.join(save_model_path, 'checkpoints'))


def initialization(seed=0):
    initialize_seed(seed)
    initialize_log_directory()


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_resample_loss_by_type(exp_type):
    class_freq = []
    train_num = 0
    exp_type = exp_type.upper()
    if exp_type == 'SUPER':
        class_freq = [8093, 4054, 2494, 4933, 4704]
        train_num = 17084
    elif exp_type == 'SUB':
        class_freq = [2466, 428, 432, 62, 2618, 894, 756, 318, 1019, 630, 1437, 341,
                      161, 1708, 7596, 615, 13, 79, 102, 24, 1792, 64, 658]
        train_num = 17084
    elif exp_type == 'RHYTHM':
        class_freq = [13404, 1211, 661, 618, 509, 237, 128, 66, 59, 21, 19, 16]
        train_num = 16831
    elif exp_type == 'DIAG':
        class_freq = [1461, 615, 145, 94, 7596, 2143, 1887, 1708, 1298, 1019, 894,
                      634, 630, 527, 432, 428, 383, 341, 282, 232, 175, 171, 161,
                      143, 141, 135, 116, 113, 102, 84, 79, 77, 64, 62, 41, 35,
                      26, 24, 14, 13, 13, 12, 11, 12]
        train_num = 17084
    elif exp_type == 'FORM':
        class_freq = [1461, 615, 145, 94, 2683, 915, 807, 701, 438, 350, 340, 318, 272, 235, 145, 49, 28, 22, 8]
        train_num = 7197
    elif exp_type == 'ALL':
        class_freq = [1461, 615, 145, 94, 7596, 2143, 1887, 1708, 1298, 1019, 894,
                      634, 630, 527, 432, 428, 383, 341, 282, 232, 175, 171, 161,
                      143, 141, 135, 116, 113, 102, 84, 79, 77, 64, 62, 41, 35,
                      26, 24, 14, 13, 13, 12, 11, 12, 2683, 915, 807, 701, 438,
                      350, 340, 318, 272, 235, 145, 49, 28, 22, 8, 13404, 1211,
                      661, 618, 509, 237, 128, 66, 59, 21, 19, 16]
        train_num = 17418
    elif exp_type == 'CPSC':
        class_freq = [742, 952, 575, 178, 1502, 492, 561, 688, 185]
        train_num = 5501

    return class_freq, train_num
