import os
import argparse

import numpy as np
import torch
import tqdm

from dataset import get_loaders
from ecg_models.ITMN import ITMN
from utils.metrics import calculate_metrics
from utils.utils import *

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


def test(params, ckpt_path):
    # get config
    model_config = params['model']
    hyperparameters = params['hyperparameters']
    exp_type = params['exp_type']

    # get device
    if params['aux']['device'] == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device:', torch.cuda.get_device_name(device) if device != 'cpu' else 'cpu')
    else:
        device = 'cpu'
        print('Device: cpu')

    # get data
    _, _, test_loader, num_class, label_list = get_loaders(params['data'], exp_type, hyperparameters['batch_size'])

    # load model
    model = ITMN(n_classes=num_class, **model_config).to(device)
    print('Number of model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model_results = []
    targets = []
    test_iterator = tqdm.tqdm(test_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i, samples in enumerate(test_iterator):
        # Get data
        waveform, labels = samples['waveform'], samples['label']

        # Forward pass
        outputs = model(waveform.to(device))
        p_outputs = torch.sigmoid(outputs)

        # calculate metrics
        model_results.extend(p_outputs.cpu().detach().numpy())
        targets.extend(labels.cpu().numpy())

    metrics = calculate_metrics(np.array(targets), np.array(model_results))
    print('AUC: {:.4f} | TPR: {:.4f}'.format(metrics['AUC'], metrics['TPR']))


if __name__ == '__main__':
    # define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', type=str, default='super', help='Experiment type')
    args = parser.parse_args()
    assert args.exp_type in ['super', 'sub', 'rhythm', 'all', 'diag', 'form', 'cpsc']

    # get configuration
    config = get_config('config.yaml')
    config['exp_type'] = args.exp_type.lower()

    # test
    ckpt_path = config['test_ckpt_path']
    test(config, ckpt_path)
