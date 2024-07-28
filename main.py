import os
import time
from datetime import datetime
import argparse
import pickle

import numpy as np
import torch
import tqdm

from dataset import get_loaders
from ecg_models.ITMN import ITMN
from losses.resample_loss import ResampleLoss
from losses.cb_loss import CBLoss
from utils.metrics import calculate_metrics
from utils.utils import *

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


def get_loss(use_loss='standard', exp_type='SUPER'):
    if use_loss == 'DB':
        print('Use Resample Loss')
        class_freq, train_num = get_resample_loss_by_type(exp_type)
        print(class_freq)
        print(train_num)
        loss_fn = ResampleLoss(
            use_sigmoid=True,
            reweight_func="rebalance",
            loss_weight=1.0,
            focal=dict(focal=True, alpha=0.5, gamma=2),
            logit_reg=dict(init_bias=0.05, neg_scale=2.0),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.05),
            class_freq=class_freq,
            train_num=train_num,
        )
    elif use_loss == 'FOCAL':
        print('Use Focal Loss')
        class_freq, train_num = get_resample_loss_by_type(exp_type)
        print(class_freq)
        print(train_num)
        loss_fn = CBLoss(loss_type='focal_loss',
                         samples_per_class=None,
                         class_balanced=False)
    elif use_loss == 'CB':
        print('Use CB Focal Loss')
        class_freq, train_num = get_resample_loss_by_type(exp_type)
        print(class_freq)
        print(train_num)
        loss_fn = CBLoss(loss_type='focal_loss',
                         samples_per_class=class_freq,
                         class_balanced=True)
    elif use_loss == 'WBCE':
        print('Use Weighted BCE Loss')
        class_freq, train_num = get_resample_loss_by_type(exp_type)
        print(class_freq)
        print(train_num)
        loss_fn = CBLoss(loss_type='binary_cross_entropy',
                         samples_per_class=class_freq,
                         class_balanced=True)
    else:
        print('Use Standard BCE Loss')
        loss_fn = torch.nn.BCEWithLogitsLoss()

    return loss_fn


def get_optimizer(model, params, opt='ADAMW'):
    opt = opt.upper()
    if opt == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    elif opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], betas=(0.9, 0.999), weight_decay=1e-4)

    return optimizer


def train_one_epoch(data_loader, model, criterion, optimizer, device='cpu'):
    training_loss = 0
    training_acc = 0
    data_iterator = tqdm.tqdm(data_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    model.train()
    for i, samples in enumerate(data_iterator):
        # Get data
        waveform, labels = samples['waveform'], samples['label']

        # clear gradient
        optimizer.zero_grad()

        # Forward pass
        outputs = model(waveform.to(device))
        p_outputs = torch.sigmoid(outputs)

        # calculate loss
        loss = criterion(outputs.to(device), labels.to(device))
        acc = ((p_outputs >= 0.5).cpu().int() == labels).sum() / (labels.shape[0] * labels.shape[1])

        # Backprop and optimize
        loss.backward()
        optimizer.step()

        # update running metrics
        training_loss += loss.item()
        training_acc += acc

    training_loss /= len(data_loader)
    training_acc /= len(data_loader)

    return training_loss, training_acc


def evaluate_one_epoch(data_loader, model, criterion, device='cpu'):
    validation_loss = 0
    validation_acc = 0
    model_results = []
    targets = []
    data_iterator = tqdm.tqdm(data_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    model.eval()
    with torch.no_grad():
        for i, samples in enumerate(data_iterator):
            # Get data
            waveform, labels = samples['waveform'], samples['label']

            # Forward pass
            outputs = model(waveform.to(device))
            p_outputs = torch.sigmoid(outputs)

            # calculate loss
            loss = criterion(outputs.to(device), labels.to(device))
            acc = ((p_outputs >= 0.5).cpu().int() == labels).sum() / (labels.shape[0] * labels.shape[1])

            # update running metrics
            validation_loss += loss.item()
            validation_acc += acc

            model_results.extend(p_outputs.cpu().detach().numpy())
            targets.extend(labels.cpu().numpy())

    validation_loss /= len(data_loader)
    validation_acc /= len(data_loader)
    metrics = calculate_metrics(np.array(targets), np.array(model_results))

    return validation_loss, validation_acc, metrics


def train(params, save_model_path):
    # get config
    hyperparameters = params['hyperparameters']
    model_config = params['model']
    exp_type = params['exp_type']
    with open(os.path.join(os.getcwd(), f'{save_model_path}/settings.txt'), 'w') as f:
        f.write(str(params))

    # get device
    if params['aux']['device'] == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device:', torch.cuda.get_device_name(device) if device != 'cpu' else 'cpu')
    else:
        device = 'cpu'
        print('Device: cpu')

    # get data
    train_loader, val_loader, test_loader, num_class, _ = get_loaders(params['data'], exp_type, hyperparameters['batch_size'])

    # define model
    model = ITMN(n_classes=num_class, **model_config).to(device)
    print('Number of model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # get criterion, optimizer/scheduler
    criterion = get_loss(hyperparameters['use_loss'], exp_type)
    optimizer = get_optimizer(model, hyperparameters, 'adamw')

    num_epochs = hyperparameters['epochs']
    best_loss = float('inf')
    best_score = 0
    train_loss_log = []
    validation_loss_log = []

    print('Start training...')
    print('Epochs:', num_epochs)
    print('Iterations per training epoch:', len(train_loader))
    print('Iterations per validation epoch:', len(val_loader))

    for epoch in range(num_epochs):
        print(f'####### Epoch [{epoch + 1}/{num_epochs}] #######')
        for param_group in optimizer.param_groups:
            if (epoch + 1) % 15 == 0:
                param_group['lr'] /= 10
            print('LR: {:.6f}'.format(param_group['lr']))

        # training
        t0 = time.time()
        training_loss, training_acc = train_one_epoch(train_loader, model, criterion, optimizer, device)

        # validation
        validation_loss, validation_acc, metrics = evaluate_one_epoch(val_loader, model, criterion, device)

        # test
        _, _, test_metrics = evaluate_one_epoch(test_loader, model, criterion, device)

        # calculate global loss/metrics and log process
        print("Epoch[{}/{}] - Loss: {:.4f} - Accuracy: {:.4f} - ValLoss: {:.4f} - ValAccuracy: {:.4f}, ETA: {:.0f}s"
              .format(epoch + 1, num_epochs, training_loss, training_acc, validation_loss, validation_acc,
                      time.time() - t0))
        print('Validation | AUC: {:.4f} | TPR: {:.4f}'.format(metrics['AUC'], metrics['TPR']))
        print('Test | AUC: {:.4f} | TPR: {:.4f}'.format(test_metrics['AUC'], test_metrics['TPR']))

        train_loss_log.append(training_loss)
        validation_loss_log.append(validation_loss)

        # save checkpoints
        if best_loss > validation_loss:
            best_loss = validation_loss
            print(f'Save best loss checkpoint at epoch {epoch + 1}')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(os.getcwd(), f'{save_model_path}/checkpoints/best_loss_checkpoint.pth'))

        score = test_metrics['TPR']
        if score > best_score:
            best_score = score
            print(f'Save best metric checkpoint at epoch {epoch + 1}')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': test_metrics,
            }, os.path.join(os.getcwd(), f'{save_model_path}/checkpoints/best_metric_checkpoint.pth'))

    loss_history = {
        'train': train_loss_log,
        'validation': validation_loss_log,
    }

    with open(f'{save_model_path}/history.pkl', 'wb') as f:
        pickle.dump(loss_history, f)


if __name__ == '__main__':
    # define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', type=str, default='super', help='Experiment type')
    args = parser.parse_args()
    assert args.exp_type in ['super', 'sub', 'rhythm', 'all', 'diag', 'form', 'cpsc']

    # get configuration
    config = get_config('config.yaml')
    config['exp_type'] = args.exp_type.lower()

    # initialization
    initialization(config['aux']['seed'])
    current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    model_name = f'{args.exp_type.upper()}_ITMN'
    save_model_path = f'logs/{args.exp_type.upper()}/{current_time}_{model_name}'
    init_dir(save_model_path)

    train(config, save_model_path)
