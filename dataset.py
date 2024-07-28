import os
import ast

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import wfdb

from utils.preprocessing import butter_bandpass_filter

EXP_CONFIGS = {
    'SUPER': {
        'num_class': 5,
        'label_dict': {'NORM': 0, 'CD': 1, 'HYP': 2, 'MI': 3, 'STTC': 4},
        'label_list': ['NORM', 'CD', 'HYP', 'MI', 'STTC'],
    },
    'SUB': {
        'num_class': 23,
        'label_dict': {'AMI': 0, 'CLBBB': 1, 'CRBBB': 2, 'ILBBB': 3, 'IMI': 4, 'IRBBB': 5, 'ISCA': 6, 'ISCI': 7,
                       'ISC_': 8, 'IVCD': 9, 'LAFB/LPFB': 10, 'LAO/LAE': 11, 'LMI': 12, 'LVH': 13, 'NORM': 14,
                       'NST_': 15, 'PMI': 16, 'RAO/RAE': 17, 'RVH': 18, 'SEHYP': 19, 'STTC': 20, 'WPW': 21, '_AVB': 22},
        'label_list': ['AMI', 'CLBBB', 'CRBBB', 'ILBBB', 'IMI', 'IRBBB', 'ISCA', 'ISCI', 'ISC_', 'IVCD', 'LAFB/LPFB',
                       'LAO/LAE', 'LMI', 'LVH', 'NORM', 'NST_', 'PMI', 'RAO/RAE', 'RVH', 'SEHYP', 'STTC', 'WPW',
                       '_AVB'],
    },
    'RHYTHM': {
        'num_class': 12,
        'label_dict': {'SR': 0, 'AFIB': 1, 'STACH': 2, 'SARRH': 3, 'SBRAD': 4, 'PACE': 5, 'SVARR': 6, 'BIGU': 7,
                       'AFLT': 8, 'SVTAC': 9, 'PSVT': 10, 'TRIGU': 11},
        'label_list': ['SR', 'AFIB', 'STACH', 'SARRH', 'SBRAD', 'PACE', 'SVARR', 'BIGU',
                       'AFLT', 'SVTAC', 'PSVT', 'TRIGU']
    },
    'FORM': {
        'num_class': 19,
        'label_dict': {'NDT': 0, 'NST_': 1, 'DIG': 2, 'LNGQT': 3, 'ABQRS': 4, 'PVC': 5, 'STD_': 6, 'VCLVH': 7,
                       'QWAVE': 8, 'LOWT': 9, 'NT_': 10, 'PAC': 11, 'LPR': 12, 'INVT': 13, 'LVOLT': 14, 'HVOLT': 15,
                       'TAB_': 16, 'STE_': 17, 'PRC(S)': 18},
        'label_list': ['NDT', 'NST_', 'DIG', 'LNGQT', 'ABQRS', 'PVC', 'STD_', 'VCLVH', 'QWAVE', 'LOWT', 'NT_', 'PAC',
                       'LPR',
                       'INVT', 'LVOLT', 'HVOLT', 'TAB_', 'STE_', 'PRC(S)']
    },
    'DIAG': {
        'num_class': 44,
        'label_dict': {'NDT': 0, 'NST_': 1, 'DIG': 2, 'LNGQT': 3, 'NORM': 4, 'IMI': 5, 'ASMI': 6, 'LVH': 7, 'LAFB': 8,
                       'ISC_': 9,
                       'IRBBB': 10, '1AVB': 11, 'IVCD': 12, 'ISCAL': 13, 'CRBBB': 14, 'CLBBB': 15, 'ILMI': 16,
                       'LAO/LAE': 17,
                       'AMI': 18, 'ALMI': 19, 'ISCIN': 20, 'INJAS': 21, 'LMI': 22, 'ISCIL': 23, 'LPFB': 24, 'ISCAS': 25,
                       'INJAL': 26, 'ISCLA': 27, 'RVH': 28, 'ANEUR': 29, 'RAO/RAE': 30, 'EL': 31, 'WPW': 32,
                       'ILBBB': 33,
                       'IPLMI': 34, 'ISCAN': 35, 'IPMI': 36, 'SEHYP': 37, 'INJIN': 38, 'INJLA': 39, 'PMI': 40,
                       '3AVB': 41,
                       'INJIL': 42, '2AVB': 43},
        'label_list': ['NDT', 'NST_', 'DIG', 'LNGQT', 'NORM', 'IMI', 'ASMI', 'LVH', 'LAFB', 'ISC_', 'IRBBB', '1AVB',
                       'IVCD',
                       'ISCAL', 'CRBBB', 'CLBBB', 'ILMI', 'LAO/LAE', 'AMI', 'ALMI', 'ISCIN', 'INJAS', 'LMI', 'ISCIL',
                       'LPFB',
                       'ISCAS', 'INJAL', 'ISCLA', 'RVH', 'ANEUR', 'RAO/RAE', 'EL', 'WPW', 'ILBBB', 'IPLMI', 'ISCAN',
                       'IPMI',
                       'SEHYP', 'INJIN', 'INJLA', 'PMI', '3AVB', 'INJIL', '2AVB']
    },
    'ALL': {
        'num_class': 71,
        'label_dict': {'NDT': 0, 'NST_': 1, 'DIG': 2, 'LNGQT': 3, 'NORM': 4, 'IMI': 5, 'ASMI': 6, 'LVH': 7, 'LAFB': 8,
                       'ISC_': 9, 'IRBBB': 10,
                       '1AVB': 11, 'IVCD': 12, 'ISCAL': 13, 'CRBBB': 14, 'CLBBB': 15, 'ILMI': 16, 'LAO/LAE': 17,
                       'AMI': 18, 'ALMI': 19,
                       'ISCIN': 20, 'INJAS': 21, 'LMI': 22, 'ISCIL': 23, 'LPFB': 24, 'ISCAS': 25, 'INJAL': 26,
                       'ISCLA': 27, 'RVH': 28,
                       'ANEUR': 29, 'RAO/RAE': 30, 'EL': 31, 'WPW': 32, 'ILBBB': 33, 'IPLMI': 34, 'ISCAN': 35,
                       'IPMI': 36, 'SEHYP': 37,
                       'INJIN': 38, 'INJLA': 39, 'PMI': 40, '3AVB': 41, 'INJIL': 42, '2AVB': 43, 'ABQRS': 44, 'PVC': 45,
                       'STD_': 46,
                       'VCLVH': 47, 'QWAVE': 48, 'LOWT': 49, 'NT_': 50, 'PAC': 51, 'LPR': 52, 'INVT': 53, 'LVOLT': 54,
                       'HVOLT': 55,
                       'TAB_': 56, 'STE_': 57, 'PRC(S)': 58, 'SR': 59, 'AFIB': 60, 'STACH': 61, 'SARRH': 62,
                       'SBRAD': 63, 'PACE': 64,
                       'SVARR': 65, 'BIGU': 66, 'AFLT': 67, 'SVTAC': 68, 'PSVT': 69, 'TRIGU': 70},
        'label_list': ['NDT', 'NST_', 'DIG', 'LNGQT', 'NORM', 'IMI', 'ASMI', 'LVH', 'LAFB', 'ISC_', 'IRBBB', '1AVB',
                       'IVCD', 'ISCAL',
                       'CRBBB', 'CLBBB', 'ILMI', 'LAO/LAE', 'AMI', 'ALMI', 'ISCIN', 'INJAS', 'LMI', 'ISCIL', 'LPFB',
                       'ISCAS', 'INJAL',
                       'ISCLA', 'RVH', 'ANEUR', 'RAO/RAE', 'EL', 'WPW', 'ILBBB', 'IPLMI', 'ISCAN', 'IPMI', 'SEHYP',
                       'INJIN', 'INJLA',
                       'PMI', '3AVB', 'INJIL', '2AVB', 'ABQRS', 'PVC', 'STD_', 'VCLVH', 'QWAVE', 'LOWT', 'NT_', 'PAC',
                       'LPR', 'INVT',
                       'LVOLT', 'HVOLT', 'TAB_', 'STE_', 'PRC(S)', 'SR', 'AFIB', 'STACH', 'SARRH', 'SBRAD', 'PACE',
                       'SVARR', 'BIGU',
                       'AFLT', 'SVTAC', 'PSVT', 'TRIGU']
    },
    'CPSC': {
        'num_class': 9,
        'label_dict': {'NORM': 0, 'AFIB': 1, '1AVB': 2, 'CLBBB': 3, 'CRBBB': 4, 'PAC': 5, 'VPC': 6, 'STD_': 7, 'STE_': 8},
        'label_list': ['NORM', 'AFIB', '1AVB', 'CLBBB', 'CRBBB', 'PAC', 'VPC', 'STD_', 'STE_']
    },
}


class ECGDataset(Dataset):
    def __init__(self, dataset, base_data_path, exp_config, exp_type, signal_freq='LOW'):
        self.dataset = dataset
        self.base_data_path = base_data_path
        self.signal_freq = signal_freq
        self.exp_type = exp_type
        self.num_class = exp_config['num_class']
        self.label_dict = exp_config['label_dict']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.exp_type == 'cpsc':
            filename_lr, _, raw_label = self.dataset[idx]
            waveform = np.load(f'{self.base_data_path + filename_lr}.npy').T
        else:
            if self.signal_freq == 'LOW':
                filename_lr, _, raw_label = self.dataset[idx]
                waveform, _ = wfdb.rdsamp(self.base_data_path + filename_lr)
                waveform = butter_bandpass_filter(waveform.T, lowcut=0.5, highcut=45, fs=100).T
            else:
                _, filename_lr, raw_label = self.dataset[idx]
                waveform, _ = wfdb.rdsamp(self.base_data_path + filename_lr)
                waveform = butter_bandpass_filter(waveform.T, lowcut=0.5, highcut=45, fs=500).T
        waveform = np.nan_to_num(waveform)

        raw_label = ast.literal_eval(raw_label)
        label = np.zeros(self.num_class)
        for x in raw_label:
            label[self.label_dict[x]] = 1

        sample = {
            'waveform': torch.from_numpy(waveform).type(torch.FloatTensor),
            'label': torch.from_numpy(label).type(torch.FloatTensor)
        }

        return sample


def get_loaders(params, exp_type, batch_size):
    base_data_path = params[exp_type]['base_data_path']
    train_dataset_path = params[exp_type]['train_set_path']
    valid_dataset_path = params[exp_type]['valid_set_path']
    test_dataset_path = params[exp_type]['test_set_path']

    exp_config = EXP_CONFIGS[exp_type.upper()]
    num_class = exp_config['num_class']
    label_list = exp_config['label_list']

    train_set = pd.read_csv(train_dataset_path).to_numpy()
    valid_set = pd.read_csv(valid_dataset_path).to_numpy()
    test_set = pd.read_csv(test_dataset_path).to_numpy()

    train_dataset = ECGDataset(dataset=train_set, base_data_path=base_data_path,
                               exp_config=exp_config, exp_type=exp_type)
    valid_dataset = ECGDataset(dataset=valid_set, base_data_path=base_data_path,
                               exp_config=exp_config, exp_type=exp_type)
    test_dataset = ECGDataset(dataset=test_set, base_data_path=base_data_path,
                              exp_config=exp_config, exp_type=exp_type)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=4,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              num_workers=4)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=4)

    return train_loader, valid_loader, test_loader, num_class, label_list


if __name__ == '__main__':
    pass
