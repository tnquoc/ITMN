# ITMN

This is the official implementation of paper "Inception Time Mamba Network with Distribution
Balanced Loss for Multi-Label ECG Classification"

## Prerequisites

- Install dependencies: ```pip install -r requirements.txt```
- Install Mamba SSM: ```pip install mamba-ssm``` (If there are any errors, please following this [repo](https://github.com/state-spaces/mamba) to download the Mamba SSM module)
- Prepare the **wfdb** library
- GPU is needed to run support Mamba SSM module.

## Dataset

- Download PTB-XL dataset ([link](https://www.physionet.org/content/ptb-xl/1.0.1/))
- Download CPSC2018 dataset ([link](http://2018.icbeb.org/Challenge.html))

## Training

1. Prepare Dataset
   - We provide the train/validation/test set in .csv format in folder **data** corresponding to all classification task within PTB-XL and CPSC2018 datasets.
   - Modify field **base_data_path** in file **config.yaml** by the corresponding path to the downloaded PTB-XL and CPSC2018 datasets.
2. Modify field **use_loss** (selected from **["DB", "FOCAL", "CB", "WBCE", "STANDARD"]**) in file **config.yaml** to train model with different loss functions:
   - STANDARD: Binary-Cross Entropy (BCE) Loss
   - WBCE: Weighted BCE Loss
   - FOCAL: Focal Loss
   - CB: Class-Balanced Focal Loss
   - DB: Distribution-Balanced Loss
3. Run file **main.py** with specific arguments to train model:
   - exp_type: experiment type (selected from **["super", "sub", "rhythm", "all", "diag", "form", "cpsc"]**) corresponding to different classification tasks within PTB-XL and CPSC2018 datasets.
   - Example: to train model for "SUPER" task in PTB-XL dataset, run the command
   ```commandline
   python main.py --exp_type super
   ```
4. Structure:
   - Folder **logs** with other sub-folders are created.
   - Checkpoints are saved in folder **logs/{exp_type}/checkpoints**

## Inference

1. Modify value in file **config.yaml**:
   - test_ckpt_path: path to checkpoint used to test
2. We release [checkpoints](https://drive.google.com/drive/folders/1YfN9upk4ZPwADSsbL5BjUvl3GYzrVBux?usp=sharing) corresponding to each classification task in the PTB-XL dataset reported in the paper.
3. Run file **test.py** with specific arguments to inference:
   - exp_type: experiment type (selected from **["super", "sub", "rhythm", "all", "diag", "form", "cpsc"]**)
   - Note: the value of exp_type must correspond to the checkpoint.
   - Example: to test model for "SUPER" task in PTB-XL dataset, run the command
   ```commandline
   python test.py --exp_type super
   ```

## Citation