# Vessel-MAYON

Vessel-MAYON:A 3D Direction-Aware Deep Network for Intracranial Vessel Segmentation in MRA Images


## Installation

Follow the official nnU-Net v2 installation guide:  
[https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
Make sure the environment is properly set up and the dataset structure follows the nnU-Net requirements

you can also follow reuqirement:
```bash
pip install -r requirements.txt
```



## Project Structure
```text
Vessel-MAYON/
├── Vessel-MAYON/
│   ├── dataset/
│   │   ├── nnUNet_raw/           # Raw data
│   │   ├── nnUNet_preprocessed/  # Preprocessed data
│   │   └── nnUNet_results/       # Training results
│   └── nnunetv2/
│       └── training/
│           └── nnUNetTrainer/
│               ├── mayon                  # Model code
│               └── nnUNetTrainermayon     # Custom trainer
```


## Dataset Preparation

Organize your dataset into the required `nnUNet_raw` structure.  
Assign a **dataset ID** (e.g., 639, 712) before preprocessing.


## ⚙️ Workflow


### 1. Parameter Explanation


| Parameter | Description | Example |
|-----------|-------------|---------|
| `-d <DATASET_ID>` | Dataset ID assigned to your dataset | `-d 639` |
| `--verify_dataset_integrity` | (Optional) Checks dataset integrity before preprocessing | `--verify_dataset_integrity` |
| `-c 3d_fullres` | Use 3D full-resolution configuration | `-c 3d_fullres` |
| `<FOLD>` | Fold number for training (0–4 or `all`) | `1` |
| `-tr <TRAINER_NAME>` | Trainer class name, must be consistent between training and prediction | `-tr nnUNetTrainerCustom` |
| `-i <INPUT_FOLDER>` | Input folder containing images for prediction | `-i ./imagesTs/` |
| `-o <OUTPUT_FOLDER>` | Output folder where predictions will be saved | `-o ./results/` |
| `--save_probabilities` | (Optional) Save probability maps during prediction | `--save_probabilities` |


### 2. Preprocessing
```bash
nnUNetv2_plan_and_preprocess \
    -d <DATASET_ID> \
    --verify_dataset_integrity \
    -c 3d_fullres
```


### 3. Training
```bash
nnUNetv2_train \
    <DATASET_ID> \
    3d_fullres \
    <FOLD> \
    -tr <TRAINER_NAME>
```


### 4. Prediction
```bash
nnUNetv2_predict \
    -d <DATASET_ID> \
    -i <INPUT_FOLDER> \
    -o <OUTPUT_FOLDER> \
    -f <FOLD> \
    -tr <TRAINER_NAME> \
    -c 3d_fullres \
    --save_probabilities
```


## Data & Pretrained Models

All datasets and pretrained weights are packed and shared here:  
[Google Drive Link](https://drive.google.com/drive/folders/1dhg0t20wR4VvhiNyz6uYleklYyrBINYd?dmr=1&ec=wgc-drive-globalnav-goto)

---

### Directory Structure
```text
Vessel-MAYON/
├── pretrained_mayon/ # Pretrained model weights
├── raw dataset/ # Raw training dataset
└── test set/ # Test dataset
```







