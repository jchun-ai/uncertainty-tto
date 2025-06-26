# 📌 1. Project Overview

This project aims to develop and maintain a robust AI-based segmentation framework for adaptive radiotherapy applications.  
The primary focus is to automate the delineation of anatomical structures—such as organs-at-risk (OARs) and tumor volumes—on CT and MRI, enabling faster and more consistent treatment planning.

---

### 🧭 Objectives

- Accelerate segmentation workflows in radiation oncology
- Support high-fidelity auto-contouring with transformer-based models
- Enable uncertainty-aware decision-making in clinical scenarios
- Facilitate test-time model adaptation to individual patient anatomy

### 🧩 Key Features

- 3D Transformer-based segmentation models (e.g., Swin UNETR)
- Monte Carlo Dropout and Ensemble Uncertainty Estimation
- Modularized PyTorch-based training and inference pipelines
- Test-time adaptation strategies for personalized AI
- Reproducible evaluation on public and internal datasets (e.g., HNTS-MRG 2024, internal pancreas/cyst)

### ⚠️ Notes

- This project is under active development and intended for internal research use only.

<br/>
<br/>  

# ⚙️ 2. Environment & Dependencies

This project is based on a PyTorch Docker environment and built for GPU-enabled systems with CUDA 12.1.  
The base image used is:

```
pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel
```

All core dependencies are installed via `monai_requirements.txt`.

---

### 🐳 Docker-based Setup (Recommended)

Use the following Dockerfile snippet to replicate the development environment:

```Dockerfile
# Base image
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

# Working directory (YOUR DIRECTORY)
WORKDIR /Morfeus/JChun

# Set non-interactive shell mode
ENV DEBIAN_FRONTEND=noninteractive

# Copy requirements
COPY monai_requirements.txt /Morfeus/JChun

# Upgrade pip and install dependencies
RUN python3 -m pip install --no-cache-dir --upgrade --pre pip
RUN python3 -m pip install --no-cache-dir -r monai_requirements.txt
```

### 📦 Core Dependencies (via `monai_requirements.txt`)

| Package            | Version Range   | Purpose                                  |
|--------------------|------------------|-------------------------------------------|
| `monai[all]`       | 1.4              | Medical imaging framework                |

---

### ⚠️ Notes

- CUDA 12.1 with cuDNN 8 is required for full GPU acceleration.
- Avoid mixing PyTorch or MONAI versions not explicitly compatible with CUDA 12.1.

<br/>
<br/>  

# 🏗 3. Directory Structure


```
.
├── ! history/                          # Auto-generated log and output 
├── 3d_segmentation/                    # Core training/inference scripts
│   ├── swin_unetr_btcv_segmentation_3d_singleCh.py
│   ├── swin_unetr_btcv_segmentation_3d_singleCh_test.py
│   └── swin_unetr_btcv_segmentation_3d_singleCh.ipynb    # Jupyter development notebook
├── requirements.txt                    # Dependency list for pip
├── Dockerfile_uncertainty_tto          # Dockerfile (based on PyTorch 2.1.1 + CUDA 12.1)
└── README.md                           
directory per experiment

```

### 🔖 Notes

- `3d_segmentation/`: Contains task-specific scripts for training and inference (single channel segmentation)
- `swin_unetr_btcv_segmentation_3d_singleCh.ipynb`: Used for quick prototyping and debugging
- `Dockerfile_uncertainty_tto`: Builds the MONAI runtime environment on top of `pytorch:2.1.1-cuda12.1-cudnn8-devel`

---
### 🧪 Dataset References 

This project currently supports experiments on the following datasets:

- **HNT MR GTVs**  
  - JSON: `dataset_HNT_fold0.json`  
  - Foreground class: `1`, `2`
  - Additional: `--intensity_range 0.0 255.0`  


<br/>
<br/>  

# 🧠 4. Model Summary

The core model architecture in this project is based on **Swin UNETR**, a 3D transformer-based network for medical image segmentation.  
It combines the hierarchical attention mechanism of Swin Transformers with the encoder-decoder structure of U-Net.

---

### 🧬 Model Architecture

- **Backbone**: Swin Transformer (3D patch embedding + window attention)
- **Decoder**: U-Net-style upsampling path with skip connections
- **Input**: 1-channel 3D CT or MRI images
- **Output**: Binary or multi-label segmentation masks (`channels_out=2` or more)
- **Dropout**: Optional MC Dropout for uncertainty estimation (`--prob_drop`)
- **Implementation**: Based on MONAI `SwinUNETR` and customized pipeline

> 🔎 For detailed code, see:  
> `3d_segmentation/swin_unetr_btcv_segmentation_3d_singleCh.py`  
> `3d_segmentation/swin_unetr_btcv_segmentation_3d_singleCh_test.py`

---

### 🧪 Loss & Metrics

- **Loss Function**:  
  - `DiceCELoss`: Combines Dice Loss (structure-aware) and Cross Entropy Loss (class confidence)
- **Metrics Used**:  
  - Dice Similarity Coefficient (DSC)  
  - 95% Hausdorff Distance (HD95)  
  - Mean Surface Distance (MSD)

---

### 🧭 Supported Modes

| Mode              | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `train`           | Standard training mode with caching, augmentation, and checkpoint saving     |
| `test`            | Evaluation on test set with optional softmax, dropout, and postprocessing    |
| `personalization` | Test-time fine-tuning using uncertainty-guided selection or fixed iterations |

> 🧠 Personalization mode is configured using dropout-based uncertainty and supports case-wise adaptation.

---

### 📈 Performance Example (HNT, Fold 0)

| Model Type         | Mean DSC (Foreground Classes) |
|--------------------|-------------------------------|
| Baseline Model     | 0.XXX                         |
| Personalized Model | 0.XXX                         |

> 🧪 Dataset: `HNT MR GTVs`  
> `channels_out=2`, `fg_class=1 2`

---

### 🧠 Notes

- The model supports **foreground class masking** using `--fg_class`  
  → This allows combining multiple semantic labels (e.g., `1`, `2`) into a **single binary foreground mask** for training.  
  → For example, in a multi-class dataset where `label==1` is cyst and `label==2` is duct, passing `--fg_class 1 2` will treat both as foreground (`1`) and all others as background (`0`).
- `--channels_out` defines the number of output classes. For binary segmentation (foreground vs. background), set `channels_out=2`.
- `--prob_drop` defines dropout probability used in both training and test-time inference for uncertainty-aware personalization.

<br/>
<br/>  

# 🚀 5. How to Run

> 📓 For a quick overview with runnable examples, see the notebook:  
> `3d_segmentation/swin_unetr_btcv_segmentation_3d_singleCh.ipynb`  
> It demonstrates the full pipeline including argument setup, visualization, and output structure.

This project supports three execution modes: `train`, `test`, and `personalization`.  
All are configured via CLI arguments and launched from Python scripts inside the `3d_segmentation/` directory.


---

### 🧪 Training

```bash
python 3d_segmentation/swin_unetr_btcv_segmentation_3d_singleCh.py \
  -exp_name '<EXP_NAME>' \
  -json_dir './data/Dataset001_HNT/dataset_HNT_fold0.json' \
  -size_cache_train 130 \
  -size_cache_valid 35 \
  -max_iterations 30000 \
  -channels_out 2 \
  -num_samples 10 \
  -fg_class 1 2 \
  -prob_drop 0.05
```

> Logs, metrics, and experiment backups will be saved under:  
> `! history/<EXP_NAME>/`  
> with results stored in `results/`, and script/yaml backups timestamped.

---

### 🔎 Inference (Testing)

```bash
python 3d_segmentation/swin_unetr_btcv_segmentation_3d_singleCh_test.py \
  -exp_name '<EXP_NAME>' \
  -json_dir './data/Dataset001_HNT/Dataset001_HNT/dataset_HNT_fold0.json' \
  -channels_out 2 \
  -fg_class 1 2 \
  -prob_drop 0.05 \
  -postfix 'do005'
```

> Prediction NIfTI files, log output, and per-case DSC results will be saved in:  
> `! history/<EXP_NAME>/results_<postfix>/`

---

### 🧠 Personalization

This mode performs patient-specific fine-tuning at test-time using a small number of training samples (e.g., `1`)  
and Monte Carlo Dropout-based uncertainty estimation.

```bash
python 3d_segmentation/swin_unetr_btcv_segmentation_3d_singleCh.py \
  --exp_name '<EXP_NAME>' \
  --json_dir './data/Dataset001_HNT/dataset_HNT_overfit0.json' \
  --size_cache_train 1 \
  --size_cache_valid 1 \
  --max_iterations 100 \
  --is_overfit 1 \
  --channels_out 2 \
  --prob_drop 0.05 \
  --postfix 'do005' \
  --eval_num 1 \
  --num_samples 10
```

> Personalized model weights and metrics will be saved under:  
> `! history/<EXP_NAME>/results_personalized_<postfix>/`

> 🧪 Personalization and inference jobs should use `--postfix` to avoid overwriting results.

---

<br/>
<br/>  

# 🔍 6. Evaluation & Logging

- All logs, metrics, and outputs are saved under:  
  `! history/<EXP_NAME>/`
- Training logs are redirected to `output.log`, and model checkpoints are stored in `results*/`
- Metric CSVs include:
  - `epoch_loss_values.csv`
  - `metric_values.csv` (DSC, HD95, MSD)
  - `validation_dsc.csv` (per-case DSC)
- Personalized runs are saved in:  
  `results_personalized_<postfix>/`
- Monte Carlo uncertainty values (`std_values`) are recorded for testing & each adaptation step.

<br/>
<br/>  

# ✅ 7. Contribution Checklist

- [✅] Adapted MONAI's official tutorial code to suit this project's personalization pipeline.
- [🛠️] Wrote a companion `.ipynb` file to showcase typical usage and output interpretation. *(In progress)*
- [❌] Update with publication link (e.g., arXiv or journal page) once available.

<br/>
<br/>  
