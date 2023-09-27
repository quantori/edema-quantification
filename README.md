# Bridging the Gap: Explainable AI for Radiologists in the Detection and Classification of Radiographic Pulmonary Edema Features

## ℹ️ Purpose

This study aimed to develop a deep learning-based methodology for the detection and classification of radiographic features associated with pulmonary edema.

## ℹ️ Data and Methods

This retrospective study utilized a dataset from the MIMIC database comprising 1000 chest X-rays from 741 patients with suspected pulmonary edema. These images were manually annotated by an experienced radiologist, followed by lung segmentation using an ensemble of three segmentation networks. Subsequently, eight object detection networks were employed to identify radiological features categorized by severity. As a final step in the methodology, post-processing included box confidence filtering and soft non-maximum suppression. Descriptive statistics, including Average Precision (AP), mean Average Precision (mAP), and latency calculations, were used to evaluate performance, providing a comprehensive analysis of radiological features associated with pulmonary edema.

<p align="center">
  <img width="100%" height="100%" src="media/annotation_method.png" alt="Annotation method">
</p>

<p align="left">
    Figure 1. Annotation methodology for chest X-rays. Various radiological features indicative of pulmonary edema are meticulously annotated for comprehensive analysis. Cephalization, representing the redistribution of blood flow in the upper lung fields, is delineated by cyan polylines. Kerley lines, indicating interstitial edema, are represented by green lines. Pleural effusions, characterized by accumulation of fluid in the pleural space, are highlighted in purple areas. Infiltrates, indicating alveolar edema or inflammation, are visualized by blue areas. Finally, bat wings, suggesting a butterfly pattern of alveolar edema, are highlighted in yellow areas. This detailed annotation method provides essential visual cues for accurate detection and analysis of radiological features.
</p>

<p align="center">
  <img width="100%" height="100%" src="media/proposed_method.png" alt="Proposed method">
</p>

<p align="left">
    Figure 2. Schematic representation of the proposed approach. The original CXR is fed into the first segmentation stage where a lung mask is predicted. The CXR is then cropped using the lung mask and is further fed into detection stage to identify radiological features that can be used for further clinical analysis.
</p>

## ℹ️ Results

The [SABL](https://arxiv.org/abs/1912.04260) model emerged as the top performer, achieving the highest mAP of 0.568 and excelling in the detection of effusion, infiltrate and bat wings. Notably, the [TOOD](https://arxiv.org/abs/2108.07755) model demonstrated robust capabilities, particularly excelling in bat wing detection with an AP score of 0.918. The [Cascade RPN](https://arxiv.org/abs/1909.06720) and [GFL](https://arxiv.org/abs/2006.04388) models maintained consistently strong performance across all features. The [PAA](https://arxiv.org/abs/2007.08103) and [FSAF](https://arxiv.org/abs/1903.00621) models exhibited balanced performance, achieving mAP scores of 0.506 and 0.510, respectively. Surprisingly, the baseline [Faster R-CNN](https://arxiv.org/abs/1506.01497) model delivered competitive results with an mAP of 0.509. In addition, we evaluated the latency of these models, revealing processing times ranging from 42 ms to 104 ms per image. These latency values provide valuable insight into the real-time efficiency of each model, enhancing our understanding of their practical utility.

<p align="center">
  <img width="100%" height="100%" src="media/model_performance.png" alt="Model performance">
</p>

<p align="left">
    Figure 3. Comparison of the networks based on their mAP scores, latency, and the number of parameters.
</p>

| ![Bat](media/predictions_bat.png "Bat") | ![Effusion](media/predictions_effusion.png "Effusion") |
|:---------------------------------------:|:------------------------------------------------------:|
|                 (a) Bat                 |                      (b) Effusion                      |

<p align="center">
    Figure 4.Comparison of predictions and their confidences: ground truth (purple boxes and masks) vs. network predictions (yellow boxes).
</p>

## ℹ️ Conclusion

The proposed methodology effectively highlighted and classified pulmonary edema features, positioning it as a promising candidate for the development of a clinical support tool aimed at assisting radiologists in the diagnosis and severity assessment of pulmonary edema.

## ℹ️ Requirements

- Linux or macOS (Windows has not been officially tested)
- Python 3.8.x

## ℹ️ Installation

Step 1: Download and install Miniconda
``` bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Step 2: Clone the repository, create a conda environment, and install the requirements for the repository
``` bash
git clone https://github.com/quantori/edema-quantification.git
cd edema-quantification
chmod +x create_env.sh
source create_env.sh
```

Step 3: Initialize git hooks using the pre-commit framework
``` bash
pre-commit install
```

Step 4: Download datasets using DVC
- Source datasets
``` bash
dvc pull dvc/data/edema.dvc
dvc pull dvc/data/healthy.dvc
```
- Stacked datasets
``` bash
dvc pull dvc/data/edema_stacked.dvc
dvc pull dvc/data/healthy_stacked.dvc
```
NOTE: Since data storage is organized through AWS S3, you should first request access to this repository by configuring your AWS credentials.
