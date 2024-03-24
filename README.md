[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8390417.svg)](https://doi.org/10.5281/zenodo.8390417)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8393566.svg)](https://doi.org/10.5281/zenodo.8393566)
[![DOI](https://img.shields.io/badge/DOI-10.1093/radadv/umae003-B31B1B)](https://doi.org/10.1093/radadv/umae003)

# Explainable AI to identify radiographic features of pulmonary edema

<a name="contents"></a>
## 📖 Contents
- [Introduction](#introduction)
- [Data](#data)
- [Methods](#methods)
- [Results](#results)
- [Conclusion](#conclusion)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Access](#data-access)
- [How to Cite](#how-to-cite)

<a name="introduction"></a>
## 🎯 Introduction
This repository contains the research and findings of our study on the automated detection of pulmonary edema from chest X-ray images using advanced machine learning techniques. Pulmonary edema, a condition characterized by fluid accumulation in the lungs, is a critical indicator of various underlying health issues, including congestive heart failure. The timely and accurate diagnosis of this condition is crucial for effective patient management and treatment planning.

<a name="data"></a>
## 📁 Data
This study utilized a dataset comprised of 1000 chest X-ray images sourced from the MIMIC database, representing 741 patients suspected of having pulmonary edema. The images were annotated by an expert radiologist, identifying key radiographic signs of edema such as _cephalization_, _Kerley lines_, _pleural effusion_, _bat wings_, and _infiltrates_ (<a href="#figure-1">Figure 1</a>). Each feature was meticulously labeled, employing polylines for cephalization and Kerley lines, and segmentation masks for pleural effusions, infiltrates, and bat wings. The selection of 1000 cases was based on a balance between dataset size and the practicality of managing annotation resources and computational demands, ensuring a robust dataset for model training and evaluation. The annotation process was facilitated using the [Supervisely platform](https://supervisely.com/), emphasizing the detailed and patient-centric approach in dataset preparation.

<p align="center">
  <img id="figure-1" width="100%" height="100%" src="media/annotation_method.png" alt="Annotation method">
</p>

<p align="left">
    <em><strong>Figure 1.</strong> Annotation methodology for chest X-rays. Various radiological features indicative of pulmonary edema are meticulously annotated for comprehensive analysis. Cephalization, representing the redistribution of blood flow in the upper lung fields, is delineated by cyan polylines. Kerley lines, indicating interstitial edema, are represented by green lines. Pleural effusions, characterized by accumulation of fluid in the pleural space, are highlighted in purple areas. Infiltrates, indicating alveolar edema or inflammation, are visualized by blue areas. Finally, bat wings, suggesting a butterfly pattern of alveolar edema, are highlighted in yellow areas. This detailed annotation method provides essential visual cues for accurate detection and analysis of radiological features.</em>
</p>

<a name="methods"></a>
## 🔬 Methods
The methodology employed in this study encompasses two main stages: lung segmentation and edema feature detection (<a href="#figure-2">Figure 2</a>). The lung segmentation stage utilized an ensemble of three distinct neural networks to enhance segmentation accuracy. After segmentation, eight object detection networks were employed for edema feature localization. Each network was rigorously evaluated based on their average precision (AP) and mean average precision (mAP) to ascertain their effectiveness in accurately identifying and localizing radiographic features indicative of pulmonary edema. This dual-stage approach, integrating lung segmentation with targeted feature detection, represents a comprehensive strategy for identifying pulmonary edema from chest X-rays.

The segmentation stage was crucial for accurately delineating lung boundaries, leveraging an ensemble approach to combine the strengths of multiple segmentation models such as [DeepLabV3](http://arxiv.org/abs/1706.05587), [MA-Net](https://ieeexplore.ieee.org/document/9201310), and [FPN](). This step ensured the precise extraction of lung regions, which is foundational for effective feature detection.

In the detection stage, the study evaluated eight different object detection networks, each with their strengths in identifying specific features of pulmonary edema. The following networks were used: [TOOD](https://ieeexplore.ieee.org/document/9710724), [GFL](https://arxiv.org/abs/2006.04388v1), [PAA](https://dl.acm.org/doi/10.1007/978-3-030-58595-2_22), [SABL](https://link.springer.com/chapter/10.1007/978-3-030-58548-8_24), [FSAF](https://ieeexplore.ieee.org/document/8953532), [Cascade RPN](https://arxiv.org/abs/1909.06720), [ATSS](https://ieeexplore.ieee.org/document/9156746), and [Faster R-CNN](https://ieeexplore.ieee.org/document/7485869). The performance of these networks was measured using AP and mAP metrics, providing a quantitative basis for comparing their effectiveness.

<p align="center">
  <img id="figure-2" width="100%" height="100%" src="media/proposed_method.png" alt="Proposed method">
</p>

<p align="left">
    <em><strong>Figure 2.</strong> Schematic representation of the proposed approach. The original CXR is fed into the first segmentation stage where a lung mask is predicted. The CXR is then cropped using the lung mask and is further fed into detection stage to identify radiological features that can be used for further clinical analysis.</em>
</p>

<a name="results"></a>
## 📈 Results
The [SABL](https://arxiv.org/abs/1912.04260) model emerged as the top performer, achieving the highest mAP of 0.568 and excelling in the detection of effusion, infiltrate and bat wings (<a href="#figure-3">Figure 3</a>). Notably, the [TOOD](https://arxiv.org/abs/2108.07755) model demonstrated robust capabilities, particularly excelling in bat wing detection with an AP score of 0.918. The [Cascade RPN](https://arxiv.org/abs/1909.06720) and [GFL](https://arxiv.org/abs/2006.04388) models maintained consistently strong performance across all features. The [PAA](https://arxiv.org/abs/2007.08103) and [FSAF](https://arxiv.org/abs/1903.00621) models exhibited balanced performance, achieving mAP scores of 0.506 and 0.510, respectively. Surprisingly, the baseline [Faster R-CNN](https://arxiv.org/abs/1506.01497) model delivered competitive results with an mAP of 0.509. In addition, we evaluated the latency of these models, revealing processing times ranging from 42 ms to 104 ms per image. These latency values provide valuable insight into the real-time efficiency of each model, enhancing our understanding of their practical utility.

<p align="center">
  <img id="figure-3" width="100%" height="100%" src="media/model_performance.png" alt="Model performance">
</p>

<p align="center">
    <em><strong>Figure 3.</strong> Comparison of the networks based on their mAP scores, latency, and the number of parameters.</em>
</p>

| ![Bat](media/predictions_bat.png "Bat") | ![Effusion](media/predictions_effusion.png "Effusion") |
|:---------------------------------------:|:------------------------------------------------------:|
|                 (a) Bat                 |                      (b) Effusion                      |

<p align="center">
    <em><strong>Figure 4.</strong>Comparison of predictions and their confidences: ground truth (purple boxes and masks) vs. network predictions (yellow boxes).</em>
</p>

<a name="conclusion"></a>
## 🏁 Conclusion
The proposed methodology effectively highlighted and classified pulmonary edema features, positioning it as a promising candidate for the development of a clinical support tool aimed at assisting radiologists in the diagnosis and severity assessment of pulmonary edema.

<a name="requirements"></a>
## 💻 Requirements

- Operating System
  - [x] macOS
  - [x] Linux
  - [x] Windows (limited testing carried out)
- Python 3.8.x
- Required core packages: [dev.txt](https://github.com/quantori/edema-quantification/blob/main/requirements/dev.txt)

<a name="installation"></a>
## ⚙ Installation

**Step 1:** Download and install Miniconda
``` bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

**Step 2:** Clone the repository, create a conda environment, and install the requirements
``` bash
git clone https://github.com/quantori/edema-quantification.git
cd edema-quantification
chmod +x create_env.sh
source create_env.sh
```

<a name="data-access"></a>
## 🔐 Data Access

All essential components of the study, including the curated dataset and trained models, have been made publicly available:
- **Dataset:** [https://zenodo.org/doi/10.5281/zenodo.8383776](https://zenodo.org/doi/10.5281/zenodo.8383776)
- **Models:** [https://zenodo.org/doi/10.5281/zenodo.8393565](https://zenodo.org/doi/10.5281/zenodo.8393565)

<a name="how-to-cite"></a>
## 🖊️ How to Cite

Please cite [our paper](https://doi.org/10.1093/radadv/umae003) if you found our data, methods, or results helpful for your research:

> Danilov V.V., Makoveev A.O., Proutski A., Ryndova I., Karpovsky A., Gankin Y. (**2024**). _Explainable AI to identify radiographic features of pulmonary edema_. **Radiology Advances**. DOI: [https://doi.org/10.1093/radadv/umae003](https://doi.org/10.1093/radadv/umae003)
