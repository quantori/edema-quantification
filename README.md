[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8393566.svg)](https://doi.org/10.5281/zenodo.8393566)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8390417.svg)](https://doi.org/10.5281/zenodo.8390417)
[![DOI](https://img.shields.io/badge/DOI-10.1093/radadv/umae003-red.svg)](https://doi.org/10.1093/radadv/umae003)

# Explainable AI to identify radiographic features of pulmonary edema

<a name="table-of-contents"></a>
## 📖 Contents
- [Purpose](#purpose)
- [Data and Methods](#data-and-methods)
- [Results](#results)
- [Conclusion](#conclusion)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Access](#data-access)
- [How to Cite](#how-to-cite)

<a name="purpose"></a>
## 🎯 Purpose

This study aimed to develop a deep learning-based methodology for the detection and classification of radiographic features associated with pulmonary edema.

<a name="data-and-methods"></a>
## 🔬 Data and Methods

This retrospective study utilized a dataset from the MIMIC database comprising 1000 chest X-ray images from 741 patients with suspected pulmonary edema. The images were annotated by an experienced radiologist, who labeled radiographic manifestations of cephalization, Kerley lines, pleural effusion, bat wings, and infiltrate features of edema (<a href="#figure-1">Figure 1</a>). The proposed methodology involves two consecutive stages: lung segmentation and edema feature localization (<a href="#figure-2">Figure 2</a>). The segmentation stage is implemented using an ensemble of three networks. In the subsequent localization stage, we evaluated eight object detection networks, assessing their performance, with average precision (AP) and mean AP (mAP).

<p align="center">
  <img id="figure-1" width="100%" height="100%" src="media/annotation_method.png" alt="Annotation method">
</p>

<p align="left">
    <em><strong>Figure 1.</strong> Annotation methodology for chest X-rays. Various radiological features indicative of pulmonary edema are meticulously annotated for comprehensive analysis. Cephalization, representing the redistribution of blood flow in the upper lung fields, is delineated by cyan polylines. Kerley lines, indicating interstitial edema, are represented by green lines. Pleural effusions, characterized by accumulation of fluid in the pleural space, are highlighted in purple areas. Infiltrates, indicating alveolar edema or inflammation, are visualized by blue areas. Finally, bat wings, suggesting a butterfly pattern of alveolar edema, are highlighted in yellow areas. This detailed annotation method provides essential visual cues for accurate detection and analysis of radiological features.</em>
</p>

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

**Step 1: Download and install Miniconda**
``` bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

**Step 2: Clone the repository, create a conda environment, and install the requirements**
``` bash
git clone https://github.com/quantori/edema-quantification.git
cd edema-quantification
chmod +x create_env.sh
source create_env.sh
```

<a name="data-access"></a>
## 🔐 Data Access

You can find the labeled dataset and models on the Zenodo platform:
- Dataset: [https://zenodo.org/doi/10.5281/zenodo.8383776](https://zenodo.org/doi/10.5281/zenodo.8383776)
- Models: [https://zenodo.org/doi/10.5281/zenodo.8393565](https://zenodo.org/doi/10.5281/zenodo.8393565)

To download all research artifacts, including intermediate and visualization datasets, we recommend using the  [DVC framework](https://dvc.org/). Please note that you may encounter errors while downloading datasets or models due to insufficient permissions for accessing data stored on AWS S3. If you experience any issues with downloading models, please contact [Viacheslav Danilov](https://github.com/ViacheslavDanilov) at <a href="mailto:viacheslav.v.danilov@gmail.com">viacheslav.v.danilov@gmail.com</a> to gain access to the DVC repository.

**Step 1. To download the data via DVC, clone the repository:**
``` bash
git clone https://github.com/quantori/edema-quantification.git
```

**Step 2. Install DVC:**
``` bash
pip install dvc==2.58.2 dvc-s3==2.22.0
```

**Step 3. Download the datasets using DVC**

<p align="right">
    Table 1. Datasets used during the development of the proposed solution.
</p>

|                                                                     Dataset                                                                      |                                                                                                                                         Description                                                                                                                                         | Size, Gb |              Download Command               |
|:------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:-------------------------------------------:|
|              [SLY](https://github.com/quantori/edema-quantification/blob/3ddc1120a8dd58ce970380ef189f9238a0872013/dvc/data/sly.dvc)              | This dataset consists of 1,000 chest X-rays obtained from 741 patients, annotated by an experienced clinician using the [Supervisely](https://supervisely.com/) platform. The annotations are stored in JSON format, and the images consist of stacked frontal and horizontal chest X-rays. |   8.2    |       ```dvc pull dvc/data/sly.dvc```       |
|            [Edema](https://github.com/quantori/edema-quantification/blob/3ddc1120a8dd58ce970380ef189f9238a0872013/dvc/data/edema.dvc)            |                                                     This dataset comprises 2,978 chest X-ray studies of patients diagnosed with pulmonary edema, sourced from the [MIMIC database](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).                                                     |   12.4   |      ```dvc pull dvc/data/edema.dvc```      |
|   [Edema (stacked)](https://github.com/quantori/edema-quantification/blob/3ddc1120a8dd58ce970380ef189f9238a0872013/dvc/data/edema_stacked.dvc)   |                                                  This dataset consists of 3,816 stacked chest X-ray images of patients diagnosed with pulmonary edema, obtained from [MIMIC database](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).                                                  |   31.3   |  ```dvc pull dvc/data/edema_stacked.dvc```  |
|          [Healthy](https://github.com/quantori/edema-quantification/blob/3ddc1120a8dd58ce970380ef189f9238a0872013/dvc/data/healthy.dvc)          |                                                                This dataset comprises 3,136 chest X-ray studies of healthy patients, sourced from the [MIMIC database](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).                                                                 |   13.9   |     ```dvc pull dvc/data/healthy.dvc```     |
| [Healthy (stacked)](https://github.com/quantori/edema-quantification/blob/3ddc1120a8dd58ce970380ef189f9238a0872013/dvc/data/healthy_stacked.dvc) |                                                        This dataset consists of 4,269 stacked chest X-ray images obtained from healthy patients, obtained from [MIMIC database](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).                                                        |   34.5   | ```dvc pull dvc/data/healthy_stacked.dvc``` |
|             [Intermediate](https://github.com/quantori/edema-quantification/blob/3ddc1120a8dd58ce970380ef189f9238a0872013/dvc.lock)              |                                                                                             These are intermediate datasets generated during the execution of the DVC data processing pipeline.                                                                                             |   10.2   |           ```dvc pull dvc.yaml```           |

**Step 4. Download the models using DVC**

<p align="right">
    Table 2. Lung segmentation models used during the first stage of the workflow.
</p>

|                                                                           Model                                                                            |       Task        | Dice Score  | Size, Mb |                      Download Command                      |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------:|:-----------:|:--------:|:----------------------------------------------------------:|
| [DeepLabV3+](https://github.com/quantori/edema-quantification/blob/2b29e80654aca5822bba433e3e265968473b8bc2/dvc/models/lung_segmentation/DeepLabV3%2B.dvc) | Lung Segmentation |    94.8     |    30    | ```dvc pull dvc/models/lung_segmentation/DeepLabV3+.dvc``` |
|   [DeepLabV3](https://github.com/quantori/edema-quantification/blob/2b29e80654aca5822bba433e3e265968473b8bc2/dvc/models/lung_segmentation/DeepLabV3.dvc)   | Lung Segmentation |    93.8     |    30    | ```dvc pull dvc/models/lung_segmentation/DeepLabV3.dvc```  |
|         [FPN](https://github.com/quantori/edema-quantification/blob/2b29e80654aca5822bba433e3e265968473b8bc2/dvc/models/lung_segmentation/FPN.dvc)         | Lung Segmentation |    94.9     |    23    |    ```dvc pull dvc/models/lung_segmentation/FPN.dvc```     |
|     [Linknet](https://github.com/quantori/edema-quantification/blob/2b29e80654aca5822bba433e3e265968473b8bc2/dvc/models/lung_segmentation/Linknet.dvc)     | Lung Segmentation |    94.6     |   118    |  ```dvc pull dvc/models/lung_segmentation/Linknet.dvc```   |
|       [MAnet](https://github.com/quantori/edema-quantification/blob/2b29e80654aca5822bba433e3e265968473b8bc2/dvc/models/lung_segmentation/MAnet.dvc)       | Lung Segmentation |    94.5     |    54    |   ```dvc pull dvc/models/lung_segmentation/MAnet.dvc```    |
|         [PAN](https://github.com/quantori/edema-quantification/blob/2b29e80654aca5822bba433e3e265968473b8bc2/dvc/models/lung_segmentation/PAN.dvc)         | Lung Segmentation |    94.1     |    17    |    ```dvc pull dvc/models/lung_segmentation/PAN.dvc```     |
|      [PSPNet](https://github.com/quantori/edema-quantification/blob/2b29e80654aca5822bba433e3e265968473b8bc2/dvc/models/lung_segmentation/PSPNet.dvc)      | Lung Segmentation |    94.0     |   120    |   ```dvc pull dvc/models/lung_segmentation/PSPNet.dvc```   |
|    [Unet++](https://github.com/quantori/edema-quantification/blob/2b29e80654aca5822bba433e3e265968473b8bc2/dvc/models/lung_segmentation/Unet%2B%2B.dvc)    | Lung Segmentation |    94.6     |    37    |   ```dvc pull dvc/models/lung_segmentation/Unet++.dvc```   |
|        [Unet](https://github.com/quantori/edema-quantification/blob/2b29e80654aca5822bba433e3e265968473b8bc2/dvc/models/lung_segmentation/Unet.dvc)        | Lung Segmentation |    94.5     |   225    |    ```dvc pull dvc/models/lung_segmentation/Unet.dvc```    |

<p align="right">
    Table 3. Radiographic feature detection models used during the second stage of the workflow.
</p>

|                                                                            Model                                                                            |       Task        | mAP  | Size, Mb |                      Download Command                       |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------:|:----:|:--------:|:-----------------------------------------------------------:|
|        [ATSS](https://github.com/quantori/edema-quantification/blob/34b6e234e3efc8b01cdcd921d6766e9864fea515/dvc/models/feature_detection/ATSS.dvc)         | Feature Detection | 53.2 |   3070   |    ```dvc pull dvc/models/feature_detection/ATSS.dvc```     |
| [Cascade RPN](https://github.com/quantori/edema-quantification/blob/34b6e234e3efc8b01cdcd921d6766e9864fea515/dvc/models/feature_detection/Cascade_RPN.dvc)  | Feature Detection | 54.0 |   2520   | ```dvc pull dvc/models/feature_detection/Cascade_RPN.dvc``` |
| [Faster R-CNN](https://github.com/quantori/edema-quantification/blob/34b6e234e3efc8b01cdcd921d6766e9864fea515/dvc/models/feature_detection/Faster_RCNN.dvc) | Feature Detection | 50.9 |   2490   | ```dvc pull dvc/models/feature_detection/Faster_RCNN.dvc``` |
|        [FSAF](https://github.com/quantori/edema-quantification/blob/34b6e234e3efc8b01cdcd921d6766e9864fea515/dvc/models/feature_detection/FSAF.dvc)         | Feature Detection | 51.0 |   5660   |    ```dvc pull dvc/models/feature_detection/FSAF.dvc```     |
|         [GFL](https://github.com/quantori/edema-quantification/blob/34b6e234e3efc8b01cdcd921d6766e9864fea515/dvc/models/feature_detection/GFL.dvc)          | Feature Detection | 53.4 |   3200   |     ```dvc pull dvc/models/feature_detection/GFL.dvc```     |
|         [PAA](https://github.com/quantori/edema-quantification/blob/34b6e234e3efc8b01cdcd921d6766e9864fea515/dvc/models/feature_detection/PAA.dvc)          | Feature Detection | 50.6 |   3070   |     ```dvc pull dvc/models/feature_detection/PAA.dvc```     |
|        [SABL](https://github.com/quantori/edema-quantification/blob/34b6e234e3efc8b01cdcd921d6766e9864fea515/dvc/models/feature_detection/SABL.dvc)         | Feature Detection | 56.8 |   3300   |    ```dvc pull dvc/models/feature_detection/SABL.dvc```     |
|        [TOOD](https://github.com/quantori/edema-quantification/blob/34b6e234e3efc8b01cdcd921d6766e9864fea515/dvc/models/feature_detection/TOOD.dvc)         | Feature Detection | 50.6 |   3210   |    ```dvc pull dvc/models/feature_detection/TOOD.dvc```     |

<a name="how-to-cite"></a>
## 🖊️ How to Cite

Please cite [our paper](https://doi.org/10.1093/radadv/umae003) if you found our data, methods, or results helpful for your research:

> Danilov V.V., Makoveev A.O., Proutski A., Ryndova I., Karpovsky A., Gankin Y. (**2024**). _Explainable AI to identify radiographic features of pulmonary edema_. **Radiology Advances**. DOI: [https://doi.org/10.1093/radadv/umae003](https://doi.org/10.1093/radadv/umae003)
