# IHFNet: Incomplete Multimodal Hierarchical Feature Fusion Network for Mild Cognitive Impairment Conversion Prediction



-----

## 1 Paper flowchart

![frame](assets/frame.jpg)



## 2 Source Tree

```
├── /Net
│   ├── api.py
│   ├── basic.py
│   ├── kan.py
│   ├── MultiLayerFusion.py
│   ├── poolformer.py
│   ├── ResnetEncoder.py
├── Config.py
├── Dataset.py
├── loss_function.py
├── main_rebuild.py
├── observer.py
├── README.md
├── utils.py
```

## 3 Dataset Introduction

The dataset for this study is obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI), specifically the ADNI-1 and ADNI-2 cohorts. To prevent duplication, subjects present in both datasets were removed from ADNI-2. We selected T1-weighted sMRI, FDG-PET, and clinical data, categorized into four groups: normal controls (NC), sMCI, pMCI, and AD. Demographic information of the dataset is shown in Table below. Additionally, PET data is missing for 82 pMCI and 95 sMCI cases in ADNI-1, and for 1 pMCI and 30 sMCI cases in ADNI-2.

The ADNI dataset link: [ADNI | Alzheimer's Disease Neuroimaging Initiative](https://adni.loni.usc.edu/)



## 4 Training Process

We use PyTorch version 2.6.0 with CUDA 11.8, executed on a single Nvidia V100 32GB GPU. We employed a `5-fold cross-validation` approach to ensure robust model evaluation. The model was trained from scratch in two stages, each comprising 150 epochs, with a `batch size of 8` to efficiently manage the data. To optimize the model parameters, we used the `AdamW optimizer` and set the `learning rate to 0.001` to ensure precise adjustments during the training process. Additionally, we implemented the` Cosine Learning Rate Scheduler`, with the hyperparameter $T_{max}$ set to 50, to dynamically adjust the learning rate throughout the training.

### 4.1 training step

To run our train code, please download and preprocess the ADNI dataset first, including the MRI, PET, and clinical modalities, then place the data corresponding to the three modalities into the folder shown below.

```
.
├── /ADNI1/MRI/
│   ├── xx.nii
│   ├── xx.nii
│   └── ....
├── /ADNI2/PET/
│   ├── xx.nii
│   ├── xx.nii
│   └── ...
└── ...

```

