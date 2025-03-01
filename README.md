# IHFNet: Incomplete Multimodal Hierarchical Feature Fusion Network for Mild Cognitive Impairment Conversion Prediction



-----

## 1 Paper flowchart

![frame](assets/frame.jpg)



## 2 Source Tree

```
--Net
  |--api.py
  |--basic.py
  |--kan.py
  |--MultiLayerFusion.py
  |--poolformer.py
  |--ResnetEncoder.py
--Config.py
--Dataset.py
--loss_function.py
--main_rebuild.py
--observer.py
--README.md
--utils.py
```

## 3 Dataset Introduction

The dataset for this study is obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI), specifically the ADNI-1 and ADNI-2 cohorts. To prevent duplication, subjects present in both datasets were removed from ADNI-2. We selected T1-weighted sMRI, FDG-PET, and clinical data, categorized into four groups: normal controls (NC), sMCI, pMCI, and AD. Demographic information of the dataset is shown in Table below. Additionally, PET data is missing for 82 pMCI and 95 sMCI cases in ADNI-1, and for 1 pMCI and 30 sMCI cases in ADNI-2.

The ADNI dataset link: [ADNI | Alzheimer's Disease Neuroimaging Initiative](https://adni.loni.usc.edu/)





## 4 Training Process

