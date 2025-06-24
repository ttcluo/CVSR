# CVSR:Complex-Valued Networks for Video Super-Resolution

Author : [Chuan Luo](https://orcid.org/0009-0000-7660-4239)*

(* : equal contribution)

### [News]
* --

## Abstract
  > In video super-resolution (VSR), bidirectional sequences (i.e., forward and backward sequences) inherently describe the temporal ﬂow and provide oriented motion information for high-resolution reconstruction. However, few works consider further modeling of bidirectional features, and they only aggregate these forward features and backward features into a group of ordinary feature maps via concatenation. Therefore, the correlation between the two features in opposite directions remains to be explored. In this paper, we provide a novel viewpoint for VSR, which constructs a higher relationship with respect to the forward features and backward features in the complex number domain. We respectively treat a forward feature and a backward feature as the real component and the imaginary component of a complex number and compute them according to the rules of complex number. This computation allows the bidirectional features to embrace a more coupling relation, so as to excavate high-dimension information used for reconstruction. Speciﬁcally to achieve complex VSR framework, we present two transplantable modules, i.e., complex transform and complex attention. The complex transform aims to generally change the bidirectional features to complex features by constructing a complex feature extraction. The complex attention can adaptively rectify the complex features for each complex feature map, so as to project a feature map to a speciﬁc frequency. In addition, we analyze the superiority of complex representations compared with real-valued representations. Extensive experiments on representative bidirectional networks (BasicVSR and IconVSR) and their improved form (IBRN) show the effectiveness and robustness of complex VSR framework.

## Framework of CVSR

<img src = "./asset/framework.png" width="100%" height="100%"/>

## Usage
### Data Preparation
comming soon

### Training
comming soon

### Evaluation
comming soon

### Pretrained Model
Pretrained Model(https://github.com/ttcluo/CVSR/experiments/models)

## Acknowledgement

Our code is built upon [BasicSR](https://github.com/XPixelGroup/BasicSR), which is an open-source image and video restoration toolbox based on [PyTorch](https://pytorch.org/). Thanks to the code reference from:

- [IBRN_VSR](https://github.com/hengliusky/IBRN_VSR)
