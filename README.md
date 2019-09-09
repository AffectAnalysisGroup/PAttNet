# PAttNet

Facial action units (AUs) refer to specific facial locations. Recent efforts in automatic AU detection have focused on learning their representations. Two factors have limited progress. One is that current approaches implicitly assume that facial patches are robust to head rotation. The other is that the relation between patches and AUs is pre-defined or ignored. Both assumptions are problematic. We propose a patch-attentive deep network called PAttNet for AU detection that learns mappings of patches and AUs, controls for 3D head and face rotation, and exploits co-occurrence among AUs. We encode patches with separate convolutional neural networks (CNNs) and weight the contribution of each patch to detection of specific AUs using a sigmoid patch attention mechanism. Unlike conventional softmax attention mechanisms, a sigmoidal attention mechanism allows multiple patches to contribute to detection of specific AUs. The latter is important because AUs often co-occur and multiple patches may be needed to detect them reliably. On the BP4D dataset, PAttNet improves upon state-of-the-art by 3.7\%. Visualization of the learned attention maps reveal power of this patch-based approach.

![BMVC_pipeline](https://user-images.githubusercontent.com/12033328/64030088-e0551400-cb13-11e9-8a31-c5ca19bebe3c.png)

This repository contains PyTorch implementation of [PAttNet](https://www.jeffcohn.net/wp-content/uploads/2019/07/BMVC2019_PAttNet.pdf.pdf) presented in our BMVC 2019 paper:

Itir Onal Ertugrul, Laszlo A. Jeni, and Jeffrey F. Cohn. PAttNet: Patch-attentive deep network for action unit detection. In BMVC, 2019. 

## Citation

If you use any of the resources provided on this page, please cite the following paper:

```
@inproceedings{ertugrul2019pattnet,
  title={PAttNet: Patch-attentive deep network for action unit detection},
  author={Onal Ertugrul, Itir and Jeni, L{\'a}szl{\'o} A and Cohn, Jeffrey F},
  booktitle={British Machine Vision Conference},
  year={2019}
}

```
