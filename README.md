# PAttNet

Facial action units (AUs) refer to specific facial locations. Recent efforts in automatic AU detection have focused on learning their representations. Two factors have limited progress. One is that current approaches implicitly assume that facial patches are robust to head rotation. The other is that the relation between patches and AUs is pre-defined or ignored. Both assumptions are problematic. We propose a patch-attentive deep network called PAttNet for AU detection that learns mappings of patches and AUs, controls for 3D head and face rotation, and exploits co-occurrence among AUs. We encode patches with separate convolutional neural networks (CNNs) and weight the contribution of each patch to detection of specific AUs using a sigmoid patch attention mechanism. Unlike conventional softmax attention mechanisms, a sigmoidal attention mechanism allows multiple patches to contribute to detection of specific AUs. The latter is important because AUs often co-occur and multiple patches may be needed to detect them reliably. On the BP4D dataset, PAttNet improves upon state-of-the-art by 3.7\%. Visualization of the learned attention maps reveal power of this patch-based approach.

![BMVC_pipeline_high_res](https://user-images.githubusercontent.com/12033328/64029504-b2bb9b00-cb12-11e9-8eae-1a58367eff39.png)
