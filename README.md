# Guide Dog Classifier

Classify guide dogs vs non-guide dogs using EfficientNet-B3 and transfer learning. Optimized for GPUs with 4GB VRAM (e.g., RTX 3050 Ti).

---

## Overview

- Dataset: Subset of Stanford Dogs (~2,800 images)  
- Model: EfficientNet-B3 (pretrained)  
- Goal: 80–85% accuracy (up to 90% with fine-tuning)  
- Focus: Memory-efficient training on low-end GPUs  

---

## Dataset Structure
```bash
dataset/
├── train/
│ ├── guide_dogs/
│ └── non_guide_dogs/
├── val/
│ ├── guide_dogs/
│ └── non_guide_dogs/
└── test/
├── guide_dogs/
└── non_guide_dogs/
```

---

## Notebooks

- `0_Introduction_Setup.ipynb`: Setup, dataset, model intro  
- `1_Model_Building_Frozen_Layers.ipynb`: Frozen model + classifier  
- `2_Fine_Tuning_No_Optimization.ipynb`: Fine-tuning (OOM on 4GB GPU)  
- `3_Fine_Tuning_Memory_Optimization.ipynb`: Memory optimization techniques  

---

## Requirements

- Python 3.8+  
- PyTorch 1.12+  
- CUDA 11.3+  
- GPU with 4GB+ VRAM  

---

## Setup

Download and extract the dataset into a folder named `dataset` with the structure shown above.

---

