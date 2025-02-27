# Generalized Abnormality Classification in VCE Frames  
### Using Vision Transformer and Contrastive Learning  

## Overview  
This project presents a novel approach for automated abnormality classification in Video Capsule Endoscopy (VCE) frames using Vision Transformers (ViT) and Contrastive Learning. Our method addresses challenges such as data imbalance, device heterogeneity, and limited labeled samples, achieving state-of-the-art performance with 96.56% accuracy and 0.9654 macro F1-score.  

## Motivation  
Manual analysis of VCE recordings is time-intensive, generating over 50,000 frames per examination. This project automates abnormality classification, improving diagnostic speed and accuracy while reducing healthcare costs.  

## Challenges Addressed  
- **Data Imbalance:** Rare abnormalities are underrepresented.  
- **Device Heterogeneity:** Variations in image quality across different VCE manufacturers.  
- **Limited Labeled Data:** High cost and time required for expert labeling.  

## Proposed Solution  
Our approach combines:  
1. **Self-Supervised Contrastive Learning:** Pre-training to learn robust feature representations from unlabeled data.  
2. **Vision Transformer Architecture:** Captures long-range dependencies and subtle visual patterns using attention mechanisms.  
3. **Two-Phase Training Strategy:**  
    - **Phase 1 - Contrastive Pre-training:** Learning general visual features from unlabeled VCE images.  
    - **Phase 2 - Supervised Fine-tuning:** Specializing the model for multi-class abnormality classification.  

## Key Features  
- **Self-supervised learning** for data efficiency and class imbalance mitigation.  
- **Transformer architecture** for improved feature extraction and adaptability to device heterogeneity.  
- **Two-phase training strategy** enabling efficient learning with limited labeled data.  

## Architecture  
The model is built on the Vision Transformer (ViT) architecture with the following configurations:  
- **Input Size:** 224×224×3  
- **Patch Size:** 8×8 (resulting in 784 patches)  
- **Embedding Dimension:** 384  
- **Layers:** 6  
- **Attention Heads:** 6  
- **Output:** 10 abnormality classes  

## Training Strategy  
1. **Contrastive Pre-training:**  
   - **Loss Function:** InfoNCE with temperature 0.7  
   - **Optimizer:** 8-bit AdamW with learning rate 2e-5  
   - **Batch Size:** 16  
2. **Classification Fine-tuning:**  
   - **Loss Function:** Cross-entropy with label smoothing (ϵ = 0.1)  
   - **Optimizer:** AdamW with learning rate 2e-5 and weight decay 0.01  
   - **Batch Size:** 64  
   - **Epochs:** 50  

## Results  
- **Validation Accuracy:** 96.56%  
- **Macro F1-Score:** 0.9654  
- **ROC AUC:** 0.9961  
- **PR AUC:** 0.9496  

## Comparison with Baselines  
| Model       | Accuracy | F1-Score | Precision | Specificity |  
|-------------|----------|----------|-----------|-------------|  
| Custom CNN  | 0.038    | 0.134    | 0.150     | 0.050       |  
| VGG         | 0.7168   | 0.715    | 0.720     | 0.730       |  
| SVM         | 0.8200   | 0.825    | 0.830     | 0.835       |  
| ResNet50    | 0.7600   | 0.760    | 0.770     | 0.775       |  
| **Ours**    | **0.9656** | **0.965** | **0.970** | **0.970**   |  

## Impact  
- **Efficiency:** Faster diagnosis by automating VCE frame analysis.  
- **Consistency:** Vendor-independent performance for standardized diagnostics.  
- **Accuracy:** Robust detection across a broad spectrum of GI abnormalities.  

## Limitations and Future Work  
- **Computational Cost:** High resource requirements for training.  
- **Real-time Performance:** Needs optimization for deployment in clinical settings.  
- **Multi-frame Analysis:** Potential for enhanced accuracy by incorporating temporal information.  

## Dataset  
The dataset used in this study is sourced from the [Capsule Vision 2024 Challenge](https://figshare.com/articles/dataset/Training_and_Validation_Dataset_of_Capsule_Vision_2024_Challenge/26403469).  

## Code Availability  
The implementation code is available at: [GitHub Repository](https://github.com/Priyanshu-5257/cvc-24-submission)  

## Citation  
If you find this work useful, please cite:  
```
@article{YourPaper2024,
  title={Generalized Abnormality Classification in VCE Frames Using Vision Transformer and Contrastive Learning},
  author={Priyanshu Maurya, Priyanshu Pansari, Adamya Vashistha},
  journal={Capsule Vision 2024 Challenge},
  year={2024}
}
```

## Acknowledgments  We thank the organizers of the Capsule Vision 2024 Challenge for providing the dataset and evaluation platform.  
