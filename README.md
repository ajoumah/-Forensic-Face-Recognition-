#   Forensic Face Recognition for Postmortem Identification

##   Overview

Identifying deceased individuals in forensic and humanitarian scenarios remains a major challenge due to face deformation, decomposition, pose variation, and image quality degradation. 

This repository presents a **robust forensic face recognition framework** designed to handle both **antemortem (AM)** and **postmortem (PM)** face images under real-world conditions.

The system integrates:
- Dual face detection strategies
- Face orientation correction
- Deep learning-based embedding extraction
- Multi-model evaluation and benchmarking

---

##   Abstract

Identifying deceased individuals in forensic and humanitarian circumstances is still difficult, owing to face damage, decomposition, and image disparities. This study proposes a strong face identification framework that incorporates dual-method face detection, orientation correction, and deep learning-based embedding creation. Five models—**AdaFace-ViT, FaceNet, ResNet50, SENet, and VGG16**—were evaluated using antemortem (AM) and postmortem (PM) facial images to assess robustness under degraded conditions.

---

##  Key Contributions

-   Dual face detection pipeline (RetinaFace + MTCNN / alternative fusion)
-   Automatic face alignment using landmark-based rotation correction
-   Multi-backbone embedding framework (CNN + Transformer models)
-   Comprehensive comparison of AM vs PM facial recognition performance
-   Evaluation using:
  - Top-K accuracy
  - ROC curve (AUC)
  - EER (Equal Error Rate)
  - Embedding distribution analysis

---

##   System Pipeline

The proposed framework consists of the following stages:

1. **Data Preprocessing**
   - Image filtering and normalization
   - Dataset organization (AM / PM separation)

2. **Face Detection**
   - RetinaFace-based detection
   - MTCNN-based verification
   - OpenCV / Dlib comparison (optional benchmarking)

3. **Face Alignment**
   - Eye-nose landmark geometry correction
   - Orientation normalization (-90°, 90°, 180° handling)

4. **Embedding Extraction**
   - AdaFace ViT
   - FaceNet
   - ResNet50
   - SENet
   - VGG16

5. **Matching & Retrieval**
   - FAISS-based similarity search
   - Cosine similarity scoring

6. **Evaluation**
   - Top-K accuracy (K=1, 5, 10)
   - Verification ROC (1:1 matching)
   - Embedding transformation analysis

---

##  Evaluated Models

| Model        | Type        | Description |
|--------------|------------|-------------|
| ViT          | Transformer | State-of-the-art face recognition with adaptive margin loss |
| FaceNet      | CNN         | Triplet-loss based embedding model |
| ResNet50     | CNN         | Deep residual network backbone |
| SENet        | CNN         | Channel attention-enhanced CNN |
| VGG16        | CNN         | Classical deep CNN architecture |

---

## 📁 Repository Structure
