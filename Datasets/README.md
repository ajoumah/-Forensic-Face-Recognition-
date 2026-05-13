\# 📦 Forensic Face Embedding Dataset



\##  Overview



This folder contains \*\*deep feature embeddings\*\* extracted from both:



\- \*\*Antemortem (AM)\*\* facial images

\- \*\*Postmortem (PM)\*\* facial images



for forensic face recognition research.



To preserve privacy and respect ethical considerations surrounding sensitive forensic data, \*\*the original postmortem face images are NOT included in this repository\*\*.



Instead, this dataset provides only:

\- numerical feature embeddings

\- identity pairing information

\- metadata required for reproducible evaluation



\---



\##  Ethical \& Privacy Notice



The postmortem facial images used in this research contain sensitive biometric and forensic content.



For ethical, legal, and humanitarian reasons:



\- ❌ Raw postmortem images are NOT publicly distributed

\- ❌ Facial reconstructions are NOT shared

\- ❌ Personally identifiable information (PII) is NOT included



Only extracted deep embeddings are provided to support:

\- reproducibility

\- benchmarking

\- scientific validation

\- forensic AI research



\---



\##  Dataset Description



The dataset contains embeddings extracted from:



\- \*\*191 paired identities\*\*

\- One antemortem image per identity

\- One corresponding postmortem image per identity



Each identity includes:



| Data Type | Description |

|---|---|

| Identity ID |  

| alive\_filename |  

| dead\_filename |



| Alive Embedding | Feature vector extracted from AM image |

| Dead Embedding | Feature vector extracted from PM image |



\---



\##  Evaluated Models



Embeddings were generated using the following deep learning models:



| Model | Architecture Type |

|---|---|

| ViT | Vision Transformer |

| FaceNet | CNN |

| ResNet50 | CNN |

| VGG16 | CNN |



\---



\## 📁 Dataset Structure



```text

dataset/

│

├── vit/

│   └── embeddings.csv

│

├── facenet/

│   └── embeddings.csv

│

├── resnet50/

│   └── embeddings.csv

│

│

├── vgg16/

│   └── embeddings.csv

│

└── README.md

##  Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{forensic_face_embeddings_2026,
  title={Forensic Face Embedding Dataset for Antemortem and Postmortem Matching},
  author={Ahmad El Jouma},
  year={2026}
}
```
