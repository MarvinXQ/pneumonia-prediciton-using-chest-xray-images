# Pneumonia Detection from Chest X‑rays using DenseNet121

A deep learning pipeline to detect pneumonia (binary classification: NORMAL vs PNEUMONIA) from chest X‑ray images.  
Built with **TensorFlow / Keras** and trained on the [Chest X‑Ray Pneumonia Dataset](https://www.kaggle.com/datasets/divyam6969/chest-xray-pneumonia-dataset) (divyam6969).

---

## 📌 Overview

- **Task:** Binary classification of chest X‑rays into **NORMAL** or **PNEUMONIA** (bacterial + viral combined).
- **Model:** DenseNet121 pretrained on ImageNet, with a custom classification head.
- **Key features:**
  - Class‑weighted loss to handle imbalance (NORMAL samples are fewer).
  - Two‑stage training: frozen base → fine‑tuning of top 50 layers.
  - Real‑time data augmentation.
  - Comprehensive evaluation: accuracy, precision, recall, F1‑score, AUC‑ROC, confusion matrix.
  - **Grad‑CAM** visualisation to highlight regions the model focuses on.
- **Performance on test set (40 images):**
  - Accuracy: **93%**
  - Pneumonia recall: **93%**
  - Normal recall: **90%**
  - AUC (validation): **~0.99**

---

## 📁 Dataset

**Source:** [Kaggle – Chest X‑Ray Pneumonia Dataset by divyam6969](https://www.kaggle.com/datasets/divyam6969/chest-xray-pneumonia-dataset)

**Structure used by this code:**
chest-xray-pneumonia-dataset/
├── train/
│ ├── NORMAL/
│ ├── BACTERIAL/
│ └── VIRAL/
└── test/
├── IM-...normal....jpeg
├── person...bacteria....jpeg
└── person...virus....jpeg

- Training images are organised in subfolders (`NORMAL`, `BACTERIAL`, `VIRAL`).  
- Test images are flat (no subfolders); labels are extracted from filenames (`normal`, `bacteria`, `virus`).  
- **BACTERIAL** and **VIRAL** are merged into a single **PNEUMONIA** class for binary classification.

---

## ⚙️ Pipeline Architecture

1. **Data loading & labelling**  
   - `train/` → file paths + binary label (0 = NORMAL, 1 = PNEUMONIA).  
   - `test/` → label inferred via filename (case‑insensitive).  
   - Stratified split: 15% of training data becomes validation set.

2. **Data augmentation** (only for training)  
   - Rotation, shift, shear, zoom, horizontal flip (Keras `ImageDataGenerator`).  
   - All images are resized to **224×224** and preprocessed for DenseNet121.

3. **Model**  
   - Base: `DenseNet121(weights='imagenet', include_top=False)`.  
   - Head: `GlobalAveragePooling2D` → `Dropout(0.5)` → `Dense(256, relu)` → `Dropout(0.3)` → `Dense(1, sigmoid)`.

4. **Training procedure**
   - **Stage 1:** Only the head is trained (base frozen).  
     - Optimiser: Adam (lr=1e‑4).  
     - Class weights computed from the training set.  
     - Early stopping & reduce‑learning‑rate on plateau, monitored by validation AUC.
   - **Stage 2:** Fine‑tuning – unfreeze the top 50 layers of DenseNet121.  
     - Learning rate lowered to 1e‑5.  
     - Same callbacks, class weights retained.

5. **Evaluation**  
   - Classification report, confusion matrix, ROC curve & AUC on the held‑out test set.  
   - Training history plots (loss, accuracy, AUC).

6. **Interpretability**  
   - **Grad‑CAM** heatmaps overlaid on original X‑rays to show where the model looks for pneumonia.

---

## 📊 Results

**Test set (40 images)**  

|             | precision | recall | f1‑score | support |
|-------------|-----------|--------|----------|---------|
| NORMAL      | 0.82      | 0.90   | 0.86     | 10      |
| PNEUMONIA   | 0.97      | 0.93   | 0.95     | 30      |
| **accuracy**|           |        | **0.93** | 40      |

**ROC AUC** (validation): > 0.99

**Grad‑CAM example:**  
The model consistently highlights lung fields and opacities corresponding to infection.

---

## 🚀 How to Run on Kaggle

1. **Open a Kaggle notebook** (Python, GPU enabled).  
2. **Add the dataset:** Search for `divyam6969/chest-xray-pneumonia-dataset` and attach it to the notebook.  
3. **Copy‑paste the entire code** from the notebook file into the Kaggle editor.  
4. **Run all cells.**  
5. The model will be trained, evaluated, and a Grad‑CAM visualisation will appear at the end.

**Requirements (already satisfied in Kaggle):**
- Python 3.x
- TensorFlow ≥ 2.x
- NumPy, Pandas, Matplotlib, Seaborn, Scikit‑learn, OpenCV‑python

---

## 📂 Repository Structure (if using locally)
├── README.md
├── pneumonia_detection.ipynb # Complete notebook
└── requirements.txt # pip freeze > requirements.txt


**Local setup** (if not using Kaggle):

```bash
git clone <your-repo-url>
cd pneumonia-detection
pip install -r requirements.txt


