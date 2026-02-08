# A-Novel-Baseline-Aware-Dual-Stream-ConvNeXt-Transformer-with-Differential-Feature-Fusion

This repository presents a **novel baseline-aware dual-stream ConvNeXt–Transformer architecture with differential feature fusion**, designed to achieve improved classification performance. The project includes extensive comparisons with multiple baseline models and robust evaluation using train–validation–test splits and 10-fold cross-validation.

---

## 📌 Project Overview

The objectives of this work are:
- To propose a **hybrid deep learning model** combining ConvNeXt and Transformer architectures
- To introduce **differential feature fusion** for enhanced discriminative learning
- To perform **baseline-aware evaluation** against classical and deep learning models
- To validate model robustness using **10-fold cross-validation**

---

## 🧠 Proposed Model

The proposed architecture features:
- A **dual-stream design**
- **ConvNeXt** for spatial feature extraction
- **Transformer** modules for capturing long-range dependencies
- **Differential feature fusion** to effectively combine learned representations

---

## 📂 Repository Structure


## 📘 Notebook Descriptions

### 1️⃣ Proposed Model (Train-Validation-Test comparison).ipynb
- Implements the **proposed dual-stream ConvNeXt–Transformer model**
- Evaluates performance on:
  - Training set
  - Validation set
  - Test set
- Provides comparative analysis across all three phases

---

### 2️⃣ Proposed Model 10 Fold CV.ipynb
- Performs **10-fold cross-validation** on the proposed model
- Demonstrates robustness and generalization capability
- Reports averaged performance metrics across folds

---

### 3️⃣ Baseline Models Comparison (Accuracy, Precision, Recall, F1 score, Kappa).ipynb
- Implements and evaluates the following **baseline models**:
  - SVM
  - CNN
  - CNN-BiLSTM
  - CNN-RNN
  - MLP
  - Capsule Network (CapsNet)
  - Cascade Forest
- Compares models using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Cohen’s Kappa

---

### 4️⃣ Baseline Models 10 Fold CV comparison.ipynb
- Conducts **10-fold cross-validation** for all baseline models
- Compares baseline models based on **average accuracy**
- Ensures fair and reliable evaluation

---

## 📊 Evaluation Metrics

The following metrics are used throughout the experiments:
- Accuracy
- Precision
- Recall
- F1-Score
- Cohen’s Kappa

---

## 🛠️ Technologies & Libraries

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- TensorFlow / PyTorch  
- Jupyter Notebook  
