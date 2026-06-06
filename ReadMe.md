# Diabetes Classification ANN Project
**Date Started**: 2026-06-06

This is the append-only development ledger for the Pima Indians Diabetes classification project. All configurations, execution steps, and decisions will be documented chronologically.

## Project Setup & Usage Instructions

### Dataset
**Source**: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
Download `diabetes.csv` from the link above and place it in the root directory of this project before running the scripts.

### Requirements
This project requires Python 3.10+ and the following libraries:
```bash
pip install pandas numpy scikit-learn tensorflow streamlit
```

### Execution Order
Navigate to the root directory and execute the scripts in the following order:

1. **Data Preprocessing**:
   ```bash
   python src/preprocess.py
   ```
   *(Processes `diabetes.csv` and saves arrays to `data/processed/` and the scaler to `models/`)*

2. **Model Architecture**:
   ```bash
   python src/model_architecture.py
   ```
   *(Defines the ANN structure and saves `model_architecture.json` to `models/`)*

3. **Training & Evaluation**:
   ```bash
   python src/train_evaluate.py
   ```
   *(Trains the model on CPU, evaluates it, and saves the trained weights `trained_diabetes_model.keras` and history to `models/`)*

4. **Streamlit Deployment**:
   ```bash
   streamlit run app.py
   ```
   *(Launches the interactive web application)*

---

## Phase 1: Setup & Planning

### Project Objectives
The objective of this project is to build, train, evaluate, and deploy an Artificial Neural Network (ANN) to predict the binary categorical target variable `Outcome` (Diabetes: 1, No Diabetes: 0) using the Pima Indians Diabetes Database.

### Dataset Columns
The dataset contains the following columns:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

### Hardware Constraints
- The model must strictly execute on the CPU (Intel i5).
- The use of the NVIDIA Quadro P620 GPU is explicitly disabled for TensorFlow/Keras to avoid PCIe transfer time bottlenecks, as this is a lightweight tabular dataset.

---

## Phase 2: Data Preprocessing

### Exploratory Data Analysis & Preprocessing Metrics
- **Original Dataset Shape**: (768, 9)

### Zero Value Handling
Replaced invalid zero values with column medians:
- **Glucose**: 5 zero values replaced
- **BloodPressure**: 35 zero values replaced
- **SkinThickness**: 227 zero values replaced
- **Insulin**: 374 zero values replaced
- **BMI**: 11 zero values replaced

### Class Distribution
- **Class 0 (No Diabetes)**: 500
- **Class 1 (Diabetes)**: 268

### Data Splitting & Scaling
- The feature matrix (X) was scaled using `StandardScaler`.
- The dataset was split using an 80/20 train/test split (stratified by Outcome).
- **Train set shape (X, y)**: (614, 8), (614,)
- **Test set shape (X, y)**: (154, 8), (154,)

### Artifacts Saved
- `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`
- `scaler.pkl`

### Phase 2.5: Workspace Restructuring
*Note: To adhere to clean software architecture, the artifacts were moved into structured directories:*
- **Scripts**: `src/` (contains `preprocess.py` and other future execution scripts)
- **Data Arrays**: `data/processed/` (contains all `.npy` matrix splits)
- **Models/Scalers**: `models/` (contains `scaler.pkl` and future model weights)

---

## Phase 3: Model Architecture

### Architecture Details
- **Type**: Sequential Artificial Neural Network
- **Input Layer**: 8 features
- **Hidden Layer 1**: 16 units, `relu` activation
- **Hidden Layer 2**: 8 units, `relu` activation
- **Output Layer**: 1 unit, `sigmoid` activation (binary classification)

### Compilation & Hardware Optimization
- **Loss Function**: `binary_crossentropy`
- **Optimizer**: `adam`
- **Hardware constraints enforced**: GPU explicitly disabled. TensorFlow executed exclusively on CPU.

### Model Summary
```text
Model: "sequential"
+--------------------------------------------------------------------------+
| Layer (type)                    | Output Shape           |       Param # |
|---------------------------------+------------------------+---------------|
| hidden_1 (Dense)                | (None, 16)             |           144 |
|---------------------------------+------------------------+---------------|
| hidden_2 (Dense)                | (None, 8)              |           136 |
|---------------------------------+------------------------+---------------|
| output_layer (Dense)            | (None, 1)              |             9 |
+--------------------------------------------------------------------------+
```
- **Exact Trainable Parameter Count**: 289

### Artifacts Saved
- `models/model_architecture.json`

---

## Phase 4: Training & Evaluation

### Hyperparameters
- **Batch Size**: 32
- **Validation Split**: 0.2 (20%)
- **Epochs Configured**: 100
- **Epochs Run**: 45 (Early Stopping triggered on `val_loss` with patience of 15)

### Training Metrics
**Epoch 1**:
- `loss`: 0.7258
- `accuracy`: 0.4562
- `val_loss`: 0.6721
- `val_accuracy`: 0.6098

**Final Epoch (Epoch 45)**:
- `loss`: 0.4246
- `accuracy`: 0.7882
- `val_loss`: 0.4393
- `val_accuracy`: 0.8130

### Final Test Metrics
Evaluated on the unseen `X_test` data using a 0.5 threshold probability cutoff:
- **Test Accuracy**: 0.7338 (73.38%)

#### Classification Report
- **Precision**: 0.6383
- **Recall**: 0.5556
- **F1-Score**: 0.5941

#### Confusion Matrix
```text
[[83, 17],
 [24, 30]]
```
- **True Negatives (TN)**: 83
- **False Positives (FP)**: 17
- **False Negatives (FN)**: 24
- **True Positives (TP)**: 30

### Artifacts Saved
- **Trained Model**: `models/trained_diabetes_model.keras`
- **History**: `models/training_history.json`

---

## Phase 5: Streamlit Deployment

### Application Setup
- **Entry Point**: `app.py` created in the root directory.
- **Hardware Constraints**: Explicitly enforced CPU-only execution inside `app.py` by setting `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` before any TensorFlow or Keras imports.

### UI Layout Architecture
- **Sidebar**: Titled "Patient Medical Data". Contains 8 distinct input fields (combinations of sliders and number inputs) corresponding to the dataset features. Realistic ranges and default values were assigned.
- **Main Area**: Houses the application title, description, and the "Predict" button.
- **Prediction Logic**: Inputs are collected, scaled via `models/scaler.pkl`, and passed to `models/trained_diabetes_model.keras`.
- **Output Display**: Predictions &ge; 0.5 trigger a High Risk `st.error` alert, while probabilities < 0.5 trigger a Low Risk `st.success` alert. Exact probability percentages are displayed.

### Launch Instructions
To launch the deployment server, execute the following terminal command from the project root:
```bash
streamlit run app.py
```

---
