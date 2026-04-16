 ICU-Patient-Deterioration
Machine learning model for ICU patient monitoring

1. Project Title & Problem Statement

Title

Early Prediction of ICU Patient Deterioration Using Machine Learning and Deep Learning

Problem Statement

Intensive Care Unit (ICU) patients are at high risk of sudden clinical deterioration, which can rapidly become life-threatening if not detected in time. Traditional monitoring systems rely on manual observation and threshold-based alerts, which are often reactive rather than proactive. This project predicts whether an ICU patient will experience a deterioration event within the next 12 hours using time-series vitals, laboratory results, and patient demographic data.

By leveraging supervised machine learning and deep learning models, the system aims to provide early, data-driven warnings — enabling clinicians to intervene before a patient's condition worsens, improving patient outcomes and reducing ICU mortality rates.

2. ML Pipeline Overview

The end-to-end pipeline follows a structured data science workflow:

- **Data Ingestion**: Load patients, vitals, labs, and hourly deterioration panel CSV files.
- **Data Merging**: Inner-join vitals + labs + deterioration + patients on `patient_id` and `hour_from_admission`.
- **Data Cleaning**: Drop duplicates, fill missing values with column means, remove `patient_id` from features.
- **Feature Engineering**: Drop redundant target columns and retain `deterioration_next_12h` as the prediction target.
- **Feature Scaling**: Apply `StandardScaler` to numeric features.
- **Train-Test Split**: Perform an 80/20 stratified split with `random_state=42`.
- **Model Training**: Train Logistic Regression, Decision Tree, Random Forest, XGBoost, and ANN (Keras).
- **Evaluation**: Generate classification reports with precision, recall, and F1-score per class.
- **Visualization**: Plot accuracy and F1-score comparisons, and confusion matrices.

**Pipeline Flow**:

Raw CSVs → Merge → Clean → Scale → Split → Train Models → Evaluate → Compare

 3. Dataset Details

3.1 Source

The dataset is a synthetic/simulated ICU patient dataset composed of four inter-related CSV files generated to represent realistic clinical time-series data for hospital deterioration prediction research.

3.2 Dataset Files

- `patients.csv`
  - One row per patient, containing static demographic and admission information.
  - Key columns: `patient_id`, `age`, `gender`, `comorbidity_index`, `admission_type`, `baseline_risk_score`, `los_hours`, `deterioration_event`, `deterioration_hour`.
- `vitals_timeseries.csv`
  - Hourly vital sign readings per patient.
  - Key columns: `patient_id`, `hour_from_admission`, `heart_rate`, `respiratory_rate`, `spo2_pct`, `temperature_c`, `systolic_bp`, `diastolic_bp`, `oxygen_device`, `oxygen_flow`, `mobility_score`, `nurse_alert`.
- `labs_timeseries.csv`
  - Hourly laboratory results per patient.
  - Key columns: `patient_id`, `hour_from_admission`, `wbc_count`, `lactate`, `creatinine`, `crp_level`, `hemoglobin`, `sepsis_risk_score`.
- `hospital_deterioration_hourly_panel.csv`
  - Merged hourly panel combining vitals, labs, and patient info with deterioration labels.
  - Includes `deterioration_next_12h` as the primary target.

3.3 Dataset Statistics (Post-Merge)

- Total Records (after merge): `417,866`
- Total Features (pre-cleaning): `53`
- Final Feature Count (X): `~40` numeric features
- Target Variable: `deterioration_next_12h` (binary)
- Class Distribution:
  - No Deterioration (0): `395,277` records (~94.6%)
  - Deterioration (1): `22,589` records (~5.4%)
- Train Set (80%): `~334,292` records
- Test Set (20%): `83,574` records

3.4 Target Variable

`deterioration_next_12h` is a binary label indicating whether the patient will experience clinical deterioration within the next 12 hours from the current observation hour. The dataset is heavily imbalanced, which motivated using `class_weight='balanced'` for logistic regression and careful model evaluation.

4. Model Details

Five models were trained and evaluated: four classical ML models and one deep learning model.

4.1 Machine Learning Models

| Model | Library | Key Configuration | Accuracy | F1 (Class 1) |
|---|---|---|---|---|
| Logistic Regression | scikit-learn | `class_weight='balanced'` | 72% | 0.22 |
| Decision Tree | scikit-learn | Default parameters | 96% | 0.61 |
| Random Forest | scikit-learn | Default parameters | 97% | 0.68 |
| XGBoost | xgboost | `eval_metric='logloss'` | 98% | 0.73 |

4.2 Deep Learning Model (ANN)

An Artificial Neural Network was built using TensorFlow/Keras:

- Input → Hidden 1: `Dense(32)` with ReLU
- Dropout: `30%`
- Hidden 2: `Dense(16)` with ReLU
- Output: `Dense(1)` with Sigmoid

Training configuration:

- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Epochs: 10
- Batch Size: 32

Performance:

- Training Accuracy: ~96.5%
- ANN Test Accuracy: 97%
- ANN F1 (Class 1): 0.59

4.3 Model Comparison Summary

| Model | Accuracy | Precision (1) | Recall (1) | F1 (1) |
|---|---|---|---|---|
| Logistic Regression | 72% | 0.13 | 0.71 | 0.22 |
| Decision Tree | 96% | 0.61 | 0.62 | 0.61 |
| Random Forest | 97% | 0.93 | 0.54 | 0.68 |
| XGBoost ★ Best | 98% | 0.89 | 0.62 | 0.73 |
| ANN | 97% | 0.87 | 0.44 | 0.59 |

> XGBoost achieved the highest minority-class F1-score of `0.73`, making it the best model for this imbalanced ICU deterioration prediction task.

5. Steps to Run the Project

 5.1 Environment Setup

**Option A — Google Colab (Recommended)**

1. Open Google Colab at `colab.research.google.com`.
2. Upload `MLT.ipynb` to the Colab session.
3. Upload all CSV files to Colab storage.
4. Run all cells from top to bottom (`Runtime → Run All`).

**Option B — Local Machine**

1. Install Python 3.8+ and `pip`.
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost tensorflow matplotlib jupyter
   ```
3. Place all CSV files in the same directory as the notebook.
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook MLT.ipynb
   ```

5.2 Execution Steps

1. Load all CSV files using `pandas.read_csv()`.
2. Merge datasets on `patient_id` and `hour_from_admission`.
3. Clean the data: remove duplicates, fill NaNs, and drop redundant columns.
4. Scale numeric features with `StandardScaler`.
5. Split the data into train/test sets (80/20) and verify class distribution.
6. Train models in order: Logistic Regression → Decision Tree → Random Forest → XGBoost → ANN.
7. Evaluate each model using classification reports.
8. Visualize performance using bar charts and confusion matrices.

6. Required Dependencies & Libraries

| Library | Purpose |
|---|---|
| Python 3.8+ | Core language |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| scikit-learn | ML models, scaling, metrics |
| xgboost | Gradient boosting classifier |
| tensorflow / keras | ANN / deep learning |
| matplotlib | Visualization |
| jupyter | Notebook execution |

Install for local setup:

```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib jupyter
```

For Google Colab, install XGBoost if needed:

```python
!pip install xgboost
```

7. Sample Outputs

7.1 Dataset Shape After Merge

- After vitals + labs: `(417866, 18)`
- After adding deterioration: `(417866, 44)`
- Final merged dataset: `(417866, 53)`

 7.2 Class Distribution

- `0` (No Deterioration): `395,277` (~94.6%)
- `1` (Deterioration): `22,589` (~5.4%)

7.3 XGBoost Classification Report (Best Model)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 | 0.98 | 1.00 | 0.99 | 79,011 |
| 1 | 0.89 | 0.62 | 0.73 | 4,563 |
| Accuracy | — | — | 0.98 | 83,574 |
| Macro Avg | 0.94 | 0.81 | 0.86 | 83,574 |
| Weighted Avg | 0.97 | 0.98 | 0.97 | 83,574 |

7.4 ANN Training Progress

- Epoch 1: accuracy `0.9514`, loss `0.1349`
- Epoch 5: accuracy `0.9644`, loss `0.1071`
- Epoch 10: accuracy `0.9649`, loss `0.1049`

7.5 Random Forest Confusion Matrix

- TN: `78,817`
- FP: `194`
- FN: `2,089`
- TP: `2,474`

7.6 Visualization Charts

- Accuracy comparison bar chart
- F1-score comparison bar chart
- Random Forest confusion matrix heatmap

8. Key Findings & Notes

- **Class Imbalance**: The dataset is heavily imbalanced (~94.6% vs 5.4%).
- **Best Model**: XGBoost provided the strongest minority-class F1-score at `0.73`.
- **Precision vs. Recall**: Random Forest had high precision (`0.93`) but lower recall (`0.54`) for deterioration events.
- **ANN Limitation**: The ANN achieved strong accuracy but lower F1 for the minority class compared to XGBoost.
- **Feature Engineering**: Duplicate columns from merge operations were cleaned by keeping `deterioration_next_12h` as the sole target column.

Project Structure

- `Backend/`
  - `app.py`
  - `database.py`
- `Dataset/`
  - `final_dataset.csv`
  - `patients.csv`
- `frontend/`
  - `login.html`
  - `doctor.html`
  - `admin.html`
  - `manage_users.html`
  - `patient_detail.html`
- `model/`
  - Model files and notebooks
- `UI/`
  - `style.css`


