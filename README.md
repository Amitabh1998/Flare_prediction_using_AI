# Sleep Flare Predictor

A machine learning application for predicting flare-ups in chronic conditions using sleep and symptom data.

---

## Overview

The **Sleep Flare Predictor** is a Python-based application designed to predict flare-ups in chronic conditions using sleep and symptom data. It leverages an ensemble of models (Neural Network, Random Forest, XGBoost, Logistic Regression) and provides an intuitive **Streamlit** interface for data exploration, visualization, and prediction.

### Key Features
- **Data Exploration**: View dataset (~252 rows, 15 columns), statistics, and flare distribution.
- **Visualizations**: Correlation heatmap, pain trends, and average metrics by flare status.
- **Prediction**: Input sleep and symptom data to predict flare likelihood with probability outputs.
- **Models**: Neural Network, Random Forest, XGBoost, and Logistic Regression, with the best model selected based on average recall.

### Dataset
- **Size**: ~252 rows (54% no-flare, 46% flare)
- **Features**: `pain`, `fatigue`, `sleep_hours`, `sleep_efficiency`, `pain_fatigue_interaction`, `mood_pain_interaction`, `pain_rolling_mean`

### Performance
- **Accuracy**: ~49-55%
- **XGBoost**: 52.94% accuracy, 56.85% average recall (75% flare recall, 38.71% no-flare recall)
- **Value**: High flare recall is critical for early detection in chronic condition management.

---

## Project Structure

```bash
Flare_prediction_project/
├── app/
│   ├── data.py          # Data loading and preprocessing
│   ├── model.py         # Model definitions and training logic
│   ├── main.py          # Streamlit app for visualization and prediction
│   ├── utils.py         # Utility functions for normalization and augmentation
├── data/
│   ├── data.csv         # Dataset (~252 rows, 12 raw + 3 derived features)
├── models/
│   ├── flare_model_v3.pt # Trained model file
├── tests/
│   ├── test_data.py     # Tests for data loading and splits
│   ├── test_model.py    # Tests for model inference
├── README.md            # Project documentation
```

---

## Features

### Data Exploration
- Displays dataset, descriptive statistics, and flare distribution.

### Visualizations
- Correlation heatmap
- Pain trends by subject
- Average metrics by flare status

### Prediction
- Input sleep and symptom data to predict flare likelihood with probability outputs.

### Models
- Neural Network, Random Forest, XGBoost, and Logistic Regression
- Best model selected based on average recall

### Features Used
- `pain`, `fatigue`, `sleep_hours`, `sleep_efficiency`, `pain_fatigue_interaction`, `mood_pain_interaction`, `pain_rolling_mean`

---

## Setup

### Prerequisites
- Python 3.11.0
- Conda environment

### Dependencies
```bash
scikit-learn>=1.5.0
xgboost>=2.1.0
torch>=2.0.0
pandas>=2.0.0
streamlit>=1.25.0
plotly>=5.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.2.0
pytest>=7.4.0
imblearn>=0.12.0
```

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd Flare_prediction_project
   ```

2. Create and activate a Conda environment:
   ```bash
   conda create -n flare_prediction python=3.11.0
   conda activate flare_prediction
   ```

3. Install dependencies:
   ```bash
   pip install scikit-learn>=1.5.0 xgboost>=2.1.0 torch>=2.0.0 pandas>=2.0.0 streamlit>=1.25.0 plotly>=5.10.0 matplotlib>=3.7.0 seaborn>=0.12.0 joblib>=1.2.0 pytest>=7.4.0 imblearn>=0.12.0
   ```

4. Set `PYTHONPATH`:
   ```bash
   export PYTHONPATH=\$PYTHONPATH:/path/to/Flare_prediction_project
   ```

---

## Execution

### Running Tests
Verify data and model integrity:
```bash
pytest tests/test_data.py tests/test_model.py -v
```

**Expected Output:**
```
============================= test session starts ==============================
platform darwin -- Python 3.11.0, pytest-8.4.2, pluggy-1.6.0
collecting ...
collected 3 items

tests/test_data.py::test_load_data PASSED [ 33%]
tests/test_data.py::test_data_splits PASSED [ 66%]
tests/test_model.py::test_model_inference PASSED [100%]

============================== 3 passed in 15-25s ==============================
```

### Running the Streamlit App
1. Remove old model to retrain:
   ```bash
   rm -f models/flare_model_v3.pt
   ```

2. Launch the app:
   ```bash
   streamlit run app/main.py
   ```

3. Access at [http://localhost:8503](http://localhost:8503) with three tabs:
   - **Data Exploration**: View dataset, statistics, flare distribution, and model evaluation report.
   - **Visualizations**: Correlation heatmap, pain trends, and average metrics by flare.
   - **Predict Flare**: Input symptom and sleep metrics for predictions.

---

## Prediction Example

**Input:**
- `pain=9`, `fatigue=9`, `mood=2`, `total_sleep=5.5`, `wake=18.5`, `sleep_hours=0.3`, `pain_rolling_mean=8.0`
- **Computed:** `sleep_efficiency ≈ 0.229`, `pain_fatigue_interaction = 81`, `mood_pain_interaction = 18`

**Expected Output:**
- "Flare Likely" with ~80-90% probability.

**Test Variability:**
- Try `pain=2`, `fatigue=2`, `mood=8`, `total_sleep=8.0`, `wake=16.0`, `sleep_hours=0.8`, `pain_rolling_mean=2.0` for "No Flare Expected".

---

## Checking Correlations

Verify feature correlations with `flare_next_day`:
```python
import pandas as pd
df = pd.read_csv('data/data.csv')
df['sleep_efficiency'] = df['total_sleep'] / (df['total_sleep'] + df['wake']).replace(0, 1e-10)
df['pain_fatigue_interaction'] = df['pain'] * df['fatigue']
df['mood_pain_interaction'] = df['mood'] * df['pain']
df['pain_rolling_mean'] = df.groupby('subject')['pain'].transform(lambda x: x.rolling(window=3, min_periods=1).mean()).fillna(df['pain'])
features = ['pain', 'fatigue', 'sleep_hours', 'sleep_efficiency', 'pain_fatigue_interaction', 'mood_pain_interaction', 'pain_rolling_mean', 'flare_next_day']
print(df[features].corr()['flare_next_day'].sort_values())
```

**Example Output:**
```
pain                  -0.115746
sleep_hours            -0.003796
sleep_efficiency       -0.050000
fatigue                 0.058749
mood_pain_interaction   0.100000
pain_rolling_mean      -0.120000
pain_fatigue_interaction 0.150000
flare_next_day         1.000000
```

---

## Final Performance Results

| Model               | Accuracy | Recall (No Flare) | Recall (Flare) |
|---------------------|----------|-------------------|----------------|
| Neural Network      | 49.02%   | 48.39%            | 50.00%         |
| Random Forest       | 54.90%   | 48.39%            | 65.00%         |
| XGBoost             | 52.94%   | 38.71%            | 75.00%         |
| Logistic Regression  | 49.02%   | 51.61%            | 45.00%         |

**Best Model: XGBoost**
- **Accuracy:** 52.94%
- **Average Recall:** 56.85% (No Flare: 38.71%, Flare: 75.00%)

**Test Set:** ~51 samples (20% of 252), with XGBoost correctly predicting ~27 samples.

---

## Feature Importance

| Feature                | Random Forest | XGBoost |
|------------------------|---------------|---------|
| sleep_hours            | 0.1788        | 0.1348  |
| pain_rolling_mean      | 0.1728        | 0.1716  |
| mood_pain_interaction  | 0.1646        | 0.1517  |
| sleep_efficiency       | 0.1518        | 0.1379  |
| pain_fatigue_interaction | 0.1263      | 0.1304  |
| pain                   | 0.1050        | 0.1498  |
| fatigue                | 0.1006        | 0.1238  |

---

## Analysis

- **Accuracy**: ~49-55% (modest, slightly above random guessing for a balanced dataset).
- **Recall**: Critical in medical applications; XGBoost’s 75% flare recall is valuable for early detection.
- **Feature Importance**: `pain_rolling_mean`, `mood_pain_interaction`, and `sleep_hours` are key predictors.

---

## Pros and Benefits

- **High Flare Recall (XGBoost: 75%)**: Enables proactive interventions.
- **Feature Importance Insights**: Guides future feature engineering.
- **User-Friendly Interface**: Streamlit app for patients and clinicians.
- **Robust Pipeline**: Ensures data quality and model robustness.
- **Extensible Codebase**: Modular design for future enhancements.

---

## Limitations

- **Low Accuracy (~49-55%)**: Below the ~75-90% target.
- **Low No-Flare Recall (38.71%)**: Poor performance on non-flare cases.
- **Weak Feature Correlations**: Correlations with `flare_next_day` are <0.2.
- **Small Dataset**: ~252 rows limit generalization.
- **Feature Importance Gaps**: Some features contribute minimally.

---

## Issues Encountered

- **Test Failures**: Fixed by aligning feature lists.
- **Streamlit Error**: Corrected syntax for `use_container_width`.
- **Low Model Performance**: Improved with derived features and model tuning.

---

## How the Project Worked Out

- **Successes**: 75% flare recall, user-friendly tool, feature insights, extensible codebase.
- **Limitations**: Low accuracy and no-flare recall due to weak correlations and small dataset.

---

## Importance of Recall

- **Why Recall Matters**: Prioritizes detecting flares to avoid missing critical health events.
- **Benefits**: Patient safety, clinical utility, decision support.

---

## Lessons Learned

- **Recall’s Priority**: High flare recall is more critical than accuracy.
- **Feature Importance**: Guides development.
- **Small Datasets**: Limit performance.
- **Iterative Debugging**: Essential for resolving issues.
- **Model Tuning**: Requires balance.
- **User-Centric Design**: Enhances impact.

---

## Future Improvements

- **Additional Features**: Add `fatigue_rolling_mean` or `sleep_efficiency_rolling_mean`.
- **Time-Series Models**: Implement LSTM or Transformer.
- **Data Expansion**: Collect more data.
- **Model Enhancements**: Add regularization, experiment with ensemble methods.
- **User Experience**: Add input validation, visualize feature importances.

---

## Contributors

- **Built by**: [Amitabh Das](https://github.com/amitabh1998)

---

## License

- **MIT License**: See [LICENSE](LICENSE) for details.
