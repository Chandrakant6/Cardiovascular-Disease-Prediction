# Heart Disease Predictor

A machine learning project that predicts the presence of heart disease using various classification models.

## Features

- Multiple model support:
  - Random Forest
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
- Automatic model selection based on accuracy
- Comprehensive data preprocessing:
  - Age conversion (days to years)
  - BMI calculation
  - Outlier detection and handling using IQR method
  - Feature scaling using StandardScaler
- Model persistence (save/load functionality)

## Requirements

- Python 3.7+
- Required packages (see requirements.txt)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset file (cardio_train.csv) in the project directory
2. Run the predictor:
```bash
python heart_disease_predictor.py
```

The script will:
- Load and preprocess the data
- Train multiple models
- Select the best performing model
- Save the best model for future use

## Data Format

The input data should be a CSV file with the following columns:
- age: age in days
- height: height in cm
- weight: weight in kg
- ap_hi: systolic blood pressure
- ap_lo: diastolic blood pressure
- cardio: target variable (1: presence of heart disease, 0: absence)

## Data Preprocessing

The predictor performs the following preprocessing steps:
1. Age conversion from days to years
2. BMI calculation from height and weight
3. Outlier detection and handling using IQR method
4. Feature scaling using StandardScaler

## Model Information

The predictor uses multiple models and automatically selects the best performing one based on accuracy. The models are:

1. Random Forest: An ensemble of decision trees with default parameters
2. Logistic Regression: A linear model for binary classification
3. Decision Tree: A simple tree-based model
4. KNN: K-Nearest Neighbors classifier (k=5)
5. SVM: Support Vector Machine with default parameters

## Output

The script provides:
- Training progress for each model
- Accuracy scores
- Best model selection
- Saved model files in the 'saved_models' directory

## License

MIT License 
