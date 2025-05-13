# Cardiovascular-Disease-Prediction
## problem Statement
Heart disease, also known as cardiovascular disease, is one of the most serious illnesses in both India and the rest of the globe. According to estimates, cardiac illnesses account for 28.1% of fatalities. More than 17.6 million fatalities, or a large portion of all deaths in 2016, were caused by it in 2016. Therefore, a system that can predict with exact precision and dependability is required for the appropriate and prompt diagnosis as well as the treatment of such diseases. Numerous academics do extensive research utilising a variety of machine learning algorithms to predict heart illness using different datasets that contain numerous factors that lead to heart attacks. Now it is your turn to do a analysis with the given dataset

## Project Output Instructions :
•	Perform data pre-processing operations.
•	As a part of data analysis and visualizations draw all the possible plots to provide essential informations and to derive some meaningful insights.
•	Show your correlation matrix of features according to the datasets.
•	Find out accuracy levels of various machine learning techniques such as Support Vector Machines (SVM), K-Nearest Neighbor (KNN), Decision Trees (DT) , Logistic Regression (LR) and Random Forest (RF).
•	Build your Machine learning model for heart disease detection according to the result.

## script.md description
### Data Loading and Preprocessing (load_and_preprocess_data function):
  - Loads the cardiovascular disease dataset from CSV with semicolon separator
  - Samples 10,000 records randomly (with fixed seed for reproducibility)
  - Performs basic preprocessing:
    - Converts age from days to years
    - Calculates BMI from height and weight
  - Removes outliers for key measurements:
    - Blood pressure (systolic: 50-250, diastolic: 30-200)
    - Height (100-220 cm)
    - Weight (30-200 kg)
### Data Visualization (create_visualizations function):
  - Creates a 2x2 grid of plots:
    - Age distribution by cardiovascular disease status
    - Blood pressure relationship (systolic vs diastolic)
    - BMI distribution by cardiovascular disease status
    - Correlation heatmap of numeric features
### Model Training and Evaluation (train_and_evaluate_models function):
- Implements multiple models:
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
- For each model:
  - Trains on the training data
  - Evaluates accuracy on test data
  - Saves the model to disk
- Tracks and saves the best performing model
### Model Management:
- load_best_model: Loads the best performing model and scaler
- predict_cardiovascular_disease: Makes predictions using the loaded model
- plot_model_accuracies: Visualizes the performance of all models
### Main Workflow (main function):
- Loads and preprocesses the data
- Creates visualizations
- Splits data into training and test sets
- Scales features using StandardScaler
- Trains and evaluates all models
- Saves the scaler and models
- Demonstrates loading and using the best model
### File Management:
- Creates a 'saved_models' directory
- Saves individual models with descriptive names
- Saves the scaler for future use
- Saves information about the best model
### The code is designed to:
- Handle cardiovascular disease prediction
- Provide comprehensive data analysis through visualizations
- Compare multiple machine learning models
- Save and load models for future use
- Maintain reproducibility through fixed random seeds
- Handle data preprocessing and feature scaling
- Provide example predictions
