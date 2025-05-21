import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
from joblib import dump, load

# Set random seed for reproducibility
RANDOM_SEED = 42

class HeartDiseasePredictor:
    def __init__(self):
        self.models_dir = 'saved_models'
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        # Initialize basic models
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=RANDOM_SEED),
            'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED),
            'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_SEED),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC()
        }
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the heart disease dataset"""
        print("Loading and preprocessing data...")
        df = pd.read_csv(file_path, sep=';')
        
        # Basic preprocessing
        df['age'] = (df['age'] / 365).astype(int)
        df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
        
        # Handle outliers before scaling
        numerical_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
        
        # Scale numerical features
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df
    
    def train(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        print("Training models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Test accuracy: {accuracy:.4f}")
            
            # Update best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
                print(f"New best model: {name}")
        
        # Print final results
        print("\n" + "=" * 50)
        print(f"Best Model: {self.best_model_name}")
        print(f"Best Accuracy: {self.best_accuracy:.4f}")
        print("=" * 50 + "\n")
        
        # Save best model
        self.save_model()
        
    def save_model(self):
        """Save the best model and scaler"""
        model_path = os.path.join(self.models_dir, 'best_model.joblib')
        scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
        
        dump(self.best_model, model_path)
        dump(self.scaler, scaler_path)
        print(f"Best model and scaler saved to {self.models_dir}")
    
    def load_model(self):
        """Load the saved model and scaler"""
        model_path = os.path.join(self.models_dir, 'best_model.joblib')
        scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or scaler files not found. Please train the model first.")
        
        self.best_model = load(model_path)
        self.scaler = load(scaler_path)
        print("Model and scaler loaded successfully")
    
    def predict(self, features):
        """Make predictions for new cases"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.best_model.predict(features_scaled)
        probability = self.best_model.predict_proba(features_scaled)
        
        return prediction, probability

def main():
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Load and preprocess data
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cardio_train.csv")
    df = predictor.load_and_preprocess_data(file_path)
    
    # Prepare data for training
    X = df.drop(columns=['cardio'])
    y = df['cardio']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    
    # Scale features
    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_test_scaled = predictor.scaler.transform(X_test)
    
    # Train models
    predictor.train(X_train_scaled, X_test_scaled, y_train, y_test)

if __name__ == "__main__":
    main()
