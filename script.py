import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess data
def load_and_preprocess_data(file_path, sample_size=10000):
    # Load data
    df = pd.read_csv(file_path, sep=';')
    
    # Sample data
    df = df.sample(n=sample_size, random_state=42)
    
    # Basic preprocessing
    df['age'] = (df['age'] / 365).astype(int)
    df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
    
    # Remove outliers
    df = df[
        (df['ap_hi'] > 50) & (df['ap_hi'] < 250) &
        (df['ap_lo'] > 30) & (df['ap_lo'] < 200) &
        (df['height'] > 100) & (df['height'] < 220) &
        (df['weight'] > 30) & (df['weight'] < 200)
    ]
    
    return df

# Create visualizations
def create_visualizations(df):
    plt.figure(figsize=(15, 10))
    
    # Age Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='age', hue='cardio', bins=30, kde=True)
    plt.title('Age Distribution by Cardiovascular Disease')
    plt.xlabel('Age (years)')
    plt.ylabel('Count')
    
    # Blood Pressure
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='ap_hi', y='ap_lo', hue='cardio', alpha=0.5)
    plt.title('Systolic vs Diastolic Blood Pressure')
    plt.xlabel('Systolic (ap_hi)')
    plt.ylabel('Diastolic (ap_lo)')
    
    # BMI Distribution
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='cardio', y='bmi')
    plt.title('BMI Distribution by Cardiovascular Disease')
    plt.xlabel('Cardiovascular Disease')
    plt.ylabel('BMI')
    
    # Correlation Heatmap
    plt.subplot(2, 2, 4)
    numeric_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
    correlation = df[numeric_cols + ['cardio']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.show()

# Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=5)
    }
    
    # Train and evaluate
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} accuracy: {acc:.4f}")
    
    return results

# Plot model accuracies
def plot_model_accuracies(results):
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.title('Model Accuracies Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess data
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cardio_train.csv")
    df = load_and_preprocess_data(file_path)
    
    # Create visualizations
    create_visualizations(df)
    
    # Prepare data for training
    X = df.drop(columns='cardio')
    y = df['cardio']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Print final results
    print("\nFinal Model Accuracies:")
    for model, acc in results.items():
        print(f"{model}: {acc:.4f}")
    
    # Plot results
    plot_model_accuracies(results)

if __name__ == "__main__":
    main()
