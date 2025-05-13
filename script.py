import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load data with semicolon separator
df = pd.read_csv(os.path.join(current_dir, "cardio_train.csv"), sep=';')
# print("Columns in the dataset", df.columns.tolist())

# Drop 'id' column if it exists
if 'id' in df.columns:
    df.drop(columns='id', inplace=True)

# Convert age from days to years
df['age'] = (df['age'] / 365).astype(int)

# Remove duplicates
df = df.drop_duplicates()

# Remove invalid/outlier values
df = df[
    (df['ap_hi'] > 50) & (df['ap_hi'] < 250) &
    (df['ap_lo'] > 30) & (df['ap_lo'] < 200) &
    (df['height'] > 100) & (df['height'] < 220) &
    (df['weight'] > 30) & (df['weight'] < 200)
]

# Features and target
X = df.drop(columns='cardio')
y = df['cardio']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train and evaluate
results = {}
for name, model in models.items():
    if name in ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

# Print results
print("Model Accuracies:")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")
