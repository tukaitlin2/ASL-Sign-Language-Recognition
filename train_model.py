import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import joblib

data = pd.read_csv("asl_data.csv")

print(data.head())
print(data.shape)

X = data.drop("label", axis = 1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42
)

print("\nTraining Model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

joblib.dump(model, "asl_model.pkl")
print("Model saved as asl_model.pkl")
print(data["label"].value_counts().sort_index())