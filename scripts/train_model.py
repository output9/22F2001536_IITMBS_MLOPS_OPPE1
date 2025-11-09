import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score
import os

# Load processed features
df = pd.read_parquet("data/processed/dataset_versions/v1/features_v0_sample_100.parquet")

# Basic preprocessing
df = df.dropna()
if "timestamp" in df.columns:
    df = df.drop(columns=["timestamp"])
if "stock_symbol" in df.columns:
    df = pd.get_dummies(df, columns=["stock_symbol"], drop_first=True)

# Split
y = df["target"]
X = df.drop(columns=["target", "close", "timestamp"], errors="ignore")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
# r2 = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

os.makedirs("outputs", exist_ok=True)
with open("outputs/metrics.txt", "w") as f:
    f.write(f"R2: {r2:.4f}\nMAE: {mae:.4f}\n")

# Save model
joblib.dump(model, "outputs/model_v0.pkl")
print("✅ Model trained and saved. R2:", r2, "MAE:", mae)
