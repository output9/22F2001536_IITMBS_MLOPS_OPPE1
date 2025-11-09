import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

FEATS_PATH = "data/processed/dataset_versions/v1/features_v0_sample_100.parquet"
MODEL_PATH = "outputs/model_v0_cls.pkl"
OUT_DIR = "outputs"
METRICS_JSON = os.path.join(OUT_DIR, "metrics.json")
METRICS_EVAL_TXT = os.path.join(OUT_DIR, "metrics_eval.txt")
CM_PNG = os.path.join(OUT_DIR, "confusion_matrix.png")

def load_xy():
    df = pd.read_parquet(FEATS_PATH)
    X = df[[
        "open_price", "high_price", "low_price", "close_price", "volume",
        "rolling_avg_10", "volume_sum_10", "stock_symbol"
    ]].copy()
    y = df["target"].astype(int)
    X = pd.get_dummies(X, columns=["stock_symbol"], drop_first=True)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_xy()

    clf = joblib.load(MODEL_PATH)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Save metrics
    with open(METRICS_EVAL_TXT, "w") as f:
        f.write(f"accuracy: {acc:.4f}\n")
        f.write(f"f1: {f1:.4f}\n")
        f.write("\nClassification report:\n")
        f.write(classification_report(y_test, y_pred))

    with open(METRICS_JSON, "w") as jf:
        json.dump({"accuracy": acc, "f1": f1, "confusion_matrix": cm.tolist()}, jf, indent=2)

    # Save confusion matrix image (no seaborn)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in zip([(i, j) for i in range(cm.shape[0]) for j in range(cm.shape[1])], cm.flatten()):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(CM_PNG)

    print("✅ Evaluation complete")
    print("   accuracy:", round(acc, 4), "| f1:", round(f1, 4))
    print("   metrics ->", METRICS_JSON)
    print("   report  ->", METRICS_EVAL_TXT)
    print("   plot    ->", CM_PNG)

if __name__ == "__main__":
    main()
