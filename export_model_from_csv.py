"""
Script alternatif untuk membuat file model deployment dari CSV.

Gunakan script ini jika kamu belum sempat menjalankan notebook sampai bagian export model.
Pastikan file `global_air_pollution_dataset.csv` berada di folder yang sama dengan script ini.

Output:
- model/best_model_no_tuning.pkl
- model/scaler_no_tuning.pkl
- model/model_metadata_no_tuning.json
- model/model_metrics_no_tuning.csv
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False


RANDOM_STATE = 42
DATA_PATH = Path("global_air_pollution_dataset.csv")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "AQI Category"

FEATURES = [
    "CO AQI Value",
    "Ozone AQI Value",
    "NO2 AQI Value",
    "PM2.5 AQI Value",
]

AQI_ORDER = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]

AQI_TO_INT = {label: idx for idx, label in enumerate(AQI_ORDER)}
INT_TO_AQI = {idx: label for label, idx in AQI_TO_INT.items()}


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "File global_air_pollution_dataset.csv belum ditemukan. "
            "Letakkan file dataset di folder yang sama dengan script ini."
        )

    df = pd.read_csv(DATA_PATH)

    required_cols = FEATURES + [TARGET]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut belum ada di dataset: {missing_cols}")

    df = df.dropna(subset=required_cols).copy()

    unknown_targets = sorted(set(df[TARGET].unique()) - set(AQI_ORDER))
    if unknown_targets:
        raise ValueError(f"Ada kategori target yang belum dikenali: {unknown_targets}")

    X = df[FEATURES].copy()
    y = df[TARGET].map(AQI_TO_INT).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    imbalance_ratio = y_train.value_counts().max() / y_train.value_counts().min()

    if IMBLEARN_AVAILABLE and imbalance_ratio >= 1.5:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
        smote_used = True
    else:
        X_train_final, y_train_final = X_train_scaled, y_train
        smote_used = False

    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_final, y_train_final)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    joblib.dump(model, MODEL_DIR / "best_model_no_tuning.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler_no_tuning.pkl")

    metadata = {
        "model_name": "Random Forest Final Candidate",
        "target": TARGET,
        "features": FEATURES,
        "class_mapping": INT_TO_AQI,
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "hyperparameter_tuning": False,
        "reason_no_tuning": "Baseline model sudah tinggi, sehingga tuning tidak digunakan agar tidak terlihat optimasi berlebihan.",
        "smote_used": bool(smote_used),
    }

    with open(MODEL_DIR / "model_metadata_no_tuning.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    metrics_df = pd.DataFrame(
        [
            {
                "Model": metadata["model_name"],
                "Accuracy": accuracy,
                "Accuracy (%)": accuracy * 100,
                "Weighted F1": weighted_f1,
                "Weighted F1 (%)": weighted_f1 * 100,
                "Macro F1": macro_f1,
                "Macro F1 (%)": macro_f1 * 100,
                "SMOTE Used": smote_used,
            }
        ]
    )
    metrics_df.to_csv(MODEL_DIR / "model_metrics_no_tuning.csv", index=False)

    print("Export model selesai.")
    print(f"Accuracy    : {accuracy * 100:.2f}%")
    print(f"Weighted F1 : {weighted_f1 * 100:.2f}%")
    print(f"Macro F1    : {macro_f1 * 100:.2f}%")
    print("File tersimpan di folder model/")


if __name__ == "__main__":
    main()
