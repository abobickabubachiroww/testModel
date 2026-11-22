
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix
)


# ==========================================
# 1. ПУТИ
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "raw" /
    "manufacturing_defect_dataset.csv"
)

REPORTS = ROOT / "reports"
SAVED = ROOT / "src" / "models" / "saved"

REPORTS.mkdir(parents=True, exist_ok=True)
SAVED.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. ПРИЗНАКИ
# ==========================================

FEATURES = [
    "ProductionVolume",
    "ProductionCost",
    "SupplierQuality",
    "DeliveryDelay",
    "MaintenanceHours",
    "DowntimePercentage",
    "InventoryTurnover",
    "StockoutRate",
    "WorkerProductivity",
    "SafetyIncidents",
    "EnergyConsumption",
    "EnergyEfficiency",
    "AdditiveProcessTime",
    "AdditiveMaterialCost"
]

TARGET = "DefectStatus"


# ==========================================
# 3. ЗАГРУЗКА ДАННЫХ
# ==========================================

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)

print("Обучающих наблюдений:", len(X_train))
print("Тестовых наблюдений:", len(X_test))
print("Тестовые данные не используются.")


# ==========================================
# 4. МОДЕЛИ
# ==========================================

models = {
    "Logistic Balanced": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    ),

    "RF Balanced": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "RF Balanced Subsample": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),

    "RF Standard": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=8,
        class_weight=None,
        random_state=42,
        n_jobs=-1
    )
}


# ==========================================
# 5. КРОСС-ВАЛИДАЦИЯ
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

best_score = -1
best_name = None
best_model = None
best_threshold = None

thresholds = np.arange(
    0.20,
    0.81,
    0.02
)


for name, model in models.items():

    print("\n" + "=" * 55)
    print("МОДЕЛЬ:", name)
    print("=" * 55)

    # Out-of-fold вероятности:
    # каждое наблюдение оценивается моделью,
    # которая не обучалась на нём.

    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    # Подбор порога по Balanced Accuracy

    threshold_results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = balanced_accuracy_score(
            y_train,
            predictions
        )

        threshold_results.append(
            (threshold, score)
        )

    selected_threshold, selected_score = max(
        threshold_results,
        key=lambda item: item[1]
    )

    predictions = (
        probabilities >= selected_threshold
    ).astype(int)

    metrics = {
        "Model": name,
        "Threshold": round(
            selected_threshold, 3
        ),
        "Accuracy": accuracy_score(
            y_train, predictions
        ),
        "Precision": precision_score(
            y_train, predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_train, predictions
        ),
        "F1": f1_score(
            y_train, predictions
        ),
        "ROC_AUC": roc_auc_score(
            y_train, probabilities
        ),
        "BalancedAccuracy": selected_score
    }

    results.append(metrics)

    print("Выбранный порог:", selected_threshold)
    print("Balanced Accuracy:", round(
        selected_score, 4
    ))
    print("ROC-AUC:", round(
        metrics["ROC_AUC"], 4
    ))

    print("\nМатрица ошибок:")
    print(
        confusion_matrix(
            y_train,
            predictions,
            labels=[0, 1]
        )
    )

    if selected_score > best_score:
        best_score = selected_score
        best_name = name
        best_model = model
        best_threshold = selected_threshold


# ==========================================
# 6. СРАВНЕНИЕ
# ==========================================

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "BalancedAccuracy",
    ascending=False
)

print("\nИТОГОВОЕ СРАВНЕНИЕ")
print(
    comparison.round(4).to_string(
        index=False
    )
)

comparison.to_csv(
    REPORTS / "imbalance_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 7. СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ
# ==========================================

# Обучаем выбранную модель на всей
# обучающей части (2592 наблюдения).

best_model.fit(
    X_train,
    y_train
)

joblib.dump(
    {
        "model": best_model,
        "threshold": float(best_threshold),
        "features": FEATURES,
        "model_name": best_name
    },
    SAVED / "qualitypredict_final.joblib"
)

print("\nЛУЧШАЯ МОДЕЛЬ:", best_name)
print("ЛУЧШИЙ ПОРОГ:", best_threshold)
print("BALANCED ACCURACY:", round(
    best_score, 4
))

print("\nМодель сохранена!")
print("Эксперимент завершён!")
