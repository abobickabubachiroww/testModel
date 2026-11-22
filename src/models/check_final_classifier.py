
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. ПУТИ
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "raw" /
    "manufacturing_defect_dataset.csv"
)

MODEL_PATH = (
    ROOT / "src" / "models" /
    "saved" / "qualitypredict_final.joblib"
)

REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. ЗАГРУЗКА МОДЕЛИ
# ==========================================

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

print("Модель:", artifact["model_name"])
print("Порог:", threshold)


# ==========================================
# 3. ТЕСТОВАЯ ВЫБОРКА
# ==========================================

df = pd.read_csv(DATA_PATH)

X = df[features]
y = df["DefectStatus"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. ПРОГНОЗ
# ==========================================

class_index = list(model.classes_).index(1)

probabilities = model.predict_proba(
    X_test
)[:, class_index]

predictions = (
    probabilities >= threshold
).astype(int)


# ==========================================
# 5. МЕТРИКИ
# ==========================================

metrics = {
    "Accuracy": accuracy_score(
        y_test, predictions
    ),
    "Precision": precision_score(
        y_test, predictions,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test, predictions
    ),
    "F1": f1_score(
        y_test, predictions
    ),
    "ROC_AUC": roc_auc_score(
        y_test, probabilities
    ),
    "BalancedAccuracy": balanced_accuracy_score(
        y_test, predictions
    )
}

print("\nРЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nМАТРИЦА ОШИБОК")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )
)

print("\nОТЧЁТ КЛАССИФИКАЦИИ")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Низкая дефектность",
            "Высокая дефектность"
        ],
        zero_division=0
    )
)


# ==========================================
# 6. СОХРАНЕНИЕ
# ==========================================

pd.DataFrame([metrics]).to_csv(
    REPORTS / "final_classifier_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nПроверка завершена!")
