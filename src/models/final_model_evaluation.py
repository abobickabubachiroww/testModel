
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    brier_score_loss,
    log_loss
)
from sklearn.calibration import calibration_curve
from statsmodels.stats.proportion import proportion_confint


# ==========================================
# 1. ПУТИ К ФАЙЛАМ
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" /
    "manufacturing_defect_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "src" / "models" /
    "saved" / "random_forest_tuned.joblib"
)

REPORTS_PATH = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. СПИСОК ПРИЗНАКОВ
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
# 3. ЗАГРУЗКА ДАННЫХ И МОДЕЛИ
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

model = joblib.load(MODEL_PATH)

# Проверяем порядок признаков
if hasattr(model, "feature_names_in_"):
    assert list(model.feature_names_in_) == FEATURES, (
        "Порядок признаков модели не совпадает!"
    )

# ВАЖНО:
# Этот код корректен как финальная оценка,
# только если сохранённая модель обучалась
# на X_train, а X_test не использовался
# при обучении и подборе параметров.

print("Размер тестовой выборки:", len(X_test))


# ==========================================
# 4. ПОЛУЧЕНИЕ ПРОГНОЗОВ
# ==========================================

y_pred = model.predict(X_test)

class_index = list(model.classes_).index(1)

y_proba = model.predict_proba(X_test)[
    :, class_index
]


# ==========================================
# 5. МЕТРИКИ КАЧЕСТВА
# ==========================================

metrics = {
    "Accuracy": accuracy_score(
        y_test, y_pred
    ),
    "Precision": precision_score(
        y_test, y_pred,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred
    ),
    "F1": f1_score(
        y_test, y_pred
    ),
    "ROC_AUC": roc_auc_score(
        y_test, y_proba
    ),
    "BalancedAccuracy": balanced_accuracy_score(
        y_test, y_pred
    ),
    "BrierScore": brier_score_loss(
        y_test, y_proba
    ),
    "LogLoss": log_loss(
        y_test, y_proba
    )
}

print("\nФИНАЛЬНЫЕ МЕТРИКИ")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

pd.DataFrame(
    [metrics]
).to_csv(
    REPORTS_PATH / "final_model_metrics.csv",
    index=False
)


# ==========================================
# 6. СРАВНЕНИЕ С DUMMY
# ==========================================

dummy_pred = np.ones(len(y_test))

print("\nБАЗОВАЯ СТРАТЕГИЯ: ВСЕ = 1")

print(
    "Accuracy:",
    round(
        accuracy_score(y_test, dummy_pred),
        4
    )
)

print(
    "Balanced Accuracy:",
    round(
        balanced_accuracy_score(
            y_test, dummy_pred
        ),
        4
    )
)


# ==========================================
# 7. МАТРИЦА ОШИБОК
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nМАТРИЦА ОШИБОК")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Низкая дефектность",
        "Высокая дефектность"
    ]
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Матрица ошибок Random Forest")
plt.tight_layout()

plt.savefig(
    REPORTS_PATH / "final_confusion_matrix.png",
    dpi=300
)

plt.show()


# ==========================================
# 8. КАЛИБРОВКА ВЕРОЯТНОСТЕЙ
# ==========================================

fraction_positive, mean_predicted = (
    calibration_curve(
        y_test,
        y_proba,
        n_bins=8,
        strategy="quantile"
    )
)

plt.figure(figsize=(8, 6))

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Идеальная калибровка"
)

plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    label="Random Forest"
)

plt.xlabel("Прогнозируемая вероятность")
plt.ylabel("Фактическая доля класса 1")

plt.title("Калибровочная кривая")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    REPORTS_PATH / "calibration_curve.png",
    dpi=300
)

plt.show()


# ==========================================
# 9. ПРОВЕРКА ПРОИЗВОДСТВЕННЫХ ПРАВИЛ
# ==========================================

test_data = X_test.copy()
test_data[TARGET] = y_test.to_numpy()

test_data["MaintenanceHigh"] = (
    test_data["MaintenanceHours"] >= 11
)

test_data["VolumeHigh"] = (
    test_data["ProductionVolume"] >= 800
)

test_data["RiskGroup"] = np.select(
    [
        (~test_data["MaintenanceHigh"]) &
        (~test_data["VolumeHigh"]),

        test_data["MaintenanceHigh"] &
        (~test_data["VolumeHigh"]),

        (~test_data["MaintenanceHigh"]) &
        test_data["VolumeHigh"],

        test_data["MaintenanceHigh"] &
        test_data["VolumeHigh"]
    ],
    [
        "Оба показателя ниже порогов",
        "Только обслуживание >= 11 ч",
        "Только объём >= 800",
        "Оба показателя выше порогов"
    ],
    default="Неизвестно"
)

results = []

for group, subset in test_data.groupby(
    "RiskGroup"
):

    total = len(subset)
    positive = int(subset[TARGET].sum())

    rate = positive / total

    lower, upper = proportion_confint(
        positive,
        total,
        alpha=0.05,
        method="wilson"
    )

    results.append({
        "Group": group,
        "Total": total,
        "Class1": positive,
        "Class1Share": rate,
        "CI_Lower": lower,
        "CI_Upper": upper
    })

risk_results = pd.DataFrame(results)

print("\nПРОВЕРКА ПРОИЗВОДСТВЕННЫХ ПРАВИЛ")
print(
    risk_results.round(4).to_string(
        index=False
    )
)

risk_results.to_csv(
    REPORTS_PATH / "final_risk_groups.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 10. ГРАФИК ПРОИЗВОДСТВЕННЫХ ГРУПП
# ==========================================

plt.figure(figsize=(11, 6))

plt.bar(
    risk_results["Group"],
    risk_results["Class1Share"] * 100
)

plt.ylabel("Доля высокого уровня дефектности, %")
plt.title("Проверка производственных правил")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.ylim(0, 105)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    REPORTS_PATH / "final_risk_groups.png",
    dpi=300
)

plt.show()


# ==========================================
# 11. ЗАВЕРШЕНИЕ
# ==========================================

print("\nФИНАЛЬНАЯ ПРОВЕРКА ЗАВЕРШЕНА")
print("Отчёты сохранены в reports/")
