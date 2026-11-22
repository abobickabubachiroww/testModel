
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

from pathlib import Path
from sklearn.model_selection import train_test_split


# 1. Пути проекта

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw"

MODEL_PATH = (
    PROJECT_ROOT / "src" / "models" /
    "saved" / "random_forest_tuned.joblib"
)

REPORTS_PATH = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


# 2. Загрузка данных

csv_files = list(DATA_PATH.glob("*.csv"))

if len(csv_files) != 1:
    raise ValueError("Ожидается один CSV-файл")

df = pd.read_csv(csv_files[0])

X = df.drop(columns=[
    "DefectStatus",
    "DefectRate",
    "QualityScore"
])

y = df["DefectStatus"]


# 3. Используем обучающую часть

X_train, _, y_train, _ = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Внутренняя выборка для анализа
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)


# 4. Новая модель с подобранными параметрами

saved_model = joblib.load(MODEL_PATH)

from sklearn.base import clone

model = clone(saved_model)
model.fit(X_fit, y_fit)

# Берём 300 наблюдений для ускорения SHAP
X_sample = X_val.sample(
    n=min(300, len(X_val)),
    random_state=42
)


# 5. Вычисляем SHAP

print("Начинаем SHAP-анализ...")

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

# Выбираем объяснения именно класса 1
if len(shap_values.values.shape) == 3:
    class_index = list(model.classes_).index(1)
    values = shap_values.values[:, :, class_index]
else:
    values = shap_values.values

print("SHAP рассчитан!")


# 6. Глобальная важность признаков

importance = pd.DataFrame({
    "Feature": X_sample.columns,
    "MeanAbsSHAP": np.abs(values).mean(axis=0)
})

importance = importance.sort_values(
    "MeanAbsSHAP",
    ascending=False
)

print("\nГЛОБАЛЬНАЯ SHAP-ВАЖНОСТЬ")
print(importance.round(5).to_string(index=False))

importance.to_csv(
    REPORTS_PATH / "shap_importance.csv",
    index=False,
    encoding="utf-8-sig"
)


# 7. Summary plot

plt.figure()

shap.summary_plot(
    values,
    X_sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    REPORTS_PATH / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 8. Зависимость от MaintenanceHours

shap.dependence_plot(
    "MaintenanceHours",
    values,
    X_sample,
    interaction_index=None,
    show=False
)

plt.tight_layout()

plt.savefig(
    REPORTS_PATH / "shap_maintenance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 9. Зависимость от ProductionVolume

shap.dependence_plot(
    "ProductionVolume",
    values,
    X_sample,
    interaction_index=None,
    show=False
)

plt.tight_layout()

plt.savefig(
    REPORTS_PATH / "shap_production_volume.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSHAP-анализ завершён!")
print("Графики сохранены в reports/")
