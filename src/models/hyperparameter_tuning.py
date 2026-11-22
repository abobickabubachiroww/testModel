
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw"

# Загрузка данных
csv_files = list(DATA_PATH.glob("*.csv"))

if len(csv_files) != 1:
    raise ValueError("В data/raw должен быть один CSV-файл")

df = pd.read_csv(csv_files[0])

# Исключаем потенциальную утечку данных
X = df.drop(columns=[
    "DefectStatus",
    "DefectRate",
    "QualityScore"
])

y = df["DefectStatus"]

# То же разделение, что и в предыдущих экспериментах
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Модель
model = RandomForestClassifier(
    random_state=42,
    n_jobs=1
)

# Пространство поиска параметров
param_distributions = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8],
    "class_weight": [
        None,
        "balanced",
        "balanced_subsample"
    ]
}

# Стратифицированная кросс-валидация
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Поиск оптимальных параметров
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True
)

print("Начинаем подбор гиперпараметров...")

search.fit(X_train, y_train)

print("\nЛУЧШИЕ ПАРАМЕТРЫ:")
print(search.best_params_)

print("\nЛУЧШИЙ ROC-AUC:")
print(round(search.best_score_, 4))

# Сохраняем модель
model_dir = PROJECT_ROOT / "src" / "models" / "saved"
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "random_forest_tuned.joblib"

joblib.dump(search.best_estimator_, model_path)

print("\nМодель сохранена:", model_path)

# Сохраняем результаты поиска
results = pd.DataFrame(search.cv_results_)

results.to_csv(
    PROJECT_ROOT / "reports" / "hyperparameter_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nПодбор завершён!")
