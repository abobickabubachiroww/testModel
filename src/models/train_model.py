
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# Путь к проекту
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw"

# Загрузка датасета
csv_files = list(DATA_PATH.glob("*.csv"))

if len(csv_files) != 1:
    raise ValueError("В data/raw должен быть один CSV-файл")

df = pd.read_csv(csv_files[0])

# Исключаем целевую переменную и признаки с риском утечки
excluded_columns = [
    "DefectStatus",
    "DefectRate",
    "QualityScore"
]

X = df.drop(columns=excluded_columns)
y = df["DefectStatus"]

print("Количество признаков:", X.shape[1])
print("Используемые признаки:", X.columns.tolist())

# Разделение данных: 80% обучение, 20% тест
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nОбучающая выборка:", len(X_train))
print("Тестовая выборка:", len(X_test))

# Создаём модели
models = {
    "Dummy": DummyClassifier(
        strategy="most_frequent"
    ),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []

# Обучение и проверка моделей
for name, model in models.items():

    print("\n" + "=" * 50)
    print("МОДЕЛЬ:", name)
    print("=" * 50)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    print("\nМатрица ошибок:")
    print(confusion_matrix(y_test, predictions))

    print("\nОтчёт классификации:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc
    })

# Итоговое сравнение
results_df = pd.DataFrame(results)

print("\n" + "=" * 50)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 50)

print(results_df.round(4).to_string(index=False))

# Сохраняем результаты
output_path = PROJECT_ROOT / "reports" / "model_comparison.csv"

results_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nРезультаты сохранены:", output_path)
