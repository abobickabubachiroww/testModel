
import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw"

csv_files = list(DATA_PATH.glob("*.csv"))

if len(csv_files) != 1:
    raise ValueError("В data/raw должен быть один CSV-файл")

df = pd.read_csv(csv_files[0])

X = df.drop(
    columns=[
        "DefectStatus",
        "DefectRate",
        "QualityScore"
    ]
)

y = df["DefectStatus"]

# Используем то же разделение, что и в первом эксперименте
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Кросс-валидация только на обучающей выборке
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
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

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy"
}

results = []

for name, model in models.items():

    print("\n" + "=" * 50)
    print("МОДЕЛЬ:", name)
    print("=" * 50)

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    row = {"Model": name}

    for metric in scoring:

        values = scores["test_" + metric]

        mean = values.mean()
        std = values.std()

        row[metric] = mean

        print(
            f"{metric}: {mean:.4f} "
            f"(± {std:.4f})"
        )

    results.append(row)

results_df = pd.DataFrame(results)

print("\nИТОГОВОЕ СРАВНЕНИЕ")

print(results_df.round(4).to_string(index=False))

output_path = (
    PROJECT_ROOT / "reports" / "cross_validation.csv"
)

results_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nРезультаты сохранены:", output_path)
