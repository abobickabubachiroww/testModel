
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)


# Пути проекта
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = (
    PROJECT_ROOT / "src" / "models" /
    "saved" / "random_forest_tuned.joblib"
)
REPORTS_PATH = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


# Демонстрационные экономические допущения
COST_FN = 50000  # Пропущенная дефектная партия
COST_FP = 5000   # Ложная тревога


def load_data():
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

    return X, y


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    total_cost = fn * COST_FN + fp * COST_FP

    return {
        "Threshold": round(float(threshold), 3),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "BalancedAccuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "TotalCost": int(total_cost)
    }


def main():

    X, y = load_data()

    # Сохраняем прежнее разделение
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Загружаем настроенную модель
    model = joblib.load(MODEL_PATH)

    # Пять стратифицированных разбиений
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    print("Получаем out-of-fold прогнозы...")

    probabilities = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    # Проверяем разные пороги
    thresholds = np.round(
        np.arange(0.01, 1.00, 0.01), 2
    )

    results = []

    for threshold in thresholds:
        metrics = calculate_metrics(
            y_train,
            probabilities,
            threshold
        )
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Минимизируем условные экономические потери
    best_index = results_df["TotalCost"].idxmin()
    best = results_df.loc[best_index]

    print("\nОПТИМАЛЬНЫЙ ПОРОГ")
    print(best.to_string())

    # Сравниваем с порогом 0.5
    default = calculate_metrics(
        y_train,
        probabilities,
        0.5
    )

    # Базовая стратегия: все партии = класс 1
    all_positive = calculate_metrics(
        y_train,
        np.ones(len(y_train)),
        0.5
    )

    print("\nСТАНДАРТНЫЙ ПОРОГ 0.5")
    print(pd.Series(default).to_string())

    print("\nБАЗОВАЯ СТРАТЕГИЯ: ВСЕ ПАРТИИ = 1")
    print(pd.Series(all_positive).to_string())

    # Сохраняем таблицу
    results_df.to_csv(
        REPORTS_PATH / "threshold_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Сохраняем выбранный порог и допущения
    config = {
        "threshold": float(best["Threshold"]),
        "cost_false_negative": COST_FN,
        "cost_false_positive": COST_FP,
        "selection_method": "out_of_fold_cost_minimization"
    }

    config_path = MODEL_PATH.parent / "threshold_config.json"

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    # График экономических потерь
    plt.figure(figsize=(10, 6))

    plt.plot(
        results_df["Threshold"],
        results_df["TotalCost"],
        label="Стоимость ошибок модели"
    )

    plt.axvline(
        best["Threshold"],
        color="red",
        linestyle="--",
        label=f"Выбранный порог: {best['Threshold']:.2f}"
    )

    plt.axhline(
        all_positive["TotalCost"],
        color="gray",
        linestyle=":",
        label="Стратегия: все партии = 1"
    )

    plt.xlabel("Порог классификации")
    plt.ylabel("Условные потери, руб.")
    plt.title("Экономическая оптимизация порога")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        REPORTS_PATH / "threshold_optimization.png",
        dpi=300
    )

    plt.show()

    print("\nРезультаты сохранены!")
    print("Конфигурация:", config_path)


if __name__ == "__main__":
    main()
