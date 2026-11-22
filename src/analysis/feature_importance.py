
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


# ==========================================
# 1. ПУТИ ПРОЕКТА
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw"

MODEL_PATH = (
    PROJECT_ROOT / "src" / "models" /
    "saved" / "random_forest_tuned.joblib"
)

REPORTS_PATH = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. ЗАГРУЗКА ДАННЫХ
# ==========================================

def load_data():

    csv_files = list(DATA_PATH.glob("*.csv"))

    if len(csv_files) != 1:
        raise ValueError(
            "В data/raw должен находиться один CSV-файл"
        )

    df = pd.read_csv(csv_files[0])

    # Исключаем целевую переменную и утечки
    X = df.drop(columns=[
        "DefectStatus",
        "DefectRate",
        "QualityScore"
    ])

    y = df["DefectStatus"]

    return X, y


# ==========================================
# 3. АНАЛИЗ ВАЖНОСТИ
# ==========================================

def analyze_feature_importance():

    X, y = load_data()

    # Внешнее разделение, как в предыдущих опытах
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Внутренняя валидация
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=42,
        stratify=y_train
    )

    # Берём параметры сохранённой модели,
    # но обучаем новую модель
    saved_model = joblib.load(MODEL_PATH)
    model = clone(saved_model)

    model.fit(X_fit, y_fit)

    # Исходное качество
    probabilities = model.predict_proba(X_val)[:, 1]

    baseline_auc = roc_auc_score(
        y_val, probabilities
    )

    print("\nROC-AUC на валидации:")
    print(round(baseline_auc, 4))

    # Вычисляем важность признаков
    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="roc_auc",
        n_repeats=20,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "ImportanceMean": result.importances_mean,
        "ImportanceStd": result.importances_std
    })

    importance_df = importance_df.sort_values(
        "ImportanceMean",
        ascending=False
    )

    print("\nВАЖНОСТЬ ПРОИЗВОДСТВЕННЫХ ФАКТОРОВ")
    print(
        importance_df.round(4).to_string(index=False)
    )

    # Сохраняем результаты
    importance_df.to_csv(
        REPORTS_PATH / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ==========================================
    # 4. ВИЗУАЛИЗАЦИЯ
    # ==========================================

    plot_df = importance_df.sort_values(
        "ImportanceMean",
        ascending=True
    )

    plt.figure(figsize=(11, 8))

    plt.barh(
        plot_df["Feature"],
        plot_df["ImportanceMean"],
        xerr=plot_df["ImportanceStd"],
        capsize=3
    )

    plt.axvline(
        x=0,
        color="black",
        linewidth=1
    )

    plt.xlabel("Среднее снижение ROC-AUC")
    plt.ylabel("Производственный показатель")

    plt.title(
        "Важность факторов производственного брака"
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS_PATH / "feature_importance.png",
        dpi=300
    )

    plt.show()

    print("\nАнализ важности завершён!")


if __name__ == "__main__":
    analyze_feature_importance()
