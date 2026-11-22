
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# Пути к проекту
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw"

# Настройки графиков
sns.set_theme(style="whitegrid")


def load_data():
    """Загрузка исходного датасета."""

    csv_files = list(DATA_PATH.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("CSV-файл не найден!")

    return pd.read_csv(csv_files[0])


def analyze_target(df):
    """Анализ целевой переменной."""

    print("\nРАСПРЕДЕЛЕНИЕ КЛАССОВ")

    print(df["DefectStatus"].value_counts())

    print("\nДоли классов (%):")

    print(
        df["DefectStatus"]
        .value_counts(normalize=True) * 100
    )


def analyze_features_by_class(df):
    """Сравнение производственных показателей."""

    print("\nСРЕДНИЕ ЗНАЧЕНИЯ ПО КЛАССАМ")

    comparison = df.groupby("DefectStatus").mean().T

    print(comparison.round(3).to_string())

    print("\nМЕДИАНЫ ПО КЛАССАМ")

    medians = df.groupby("DefectStatus").median().T

    print(medians.round(3).to_string())


def analyze_leakage(df):
    """Проверка потенциальной утечки данных."""

    print("\nПРОВЕРКА DEFECTRATE")

    print(
        df.groupby("DefectStatus")["DefectRate"]
        .agg(["min", "max", "mean", "median"])
    )

    print("\nКОРРЕЛЯЦИЯ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")

    print(
        df.corr(numeric_only=True)["DefectStatus"]
        .sort_values(ascending=False)
    )


def plot_correlation(df):
    """Корреляционная матрица без DefectRate."""

    analysis_df = df.drop(columns=["DefectRate"])

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        analysis_df.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Корреляционная матрица производственных показателей")

    plt.tight_layout()

    plt.savefig(
        PROJECT_ROOT / "reports" / "correlation_matrix.png",
        dpi=300
    )

    plt.show()


def plot_features_by_class(df):
    """Распределение показателей по классам."""

    features = [
        col for col in df.select_dtypes(include="number").columns
        if col not in ["DefectStatus", "DefectRate"]
    ]

    fig, axes = plt.subplots(5, 3, figsize=(18, 22))

    axes = axes.flatten()

    for i, feature in enumerate(features):

        sns.boxplot(
            data=df,
            x="DefectStatus",
            y=feature,
            ax=axes[i]
        )

        axes[i].set_title(feature)
        axes[i].set_xlabel("DefectStatus")

    plt.tight_layout()

    plt.savefig(
        PROJECT_ROOT / "reports" / "features_by_class.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":

    df = load_data()

    analyze_target(df)

    analyze_features_by_class(df)

    analyze_leakage(df)

    plot_correlation(df)

    plot_features_by_class(df)

    print("\nАнализ завершён!")
