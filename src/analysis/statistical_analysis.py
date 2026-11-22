
import pandas as pd
import numpy as np

from pathlib import Path
from scipy.stats import mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw"


def load_data():

    csv_files = list(DATA_PATH.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("CSV-файл не найден!")

    return pd.read_csv(csv_files[0])


def compare_classes(df):

    features = [
        col for col in df.select_dtypes(include="number").columns
        if col not in ["DefectStatus", "DefectRate"]
    ]

    results = []

    for feature in features:

        group_0 = df.loc[
            df["DefectStatus"] == 0, feature
        ]

        group_1 = df.loc[
            df["DefectStatus"] == 1, feature
        ]

        statistic, p_value = mannwhitneyu(
            group_0,
            group_1,
            alternative="two-sided"
        )

        pooled_std = np.sqrt(
            (
                group_0.var(ddof=1) +
                group_1.var(ddof=1)
            ) / 2
        )

        cohens_d = (
            group_1.mean() - group_0.mean()
        ) / pooled_std

        results.append({
            "Feature": feature,
            "Mean_Class_0": group_0.mean(),
            "Mean_Class_1": group_1.mean(),
            "Median_Class_0": group_0.median(),
            "Median_Class_1": group_1.median(),
            "Cohens_d": cohens_d,
            "P_Value": p_value
        })

    result_df = pd.DataFrame(results)

    # Поправка Бонферрони на множественные проверки
    alpha = 0.05 / len(features)

    result_df["Significant"] = (
        result_df["P_Value"] < alpha
    )

    result_df["Abs_Effect"] = (
        result_df["Cohens_d"].abs()
    )

    result_df = result_df.sort_values(
        "Abs_Effect",
        ascending=False
    )

    print("\nРЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОГО АНАЛИЗА")

    print(result_df.round(4).to_string(index=False))

    output_path = (
        PROJECT_ROOT / "reports" / "statistical_analysis.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nРезультаты сохранены:", output_path)


if __name__ == "__main__":

    df = load_data()

    compare_classes(df)
