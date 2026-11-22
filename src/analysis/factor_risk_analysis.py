
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from statsmodels.stats.proportion import proportion_confint


# ==========================================
# 1. НАСТРОЙКА ПУТЕЙ
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" /
    "manufacturing_defect_dataset.csv"
)

REPORTS_PATH = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. ЗАГРУЗКА ДАННЫХ
# ==========================================

df = pd.read_csv(DATA_PATH)

# Используем обучающую часть исходного
# разделения, не затрагивая финальный тест.

train_df, _ = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["DefectStatus"]
)

print("Количество партий:", len(train_df))


# ==========================================
# 3. ФУНКЦИЯ АНАЛИЗА ГРУПП
# ==========================================

def analyze_groups(data, group_column):

    results = []

    for group, subset in data.groupby(
        group_column,
        observed=True
    ):

        total = len(subset)

        defective = int(
            subset["DefectStatus"].sum()
        )

        rate = defective / total

        # 95% доверительный интервал Уилсона
        lower, upper = proportion_confint(
            defective,
            total,
            alpha=0.05,
            method="wilson"
        )

        results.append({
            "Group": str(group),
            "Total": total,
            "Defective": defective,
            "DefectShare": rate,
            "CI_Lower": lower,
            "CI_Upper": upper
        })

    return pd.DataFrame(results)


# ==========================================
# 4. АНАЛИЗ MAINTENANCE HOURS
# ==========================================

train_df = train_df.copy()

train_df["MaintenanceGroup"] = np.where(
    train_df["MaintenanceHours"] < 11,
    "Менее 11 часов",
    "11 часов и более"
)

maintenance_results = analyze_groups(
    train_df,
    "MaintenanceGroup"
)

print("\nАНАЛИЗ ВРЕМЕНИ ОБСЛУЖИВАНИЯ")
print(
    maintenance_results.round(4).to_string(
        index=False
    )
)

maintenance_results.to_csv(
    REPORTS_PATH / "maintenance_risk.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 5. АНАЛИЗ PRODUCTION VOLUME
# ==========================================

train_df["VolumeGroup"] = np.where(
    train_df["ProductionVolume"] < 800,
    "Менее 800 единиц",
    "800 единиц и более"
)

volume_results = analyze_groups(
    train_df,
    "VolumeGroup"
)

print("\nАНАЛИЗ ОБЪЁМА ПРОИЗВОДСТВА")
print(
    volume_results.round(4).to_string(
        index=False
    )
)

volume_results.to_csv(
    REPORTS_PATH / "production_volume_risk.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 6. СОВМЕСТНОЕ ВЛИЯНИЕ ФАКТОРОВ
# ==========================================

train_df["CombinedGroup"] = (
    train_df["MaintenanceGroup"] +
    " | " +
    train_df["VolumeGroup"]
)

combined_results = analyze_groups(
    train_df,
    "CombinedGroup"
)

print("\nСОВМЕСТНЫЙ АНАЛИЗ ФАКТОРОВ")
print(
    combined_results.round(4).to_string(
        index=False
    )
)

combined_results.to_csv(
    REPORTS_PATH / "combined_factor_risk.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 7. ВИЗУАЛИЗАЦИЯ
# ==========================================

def plot_results(results, title, filename):

    fig, ax = plt.subplots(figsize=(10, 6))

    values = results["DefectShare"] * 100

    lower_error = (
        results["DefectShare"] -
        results["CI_Lower"]
    ) * 100

    upper_error = (
        results["CI_Upper"] -
        results["DefectShare"]
    ) * 100

    positions = np.arange(len(results))

    ax.bar(
        positions,
        values,
        yerr=[lower_error, upper_error],
        capsize=5
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        results["Group"],
        rotation=15,
        ha="right"
    )

    ax.set_ylabel(
        "Доля партий класса 1, %"
    )

    ax.set_title(title)
    ax.set_ylim(0, 105)

    ax.grid(
        axis="y",
        alpha=0.3
    )

    for i, value in enumerate(values):
        ax.text(
            i,
            min(value + upper_error.iloc[i] + 2, 103),
            f"{value:.1f}%",
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        REPORTS_PATH / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


plot_results(
    maintenance_results,
    "Доля дефектных партий и время обслуживания",
    "maintenance_risk.png"
)

plot_results(
    volume_results,
    "Доля дефектных партий и объём производства",
    "production_volume_risk.png"
)

plot_results(
    combined_results,
    "Совместная связь факторов с дефектностью",
    "combined_factor_risk.png"
)


# ==========================================
# 8. ИТОГ
# ==========================================

print("\nАНАЛИЗ ЗАВЕРШЁН!")
print("Результаты сохранены в reports/")
