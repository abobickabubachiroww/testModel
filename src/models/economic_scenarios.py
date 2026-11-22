
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_PATH = PROJECT_ROOT / "reports"

df = pd.read_csv(
    REPORTS_PATH / "threshold_results.csv"
)

# Различные экономические сценарии
scenarios = {
    "Сценарий 1: 1 к 1": (5000, 5000),
    "Сценарий 2: 2 к 1": (10000, 5000),
    "Сценарий 3: 5 к 1": (25000, 5000),
    "Сценарий 4: 10 к 1": (50000, 5000),
    "Сценарий 5: 20 к 1": (100000, 5000)
}

results = []

for name, (cost_fn, cost_fp) in scenarios.items():

    # Рассчитываем стоимость для каждого порога
    costs = (
        df["FN"] * cost_fn +
        df["FP"] * cost_fp
    )

    best_index = costs.idxmin()

    best_threshold = df.loc[
        best_index, "Threshold"
    ]

    best_cost = costs.loc[best_index]

    # Базовая стратегия: все партии = 1
    total_negative = int(
        df.iloc[0]["TN"] + df.iloc[0]["FP"]
    )

    baseline_cost = total_negative * cost_fp

    savings = baseline_cost - best_cost

    results.append({
        "Scenario": name,
        "Cost_FN": cost_fn,
        "Cost_FP": cost_fp,
        "Best_Threshold": best_threshold,
        "Best_Cost": best_cost,
        "Baseline_Cost": baseline_cost,
        "Savings": savings
    })

results_df = pd.DataFrame(results)

print("\nЭКОНОМИЧЕСКИЕ СЦЕНАРИИ")
print(results_df.to_string(index=False))

# Сохраняем результаты
results_df.to_csv(
    REPORTS_PATH / "economic_scenarios.csv",
    index=False,
    encoding="utf-8-sig"
)

# График потенциальной экономии
plt.figure(figsize=(11, 6))

plt.bar(
    results_df["Scenario"],
    results_df["Savings"]
)

plt.title("Потенциальная экономия по сценариям")
plt.xlabel("Экономический сценарий")
plt.ylabel("Экономия, руб.")

plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(
    REPORTS_PATH / "economic_scenarios.png",
    dpi=300
)

plt.show()

print("\nАнализ завершён!")
