
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT / "data" / "raw" /
    "manufacturing_defect_dataset.csv"
)


def get_dashboard_data():
    df = pd.read_csv(DATA_PATH)

    # Распределение классов
    counts = df["DefectStatus"].value_counts()

    # Данные для графика обслуживания
    maintenance = (
        df.groupby("DefectStatus")["MaintenanceHours"]
        .mean()
    )

    # Данные для графика объёма производства
    volume = (
        df.groupby("DefectStatus")["ProductionVolume"]
        .mean()
    )

    result = {
        "total": int(len(df)),

        "class_distribution": {
            "low": int(counts.get(0, 0)),
            "high": int(counts.get(1, 0))
        },

        "maintenance_by_class": {
            "low": round(
                float(maintenance.get(0, 0)), 2
            ),
            "high": round(
                float(maintenance.get(1, 0)), 2
            )
        },

        "production_volume_by_class": {
            "low": round(
                float(volume.get(0, 0)), 2
            ),
            "high": round(
                float(volume.get(1, 0)), 2
            )
        }
    }

    return result


if __name__ == "__main__":
    data = get_dashboard_data()

    print(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        )
    )
