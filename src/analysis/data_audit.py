import pandas as pd
from pathlib import Path

# Определяем путь к проекту
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Путь к исходным данным
DATA_PATH = PROJECT_ROOT / "data" / "raw"

def load_dataset():
    """Загружает исходный CSV-файл."""

    csv_files = list(DATA_PATH.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "CSV-файл не найден в папке data/raw"
        )

    if len(csv_files) > 1:
        raise ValueError(
            "В папке несколько CSV-файлов. "
            "Оставьте один исходный датасет."
        )

    df = pd.read_csv(csv_files[0])

    print("Датасет успешно загружен!")
    print("Файл:", csv_files[0].name)

    return df

def audit_dataset(df):
    """Проводит первичный аудит данных."""

    print("\n" + "=" * 50)
    print("1. РАЗМЕР ДАТАСЕТА")
    print("=" * 50)

    print("Строк:", df.shape[0])
    print("Столбцов:", df.shape[1])

    print("\n" + "=" * 50)
    print("2. НАЗВАНИЯ СТОЛБЦОВ")
    print("=" * 50)

    print(df.columns.tolist())

    print("\n" + "=" * 50)
    print("3. ТИПЫ ДАННЫХ")
    print("=" * 50)

    print(df.dtypes)

    print("\n" + "=" * 50)
    print("4. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ")
    print("=" * 50)

    print(df.isnull().sum())

    print("\n" + "=" * 50)
    print("5. ДУБЛИКАТЫ")
    print("=" * 50)

    print("Количество:", df.duplicated().sum())

    print("\n" + "=" * 50)
    print("6. СТАТИСТИКА")
    print("=" * 50)

    print(df.describe().T)

    print("\n" + "=" * 50)
    print("7. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ")
    print("=" * 50)

    if "DefectStatus" in df.columns:
        print(df["DefectStatus"].value_counts(dropna=False))
        print("\nДоли классов (%):")
        print(
            df["DefectStatus"]
            .value_counts(normalize=True, dropna=False) * 100
        )
    else:
        print("Столбец DefectStatus не найден!")

if __name__ == "__main__":
    df = load_dataset()
    audit_dataset(df)