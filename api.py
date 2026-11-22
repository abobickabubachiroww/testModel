
from functools import lru_cache
from pathlib import Path
from typing import Optional
import math
import numpy as np
import shap
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "manufacturing_defect_dataset.csv"
MODEL_PATH = ROOT / "src" / "models" / "saved" / "qualitypredict_final.joblib"

FEATURES = [
    "ProductionVolume", "ProductionCost", "SupplierQuality", "DeliveryDelay",
    "MaintenanceHours", "DowntimePercentage", "InventoryTurnover",
    "StockoutRate", "WorkerProductivity", "SafetyIncidents",
    "EnergyConsumption", "EnergyEfficiency", "AdditiveProcessTime",
    "AdditiveMaterialCost",
]
LABELS = {
    "ProductionVolume": ("Объём производства", "ед./день"),
    "ProductionCost": ("Производственные затраты", "USD/день"),
    "SupplierQuality": ("Качество поставщиков", "%"),
    "DeliveryDelay": ("Задержка поставки", "дни"),
    "MaintenanceHours": ("Обслуживание оборудования", "ч/неделю"),
    "DowntimePercentage": ("Доля простоя", "%"),
    "InventoryTurnover": ("Оборачиваемость запасов", "коэффициент"),
    "StockoutRate": ("Доля дефицита запасов", "доля (0–1)"),
    "WorkerProductivity": ("Производительность работников", "%"),
    "SafetyIncidents": ("Инциденты безопасности", "шт./месяц"),
    "EnergyConsumption": ("Энергопотребление", "кВт·ч"),
    "EnergyEfficiency": ("Энергоэффективность", "коэффициент"),
    "AdditiveProcessTime": ("Время аддитивного процесса", "ч"),
    "AdditiveMaterialCost": ("Стоимость аддитивного материала", "USD/ед."),
}

app = FastAPI(
    title="QualityPredict API", version="2.0.0",
    description="Анализ завершённых производственных наблюдений и классификация уровня дефектности.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",
                   "http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def load_data():
    if not DATA_PATH.is_file():
        raise HTTPException(503, "Файл производственных данных не найден")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        raise HTTPException(503, "Не удалось прочитать CSV") from exc
    required = set(FEATURES) | {"DefectStatus"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(503, f"В CSV отсутствуют столбцы: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise HTTPException(503, "В необходимых столбцах CSV есть пропуски")
    if not df["DefectStatus"].isin([0, 1]).all():
        raise HTTPException(503, "DefectStatus должен содержать только 0 или 1")
    return df


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise HTTPException(503, "Файл модели не найден")
    try:
        # joblib.load выполняет десериализацию: загружайте только собственный доверенный файл.
        artifact = joblib.load(MODEL_PATH)
        if not isinstance(artifact, dict):
            raise ValueError("Артефакт должен быть словарём")
        if not {"model", "threshold", "features", "model_name"}.issubset(artifact):
            raise ValueError("В артефакте отсутствуют обязательные ключи")
        if not 0 <= float(artifact["threshold"]) <= 1:
            raise ValueError("Порог должен быть между 0 и 1")
        if set(artifact["features"]) != set(FEATURES) or len(artifact["features"]) != len(FEATURES):
            raise ValueError("Набор признаков модели не совпадает с API")
        return artifact
    except Exception as exc:
        raise HTTPException(503, "Не удалось загрузить обученную модель") from exc


def require_feature(name: str):
    if name not in FEATURES:
        raise HTTPException(404, f"Неизвестный признак: {name}. Список: GET /api/features")
    return name


def group_stats(df):
    total = len(df)
    high = int((df["DefectStatus"] == 1).sum())
    return {
        "total": total,
        "high_defect": high,
        "low_defect": total - high,
        "high_defect_share": round(high / total, 4) if total else None,
    }


def summary(s: pd.Series):
    if s.empty:
        return {"count": 0, "mean": None, "median": None, "std": None,
                "min": None, "q1": None, "q3": None, "max": None}
    def val(x):
        return round(float(x), 4) if pd.notna(x) else None
    return {
        "count": int(s.count()), "mean": val(s.mean()), "median": val(s.median()),
        "std": val(s.std()), "min": val(s.min()), "q1": val(s.quantile(.25)),
        "q3": val(s.quantile(.75)), "max": val(s.max()),
    }


@app.get("/")
def root():
    return {"service": "QualityPredict API", "version": "2.0.0", "status": "running"}


@app.get("/api/dashboard")
def dashboard():
    df = load_data()
    counts = df["DefectStatus"].value_counts()
    maintenance = df.groupby("DefectStatus")["MaintenanceHours"].mean()
    volume = df.groupby("DefectStatus")["ProductionVolume"].mean()
    return {
        "total": int(len(df)),
        "class_distribution": {"low": int(counts.get(0, 0)), "high": int(counts.get(1, 0))},
        "maintenance_by_class": {"low": round(float(maintenance.get(0, 0)), 2),
                                 "high": round(float(maintenance.get(1, 0)), 2)},
        "production_volume_by_class": {"low": round(float(volume.get(0, 0)), 2),
                                       "high": round(float(volume.get(1, 0)), 2)},
        "missing_values": int(df.isna().sum().sum()),
        "columns": int(len(df.columns)),
        "class_share": {"low": round(float((df.DefectStatus == 0).mean()), 4),
                        "high": round(float((df.DefectStatus == 1).mean()), 4)},
    }


@app.get("/api/features")
def features():
    return {"features": [{"name": name, "label": LABELS[name][0], "unit": LABELS[name][1]}
                         for name in FEATURES], "target": "DefectStatus"}


@app.get("/api/features/{name}/distribution")
def feature_distribution(name: str, bins: int = Query(15, ge=2, le=60)):
    require_feature(name)
    df = load_data()
    values = df[name].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise HTTPException(422, "Признак содержит нечисловые значения")
    if values.min() == values.max():
        edges = np.array([values.min() - .5, values.max() + .5])
    else:
        edges = np.histogram_bin_edges(values, bins=bins)
    result = []
    for cls in (0, 1):
        hist, _ = np.histogram(df.loc[df.DefectStatus == cls, name], bins=edges)
        result.append({"class": cls, "counts": hist.astype(int).tolist()})
    return {
        "feature": name, "label": LABELS[name][0], "unit": LABELS[name][1],
        "bin_edges": [round(float(x), 6) for x in edges],
        "series": result,
        "note": "Количество наблюдений по интервалам; класс 0 — низкая, 1 — высокая дефектность",
    }


@app.get("/api/features/{name}/comparison")
def feature_comparison(name: str):
    require_feature(name)
    df = load_data()
    return {
        "feature": name, "label": LABELS[name][0], "unit": LABELS[name][1],
        "by_class": {
            "low": summary(df.loc[df.DefectStatus == 0, name]),
            "high": summary(df.loc[df.DefectStatus == 1, name]),
        },
    }


@app.get("/api/correlations")
def correlations():
    df = load_data()
    cols = FEATURES + ["DefectStatus"]
    matrix = df[cols].corr(method="pearson")
    return {
        "method": "pearson", "features": cols,
        "matrix": [[round(float(x), 4) if pd.notna(x) else None for x in row]
                   for row in matrix.to_numpy()],
        "note": "Корреляция не доказывает причинно-следственную связь",
    }


@app.get("/api/maintenance-groups")
def maintenance_groups():
    df = load_data()
    return {
        "threshold_hours_per_week": 11,
        "below_threshold": group_stats(df[df.MaintenanceHours < 11]),
        "at_or_above_threshold": group_stats(df[df.MaintenanceHours >= 11]),
    }


@app.get("/api/production-groups")
def production_groups():
    df = load_data()
    return {
        "threshold_units_per_day": 800,
        "below_threshold": group_stats(df[df.ProductionVolume < 800]),
        "at_or_above_threshold": group_stats(df[df.ProductionVolume >= 800]),
    }


@app.get("/api/risk-groups")
def risk_groups():
    df = load_data()
    mh = df.MaintenanceHours >= 11
    pv = df.ProductionVolume >= 800
    groups = []
    for m in (False, True):
        for p in (False, True):
            subset = df[(mh == m) & (pv == p)]
            groups.append({"maintenance_ge_11": m, "volume_ge_800": p,
                           **group_stats(subset)})
    return {
        "thresholds": {"maintenance_hours_per_week": 11, "production_units_per_day": 800},
        "groups": groups,
        "note": "Доля класса 1 среди наблюдений группы, а не доля бракованных изделий",
    }


@app.get("/api/feature-importance")
def feature_importance():
    artifact = load_model()
    model = artifact["model"]
    if not hasattr(model, "feature_importances_"):
        raise HTTPException(501, "У этой модели нет встроенной значимости признаков")
    values = model.feature_importances_
    names = artifact["features"]
    items = [{"feature": name, "label": LABELS[name][0],
              "importance": round(float(value), 6)}
             for name, value in zip(names, values)]
    items.sort(key=lambda item: item["importance"], reverse=True)
    return {
        "method": "random_forest_impurity_importance",
        "model": artifact["model_name"], "items": items,
        "note": "Встроенная значимость Random Forest не является SHAP или доказательством причинности",
    }


@app.get("/api/model-metrics")
def model_metrics():
    # Зафиксированные результаты ранее выполненной проверки на 648 наблюдениях.
    # Не пересчитываем по всем 3240: это исказило бы оценку обобщения.
    return {
        "model": "RF Balanced Subsample", "threshold": 0.62,
        "evaluation": "previously_evaluated_test_split",
        "test_size": 648,
        "metrics": {
            "accuracy": 0.7469, "precision_class_1": 0.9441,
            "recall_class_1": 0.7431, "f1_class_1": 0.8316,
            "roc_auc": 0.7517, "balanced_accuracy": 0.7551,
        },
        "confusion_matrix": [[79, 24], [140, 405]],
        "labels": [0, 1],
        "note": "Исторический результат теста; выборка уже использовалась при исследовании моделей",
    }


@app.get("/api/recommendations")
def recommendations():
    return {
        "recommendations": [
            {
                "id": "equipment_diagnostics", "title": "Проверить состояние оборудования",
                "basis": "В данных повышенные MaintenanceHours связаны с классом высокой дефектности",
                "actions": ["Разделить плановое обслуживание и аварийные ремонты",
                            "Проверить журналы отказов и причины ремонтов",
                            "Сопоставить неисправности с результатами контроля качества"],
            },
            {
                "id": "production_load", "title": "Проверить влияние производственной нагрузки",
                "basis": "Высокий ProductionVolume связан с повышенной долей класса 1",
                "actions": ["Сравнить качество при разных уровнях загрузки",
                            "Проверить соблюдение технологических режимов",
                            "Провести контролируемое исследование возможных причин"],
            },
            {
                "id": "quality_control", "title": "Усилить контроль групп повышенной дефектности",
                "basis": "Группы с высокими MaintenanceHours или ProductionVolume чаще имеют класс 1",
                "actions": ["Назначить дополнительные выборочные проверки",
                            "Отслеживать изменения доли класса 1 после мероприятий"],
            },
        ],
        "disclaimer": "Рекомендации — гипотезы для проверки, а не доказанные причинные эффекты",
    }


class ProductionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ProductionVolume: float = Field(ge=0, allow_inf_nan=False)
    ProductionCost: float = Field(ge=0, allow_inf_nan=False)
    SupplierQuality: float = Field(ge=0, le=100, allow_inf_nan=False)
    DeliveryDelay: float = Field(ge=0, allow_inf_nan=False)
    MaintenanceHours: float = Field(ge=0, allow_inf_nan=False)
    DowntimePercentage: float = Field(ge=0, le=100, allow_inf_nan=False)
    InventoryTurnover: float = Field(ge=0, allow_inf_nan=False)
    StockoutRate: float = Field(ge=0, le=1, allow_inf_nan=False)
    WorkerProductivity: float = Field(ge=0, le=100, allow_inf_nan=False)
    SafetyIncidents: float = Field(ge=0, allow_inf_nan=False)
    EnergyConsumption: float = Field(ge=0, allow_inf_nan=False)
    EnergyEfficiency: float = Field(ge=0, allow_inf_nan=False)
    AdditiveProcessTime: float = Field(ge=0, allow_inf_nan=False)
    AdditiveMaterialCost: float = Field(ge=0, allow_inf_nan=False)


@app.post("/api/predict")
def predict_defect(data: ProductionInput):
    artifact = load_model()
    model = artifact["model"]
    values = data.model_dump()
    names = artifact["features"]
    X = pd.DataFrame([[values[name] for name in names]], columns=names)
    try:
        class_index = list(model.classes_).index(1)
        score = float(model.predict_proba(X)[0, class_index])
    except Exception as exc:
        raise HTTPException(500, "Ошибка выполнения классификации") from exc
    threshold = float(artifact["threshold"])
    predicted_class = int(score >= threshold)
    return {
        "predicted_class": predicted_class,
        "label": "Высокий уровень дефектности" if predicted_class else "Низкий уровень дефектности",
        "score": round(score, 4), "threshold": round(threshold, 4),
        "model": artifact["model_name"],
        "note": "Ретроспективная классификация; score не гарантированно откалиброванная вероятность",
    }

# ==========================================
# 11. ЛОКАЛЬНОЕ ОБЪЯСНЕНИЕ МОДЕЛИ
# ==========================================

@lru_cache(maxsize=1)
def load_explainer():
    artifact = load_model()
    classifier = artifact["model"]

    try:
        return shap.TreeExplainer(classifier)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail=(
                "Не удалось создать SHAP-объяснитель. "
                "Проверьте тип сохранённой модели."
            ),
        )


@app.post("/api/explain")
def explain_prediction(data: ProductionInput):
    artifact = load_model()

    classifier = artifact["model"]
    threshold = float(artifact["threshold"])
    features = artifact["features"]

    values = data.model_dump()

    X = pd.DataFrame(
        [[values[name] for name in features]],
        columns=features,
    )

    try:
        class_index = list(
            classifier.classes_
        ).index(1)

        score = float(
            classifier.predict_proba(X)[0][class_index]
        )

        explainer = load_explainer()
        explanation = explainer(X)

        shap_values = np.asarray(
            explanation.values
        )

        base_values = np.asarray(
            explanation.base_values
        )

        # В разных версиях SHAP бинарная
        # классификация может иметь
        # разные формы массивов.
        if shap_values.ndim == 3:
            contributions = shap_values[
                0, :, class_index
            ]

            baseline = float(
                base_values[0, class_index]
            )

        elif shap_values.ndim == 2:
            contributions = shap_values[0]

            baseline = float(
                np.ravel(base_values)[0]
            )

            # Для бинарного классификатора
            # проверяем, что объясняется
            # именно оценка класса 1.
            reconstructed = (
                baseline + contributions.sum()
            )

            if abs(reconstructed - score) > 0.01:
                raise ValueError(
                    "SHAP объясняет не оценку класса 1"
                )

        else:
            raise ValueError(
                "Неожиданная форма SHAP-значений"
            )

        reconstructed = (
            baseline + contributions.sum()
        )

        if abs(reconstructed - score) > 0.01:
            raise ValueError(
                "SHAP-значения не совпадают "
                "с оценкой классификатора"
            )

        factors = []

        for name, contribution in zip(
            features, contributions
        ):
            contribution = float(contribution)

            if contribution > 0:
                direction = "increases_score"
            elif contribution < 0:
                direction = "decreases_score"
            else:
                direction = "neutral"

            factors.append({
                "feature": name,
                "value": float(values[name]),
                "shap_value": round(
                    contribution, 6
                ),
                "direction": direction,
            })

        factors.sort(
            key=lambda item: abs(
                item["shap_value"]
            ),
            reverse=True,
        )

        predicted_class = int(
            score >= threshold
        )

        return {
            "predicted_class": predicted_class,
            "label": (
                "Высокий уровень дефектности"
                if predicted_class == 1
                else "Низкий уровень дефектности"
            ),
            "score": round(score, 4),
            "threshold": round(threshold, 4),
            "baseline_score": round(
                baseline, 4
            ),
            "top_factors": factors[:5],
            "all_factors": factors,
            "explanation_method": "SHAP TreeExplainer",
            "note": (
                "SHAP объясняет вклад признаков "
                "в оценку модели, но не доказывает "
                "причины возникновения дефектов."
            ),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Ошибка вычисления SHAP: "
                f"{error}"
            ),
        )


# Персональные рекомендации на основе SHAP для конкретного наблюдения.
PERSONALIZED_ACTIONS = {
    "ProductionVolume": ("Проверить производственную нагрузку", ["Сопоставить загрузку оборудования с результатами контроля качества", "Проверить соблюдение технологических режимов при текущем объёме"]),
    "ProductionCost": ("Проверить структуру производственных затрат", ["Сопоставить изменения затрат с материалами и технологическими операциями", "Проверить, не менялись ли поставщики или параметры производства"]),
    "SupplierQuality": ("Проверить качество сырья", ["Провести входной контроль сырья по партиям поставщиков", "Сопоставить показатели поставщиков с результатами контроля качества"]),
    "DeliveryDelay": ("Проверить влияние задержек поставок", ["Проанализировать задержки и вынужденные изменения производственного графика", "Проверить условия хранения и замены сырья при задержках"]),
    "MaintenanceHours": ("Проверить состояние оборудования", ["Разделить плановое обслуживание и аварийные ремонты", "Проверить журналы отказов и сопоставить их с дефектами"]),
    "DowntimePercentage": ("Разобрать причины простоев", ["Определить основные причины остановок оборудования", "Проверить качество продукции после перезапусков"]),
    "InventoryTurnover": ("Проверить управление запасами", ["Проверить сроки и условия хранения материалов", "Сопоставить оборачиваемость с изменениями качества сырья"]),
    "StockoutRate": ("Проверить дефицит материалов", ["Проанализировать случаи нехватки сырья и замен материалов", "Проверить влияние перебоев на соблюдение технологии"]),
    "WorkerProductivity": ("Проверить организацию работы", ["Сопоставить производительность с загрузкой смен", "Проверить соблюдение инструкций и обучение персонала"]),
    "SafetyIncidents": ("Проверить производственную безопасность", ["Разобрать зарегистрированные инциденты и нарушения процедур", "Проверить связь инцидентов с отклонениями технологического процесса"]),
    "EnergyConsumption": ("Проверить энергопотребление", ["Сопоставить расход энергии с режимами работы оборудования", "Проверить отклонения технологических параметров"]),
    "EnergyEfficiency": ("Проверить энергоэффективность оборудования", ["Проверить техническое состояние и настройки оборудования", "Сопоставить энергоэффективность с результатами контроля качества"]),
    "AdditiveProcessTime": ("Проверить параметры аддитивного процесса", ["Сопоставить длительность процесса с технологическим регламентом", "Проверить параметры операций и контрольные измерения"]),
    "AdditiveMaterialCost": ("Проверить аддитивные материалы", ["Проверить характеристики и партии используемых материалов", "Сопоставить смену материалов с результатами контроля качества"]),
}


@app.post("/api/recommendations/personalized")
def personalized_recommendations(data: ProductionInput):
    """Рекомендации для проверки факторов, повышающих оценку класса 1."""
    explanation = explain_prediction(data)
    factors = [
        factor for factor in explanation["all_factors"]
        if factor["shap_value"] > 0
    ]
    factors.sort(key=lambda factor: factor["shap_value"], reverse=True)

    recommendations_list = []
    for factor in factors[:5]:
        name = factor["feature"]
        title, actions = PERSONALIZED_ACTIONS[name]
        recommendations_list.append({
            "feature": name,
            "label": LABELS[name][0],
            "value": factor["value"],
            "shap_value": factor["shap_value"],
            "priority": "high" if len(recommendations_list) < 2 else "medium",
            "title": title,
            "description": (
                "Этот показатель повышает оценку высокого уровня дефектности "
                "для данного наблюдения по модели SHAP."
            ),
            "actions": actions,
        })

    return {
        "predicted_class": explanation["predicted_class"],
        "label": explanation["label"],
        "score": explanation["score"],
        "threshold": explanation["threshold"],
        "recommendations": recommendations_list,
        "disclaimer": (
            "Рекомендации — гипотезы для проверки. SHAP показывает вклад "
            "признаков в оценку модели, но не доказывает причины брака. "
            "При низком классе рекомендации также отражают положительные "
            "вклады в оценку класса 1, а не выявленный дефект."
        ),
    }


# Сценарный расчёт экономической эффективности.
class EconomicsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    downtime_hours_per_month: float = Field(ge=0, allow_inf_nan=False)
    cost_per_downtime_hour: float = Field(ge=0, allow_inf_nan=False)
    expected_reduction_percent: float = Field(ge=0, le=100, allow_inf_nan=False)
    implementation_cost: float = Field(ge=0, allow_inf_nan=False)
    monthly_operating_cost: float = Field(ge=0, allow_inf_nan=False)


@app.post("/api/economics/calculate")
def calculate_economics(data: EconomicsInput):
    """Оценка экономии на простоях по заданным пользователем допущениям."""
    gross = (
        data.downtime_hours_per_month
        * data.cost_per_downtime_hour
        * data.expected_reduction_percent / 100
    )
    net = gross - data.monthly_operating_cost
    payback = data.implementation_cost / net if net > 0 else None
    first_year = net * 12 - data.implementation_cost

    results = (gross, net, first_year)
    if not all(math.isfinite(x) for x in results) or (
        payback is not None and not math.isfinite(payback)
    ):
        raise HTTPException(422, "Слишком большие значения для расчёта")

    return {
        "gross_monthly_savings": round(gross, 2),
        "net_monthly_savings": round(net, 2),
        "payback_months": round(payback, 2) if payback is not None else None,
        "first_year_net_benefit": round(first_year, 2),
        "currency": "RUB",
        "scenario": "hypothetical",
        "note": (
            "Сценарный расчёт на основе допущений пользователя. "
            "Сокращение простоев не подтверждено результатами модели."
        ),
    }
