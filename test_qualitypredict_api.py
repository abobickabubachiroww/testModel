"""QualityPredict API regression tests. Run: python -m pytest -v test_qualitypredict_api.py"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

API_FILE = Path(__file__).with_name('api.py')
spec = importlib.util.spec_from_file_location('qualitypredict_api_under_test', API_FILE)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)

SAMPLE = {
    'ProductionVolume': 850, 'ProductionCost': 12000,
    'SupplierQuality': 85, 'DeliveryDelay': 3,
    'MaintenanceHours': 12, 'DowntimePercentage': 8,
    'InventoryTurnover': 5, 'StockoutRate': .05,
    'WorkerProductivity': 80, 'SafetyIncidents': 1,
    'EnergyConsumption': 1500, 'EnergyEfficiency': .8,
    'AdditiveProcessTime': 4, 'AdditiveMaterialCost': 200,
}
ECONOMICS = {
    'downtime_hours_per_month': 16,
    'cost_per_downtime_hour': 5000,
    'expected_reduction_percent': 25,
    'implementation_cost': 60000,
    'monthly_operating_cost': 2000,
}


class DummyModel:
    classes_ = np.array([0, 1])
    feature_importances_ = np.ones(len(api.FEATURES)) / len(api.FEATURES)

    def predict_proba(self, X):
        return np.array([[0.2, 0.8] for _ in range(len(X))])


class DummyExplainer:
    def __call__(self, X):
        # Baseline + SHAP contributions must equal predict_proba class 1 (0.8).
        class Result:
            values = np.zeros((1, len(api.FEATURES), 2))
            base_values = np.array([[0.2, 0.7]])
        result = Result()
        result.values[0, api.FEATURES.index('MaintenanceHours'), 1] = 0.1
        return result


@pytest.fixture
def client(monkeypatch):
    rows = []
    for i in range(8):
        row = {key: float(value) + (i * .1 if key not in ('SupplierQuality', 'WorkerProductivity', 'StockoutRate', 'DowntimePercentage') else 0)
               for key, value in SAMPLE.items()}
        row['DefectStatus'] = i % 2
        rows.append(row)
    df = pd.DataFrame(rows)
    artifact = {'model': DummyModel(), 'threshold': .62,
                'features': api.FEATURES, 'model_name': 'RF Balanced Subsample'}
    monkeypatch.setattr(api, 'load_data', lambda: df)
    monkeypatch.setattr(api, 'load_model', lambda: artifact)
    monkeypatch.setattr(api, 'load_explainer', lambda: DummyExplainer())
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.mark.parametrize('path', [
    '/', '/api/dashboard', '/api/features',
    '/api/features/MaintenanceHours/distribution',
    '/api/features/MaintenanceHours/comparison',
    '/api/correlations', '/api/maintenance-groups',
    '/api/production-groups', '/api/risk-groups',
    '/api/feature-importance', '/api/model-metrics',
    '/api/recommendations',
])
def test_get_endpoints(client, path):
    response = client.get(path)
    assert response.status_code == 200, (path, response.text)
    assert isinstance(response.json(), dict)


def test_dashboard_consistency(client):
    result = client.get('/api/dashboard').json()
    assert result['total'] == 8
    assert result['class_distribution'] == {'low': 4, 'high': 4}
    assert result['class_share']['low'] + result['class_share']['high'] == 1


def test_features_count(client):
    result = client.get('/api/features').json()
    assert len(result['features']) == 14
    assert result['target'] == 'DefectStatus'
    assert 'DefectRate' not in [f['name'] for f in result['features']]
    assert 'QualityScore' not in [f['name'] for f in result['features']]


def test_distribution_counts(client):
    result = client.get('/api/features/MaintenanceHours/distribution?bins=4').json()
    assert len(result['bin_edges']) == 5
    assert sum(sum(series['counts']) for series in result['series']) == 8


def test_correlation_dimensions(client):
    result = client.get('/api/correlations').json()
    assert len(result['features']) == 15
    assert len(result['matrix']) == 15
    assert all(len(row) == 15 for row in result['matrix'])


def test_risk_group_totals(client):
    result = client.get('/api/risk-groups').json()
    assert len(result['groups']) == 4
    assert sum(group['total'] for group in result['groups']) == 8


@pytest.mark.parametrize('path', [
    '/api/features/Unknown/distribution',
    '/api/features/Unknown/comparison',
])
def test_unknown_feature(client, path):
    assert client.get(path).status_code == 404


@pytest.mark.parametrize('bins', [0, 1, 61])
def test_invalid_histogram_bins(client, bins):
    assert client.get(f'/api/features/MaintenanceHours/distribution?bins={bins}').status_code == 422


@pytest.mark.parametrize('path', [
    '/api/predict', '/api/explain', '/api/recommendations/personalized',
])
def test_model_post_endpoints(client, path):
    response = client.post(path, json=SAMPLE)
    assert response.status_code == 200, (path, response.text)
    result = response.json()
    assert result['predicted_class'] == 1
    assert result['score'] == 0.8
    if path.endswith('explain'):
        assert len(result['all_factors']) == 14
        assert result['explanation_method'] == 'SHAP TreeExplainer'
    if path.endswith('personalized'):
        assert result['recommendations']
        assert result['recommendations'][0]['feature'] == 'MaintenanceHours'
        assert result['recommendations'][0]['priority'] == 'high'


@pytest.mark.parametrize('field,value', [
    ('ProductionVolume', -1),
    ('SupplierQuality', 101),
    ('DowntimePercentage', 101),
    ('StockoutRate', 1.1),
    ('WorkerProductivity', 101),
    ('MaintenanceHours', -5),
])
def test_invalid_prediction_inputs(client, field, value):
    payload = {**SAMPLE, field: value}
    assert client.post('/api/predict', json=payload).status_code == 422


def test_missing_and_extra_prediction_fields(client):
    missing = dict(SAMPLE)
    del missing['ProductionVolume']
    assert client.post('/api/predict', json=missing).status_code == 422
    assert client.post('/api/predict', json={**SAMPLE, 'DefectRate': 0.1}).status_code == 422


def test_economics_expected_values(client):
    response = client.post('/api/economics/calculate', json=ECONOMICS)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['gross_monthly_savings'] == 20000
    assert result['net_monthly_savings'] == 18000
    assert result['payback_months'] == 3.33
    assert result['first_year_net_benefit'] == 156000
    assert result['scenario'] == 'hypothetical'


def test_economics_no_payback(client):
    payload = {**ECONOMICS, 'expected_reduction_percent': 0}
    result = client.post('/api/economics/calculate', json=payload)
    assert result.status_code == 200
    assert result.json()['payback_months'] is None
    assert result.json()['net_monthly_savings'] == -2000


def test_economics_zero_implementation(client):
    payload = {**ECONOMICS, 'implementation_cost': 0}
    result = client.post('/api/economics/calculate', json=payload)
    assert result.status_code == 200
    assert result.json()['payback_months'] == 0


@pytest.mark.parametrize('field,value', [
    ('downtime_hours_per_month', -1),
    ('cost_per_downtime_hour', -1),
    ('expected_reduction_percent', -1),
    ('expected_reduction_percent', 101),
    ('implementation_cost', -1),
    ('monthly_operating_cost', -1),
])
def test_invalid_economics_inputs(client, field, value):
    payload = {**ECONOMICS, field: value}
    assert client.post('/api/economics/calculate', json=payload).status_code == 422


def test_economics_missing_and_extra_fields(client):
    payload = dict(ECONOMICS)
    del payload['implementation_cost']
    assert client.post('/api/economics/calculate', json=payload).status_code == 422
    assert client.post('/api/economics/calculate', json={**ECONOMICS, 'unknown': 1}).status_code == 422


def test_economics_overflow(client):
    payload = {**ECONOMICS, 'downtime_hours_per_month': 1e308, 'cost_per_downtime_hour': 1e308}
    assert client.post('/api/economics/calculate', json=payload).status_code == 422
