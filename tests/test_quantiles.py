import numpy as np
import pytest

from aeromaintain.models.quantiles import evaluate_quantiles


def test_quantile_coverage():
    y_true = np.array([
        10,
        20,
        30,
        40
    ])

    q10 = np.array([
        5, 
        15,
        35,
        35
    ])

    q50 = np.array([
        10, 
        20,
        40,
        40
    ])

    q90 = np.array([
        15, 
        25,
        45,
        45
    ])

    results = evaluate_quantiles(
        y_true,
        q10,
        q50,
        q90
    )

    assert results["Couverture réelle"] == pytest.approx(0.75)
    assert results["Largeur moyenne"] == pytest.approx(10.0)


def test_quantile_crossing():
    y_true = np.array([
        10,
        20,
        30,
        40
    ])

    q10 = np.array([
        5, 
        15,
        35,
        35
    ])

    q50 = np.array([
        10, 
        30,
        40,
        40
    ])

    q90 = np.array([
        15, 
        25,
        45,
        45
    ])

    results = evaluate_quantiles(
        y_true,
        q10,
        q50,
        q90
    )

    assert results["Taux croisement quantiles"] == pytest.approx(0.25)
