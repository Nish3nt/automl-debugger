"""
Tests for AutoML Debugger Engine
=================================
Run with:  python test_engine.py
"""

import sys
import pandas as pd
import numpy as np

sys.path.insert(0, ".")           # make src/ importable from project root
from src.debugger_engine import run_debugger_pipeline


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def assert_ok(output: dict, test_name: str):
    assert output["metrics"], f"[FAIL] {test_name}: metrics dict is empty"
    assert output["llm_analysis"], f"[FAIL] {test_name}: llm_analysis list is empty"
    assert 0 <= output["health_score"] <= 100, f"[FAIL] {test_name}: health_score out of range"
    print(f"[PASS] {test_name}")


# ─────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────

def test_regression_clean():
    section("1. Regression — clean dataset (Boston-like)")
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "area":      np.random.uniform(500, 3000, n),
        "rooms":     np.random.randint(1, 7, n),
        "age":       np.random.uniform(0, 50, n),
        "distance":  np.random.uniform(0.5, 20, n),
        "price":     np.random.uniform(100_000, 900_000, n),
    })
    output = run_debugger_pipeline(df, "price")
    assert output["task_type"] == "regression", "Expected regression"
    assert_ok(output, "Regression clean")
    print("  metrics:", output["metrics"])


def test_classification_binary():
    section("2. Classification — binary (cancer-like)")
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    df   = data.frame
    output = run_debugger_pipeline(df, "target")
    assert output["task_type"] == "classification", "Expected classification"
    assert_ok(output, "Classification binary")
    print("  accuracy_rf:", output["metrics"].get("accuracy_rf"))
    print("  roc_auc:    ", output["metrics"].get("roc_auc"))


def test_missing_values():
    section("3. Dataset with heavy missing values")
    np.random.seed(0)
    n = 200
    df = pd.DataFrame({
        "a": np.random.randn(n),
        "b": np.random.randn(n),
        "c": np.random.randn(n),
        "y": np.random.randn(n),
    })
    # Inject ~30 % missingness
    for col in ["a", "b"]:
        mask = np.random.rand(n) < 0.3
        df.loc[mask, col] = np.nan

    output = run_debugger_pipeline(df, "y")
    assert_ok(output, "Missing values")
    assert output["profile"]["missing_total"] > 0, "Should detect missing values"
    print("  missing_total:", output["profile"]["missing_total"])
    print("  health_score: ", output["health_score"])


def test_categorical_features():
    section("4. Mixed numeric + categorical dataset")
    np.random.seed(7)
    n = 250
    df = pd.DataFrame({
        "age":       np.random.randint(18, 70, n),
        "income":    np.random.uniform(20_000, 150_000, n),
        "education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], n),
        "city":      np.random.choice(["NYC", "LA", "Chicago", "Houston"], n),
        "purchased": np.random.randint(0, 2, n),
    })
    output = run_debugger_pipeline(df, "purchased")
    assert output["task_type"] == "classification", "Expected classification"
    assert_ok(output, "Categorical features")
    print("  metrics:", output["metrics"])


def test_too_small():
    section("5. Edge case — dataset too small")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    output = run_debugger_pipeline(df, "b")
    assert not output["metrics"], "Expected empty metrics for tiny dataset"
    print("[PASS] Too-small guard triggered correctly")
    print("  llm_analysis:", output["llm_analysis"])


def test_auto_target_detection():
    section("6. Auto target detection (no target_column specified)")
    np.random.seed(1)
    df = pd.DataFrame({
        "x1": np.random.randn(100),
        "x2": np.random.randn(100),
        "y":  np.random.randn(100),      # last column → auto target
    })
    output = run_debugger_pipeline(df)    # no target_column
    assert_ok(output, "Auto target detection")
    print("  detected task_type:", output["task_type"])


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_regression_clean,
        test_classification_binary,
        test_missing_values,
        test_categorical_features,
        test_too_small,
        test_auto_target_detection,
    ]

    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {e}")
            failed += 1

    section(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)