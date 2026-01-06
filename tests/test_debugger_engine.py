import pandas as pd
from src.debugger_engine import run_debugger_pipeline

def test_engine_stability():
    df = pd.read_csv("stock_data_with_indicators.csv")
    target = df.columns[-1]

    output = run_debugger_pipeline(df, target)

    required_keys = [
        "failure_mode_id",
        "diagnosis",
        "root_causes",
        "recommendations",
        "confidence_score",
        "deployment_risk_score"
    ]

    for key in required_keys:
        assert key in output, f"Missing key: {key}"

    assert isinstance(output["root_causes"], list)
    assert isinstance(output["recommendations"], list)
