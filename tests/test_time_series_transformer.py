import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def load_time_series_module(unique_name: str):
    base_dir = Path(__file__).resolve().parents[1]
    module_path = base_dir / "Time series - Transformers" / "train.py"
    spec = importlib.util.spec_from_file_location(unique_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_time_series_transformer_forward_output_shape():
    module = load_time_series_module("time_series_train_forward")
    model = module.TimeSeriesTransformer(
        input_dim=3, d_model=12, n_heads=3, num_layers=1
    )

    dummy_batch = torch.randn(2, 5, 3)
    output = model(dummy_batch)

    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()


def test_train_creates_model_artifact(tmp_path, monkeypatch):
    module = load_time_series_module("time_series_train_exec")

    data = pd.DataFrame({"value": np.linspace(0, 1, 40)})
    data_path = tmp_path / "synthetic.csv"
    data.to_csv(data_path, index=False)

    model_path = tmp_path / "model.pth"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--data",
            str(data_path),
            "--seq_length",
            "5",
            "--batch_size",
            "4",
            "--epochs",
            "1",
            "--lr",
            "0.01",
            "--model_path",
            str(model_path),
        ],
    )

    module.train()

    assert model_path.exists()
