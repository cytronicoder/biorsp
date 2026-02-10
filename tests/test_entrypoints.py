from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_script_module(script_name: str):
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_biorsp_evaluation_entrypoint_calls_prereg_pipeline(monkeypatch):
    fake_eval = types.ModuleType("biorsp.evaluation")
    fake_eval.run_prereg_pipeline = lambda _cfg: None
    monkeypatch.setitem(sys.modules, "biorsp.evaluation", fake_eval)

    module = _load_script_module("prereg_pipeline_legacy.py")
    called: list[str] = []

    def _fake_run(config_path: str) -> None:
        called.append(config_path)

    monkeypatch.setattr(module, "run_prereg_pipeline", _fake_run)
    monkeypatch.setattr(sys, "argv", ["prereg_pipeline_legacy.py", "--config", "tiny.json"])

    rc = module.main()
    assert rc == 0
    assert called == ["tiny.json"]
