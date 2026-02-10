from __future__ import annotations

import json
from pathlib import Path

import pytest

from biorsp.config import load_json_config


def test_load_project_configs():
    root = Path(__file__).resolve().parents[1]
    prereg_cfg = load_json_config(root / "configs" / "biorsp_prereg.json")
    genome_cfg = load_json_config(root / "configs" / "biorsp_genomewide.json")
    assert "h5ad_path" in prereg_cfg
    assert "strata" in prereg_cfg
    assert "h5ad_path" in genome_cfg
    assert "strata" in genome_cfg


def test_invalid_json_reports_line_and_column(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"a": 1,}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"line \d+, column \d+"):
        load_json_config(bad)


def test_non_object_json_config_rejected(tmp_path: Path):
    bad = tmp_path / "list.json"
    bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        load_json_config(bad)


def test_non_json_extension_rejected(tmp_path: Path):
    bad = tmp_path / "cfg.yaml"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Use a .json config file"):
        load_json_config(bad)
