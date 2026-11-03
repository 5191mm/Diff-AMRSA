"""Release packaging and command-line contract checks."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def test_release_excludes_intermediate_tables():
    assert not (ROOT / "processed_drug_mic_dataset.csv").exists()
    assert not (ROOT / "mrsa_scheme_a").exists()


def test_license_and_model_figure_present():
    assert (ROOT / "LICENSE.md").is_file()
    assert "Apache License" in (ROOT / "LICENSE.md").read_text(encoding="utf-8")
    assert (ROOT / "assets/diff_amrsa_overview.png").is_file()


def test_legacy_runner_rejects_unknown_step(tmp_path):
    cfg = tmp_path / "config.env"
    cfg.write_text("export DIFF_AMRSA_RAW_DIR=/tmp\n", encoding="utf-8")
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts/run_legacy_pipeline.sh"), "--config", str(cfg), "not-a-step"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 2
    assert "Unknown step" in proc.stderr
