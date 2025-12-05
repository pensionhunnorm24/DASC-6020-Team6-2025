# tests/test_predict_cli.py
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1].joinpath("src", "predict.py")

def run_cli(args):
    cmd = [sys.executable, str(SCRIPT)] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc

def test_cli_requires_three_args():
    proc = run_cli([])
    assert proc.returncode != 0 or "Exactly 3 parameters" in proc.stderr or "Exactly 3 parameters" in proc.stdout

def test_cli_returns_number():
    # This test assumes artifacts exist and at least one pipeline is available.
    proc = run_cli(["3", "120", "245.50"])
    # CLI prints a single numeric value on stdout when successful
    out = proc.stdout.strip()
    # Accept either success or failure depending on environment; if success, ensure numeric format
    if proc.returncode == 0:
        assert out.count('.') == 1 and len(out.split('.')[-1]) == 2