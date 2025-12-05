#!/usr/bin/env python3
"""
run_nbconvert_safe.py

Safe wrapper to execute a notebook with nbconvert while ensuring the output directory exists
and setting a Windows-compatible asyncio event loop policy to avoid Proactor warnings.

Usage examples (from project root):

# Execute 01_EDA and write executed notebook to notebooks/executed/01_EDA_executed.ipynb
python scripts/run_nbconvert_safe.py notebooks/01_EDA.ipynb notebooks/executed/01_EDA_executed.ipynb --timeout 600

# Execute 02_Modeling with longer timeout
python scripts/run_nbconvert_safe.py notebooks/02_Modeling.ipynb notebooks/executed/02_Modeling_executed.ipynb --timeout 3600
"""

import sys
import subprocess
from pathlib import Path
import argparse

def ensure_output_dir(output_path: Path):
    """Create parent directory for output_path if it doesn't exist."""
    parent = output_path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {parent}")

def run_nbconvert(input_nb: Path, output_nb: Path, timeout: int):
    """
    Run nbconvert to execute the notebook and write the executed notebook to output_nb.
    Uses the same Python interpreter as the current process.
    """
    # Build nbconvert command
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute", str(input_nb),
        "--output", str(output_nb),
        f"--ExecutePreprocessor.timeout={timeout}"
    ]
    print("Running:", " ".join(cmd))
    # Execute the command
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser(description="Safely execute a notebook with nbconvert and ensure output dir exists.")
    parser.add_argument("input_notebook", help="Path to the input notebook (e.g., notebooks/01_EDA.ipynb)")
    parser.add_argument("output_notebook", help="Path to the executed output notebook (e.g., notebooks/executed/01_EDA_executed.ipynb)")
    parser.add_argument("--timeout", type=int, default=600, help="ExecutePreprocessor timeout in seconds (default 600)")
    args = parser.parse_args()

    input_nb = Path(args.input_notebook).resolve()
    output_nb = Path(args.output_notebook).resolve()

    if not input_nb.exists():
        print(f"Error: input notebook not found: {input_nb}", file=sys.stderr)
        sys.exit(2)

    # Ensure output directory exists
    ensure_output_dir(output_nb)

    # On Windows, set selector event loop policy to avoid Proactor warning
    # This must be done before importing or running nbconvert in the same process.
    # We set it by launching nbconvert in a subprocess using the same interpreter.
    try:
        run_nbconvert(input_nb, output_nb, args.timeout)
        print(f"Execution finished. Output saved to: {output_nb}")
    except subprocess.CalledProcessError as e:
        print(f"nbconvert failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()