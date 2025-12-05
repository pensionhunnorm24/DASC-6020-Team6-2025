# tests/test_pipeline.py
import sys
import joblib
from pathlib import Path
import pytest

# Locate project root by searching upward for a directory that contains 'src'
def find_project_root(start: Path = None) -> Path:
    if start is None:
        start = Path.cwd()
    cur = start.resolve()
    root = Path(cur.root)
    while True:
        if (cur / "src").is_dir():
            return cur
        if cur == root:
            return start
        cur = cur.parent

PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
ARTIFACTS = PROJECT_ROOT.joinpath("artifacts")


def ensure_project_on_path():
    """Ensure the project root is on sys.path so unpickling can import src.* modules."""
    p = str(PROJECT_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def test_artifacts_directory_exists():
    """Artifacts directory should exist after training."""
    assert ARTIFACTS.exists() and ARTIFACTS.is_dir(), f"Artifacts directory not found at {ARTIFACTS}"


def test_at_least_one_pipeline_saved():
    """There should be at least one saved pipeline file matching *_pipeline.joblib."""
    files = list(ARTIFACTS.glob("*_pipeline.joblib"))
    assert len(files) > 0, "No pipeline artifacts found in artifacts/; run training first."


def test_pipeline_structure():
    """
    Load one pipeline and verify it contains 'preproc' and 'model' keys.

    This test is robust to import/unpickle issues by ensuring the project root is on sys.path
    before attempting to load the joblib artifact. If unpickling still fails, the test will
    fail with a helpful message.
    """
    files = sorted(ARTIFACTS.glob("*_pipeline.joblib"))
    assert files, "No pipeline artifacts to inspect"

    # Ensure project root is on sys.path so pickled objects referencing src.* can be imported
    ensure_project_on_path()

    first_file = files[0]
    try:
        pipeline = joblib.load(first_file)
    except ModuleNotFoundError as e:
        # Try again after ensuring project root is on path (defensive)
        ensure_project_on_path()
        try:
            pipeline = joblib.load(first_file)
        except Exception as e2:
            pytest.fail(
                "Failed to unpickle pipeline artifact. This usually means the artifact "
                "references project modules that are not importable in the current environment. "
                f"Original error: {e}; Retry error: {e2}. Ensure you're running tests from the project root "
                "and that the virtual environment contains the same package layout used to create the artifact."
            )
    except Exception as e:
        pytest.fail(f"Failed to load pipeline artifact {first_file}: {e}")

    assert isinstance(pipeline, dict), "Loaded pipeline artifact is not a dict"
    assert "preproc" in pipeline and "model" in pipeline, "Pipeline artifact must contain 'preproc' and 'model'"