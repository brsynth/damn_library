import os
import sys
import subprocess
import tempfile
import pytest
import damn

@pytest.mark.integration
def test_main_execution_putida():
    """
    Integration test: run the main damn_predict script via CLI.
    Uses the packaged model in ./data/modul.zip.
    """

    TEST_FOLDER = os.path.dirname(__file__)
    data_file = os.path.join(TEST_FOLDER, "data", "modul.zip")
    assert os.path.isfile(data_file), "data zip file not found"
    figure_dir=os.path.join(TEST_FOLDER, "output")
    assert os.path.isfile(data_file), "output file not found"


    # Run script as subprocess (like calling from CLI)
    result = subprocess.run(
        [
            sys.executable, "-m", "damn", "predict",
            "--organism", "putida",
            "--model-dir", data_file,
            "--figure-dir", figure_dir
        ],
        capture_output=True,
        text=True,
        cwd=os.path.join(TEST_FOLDER, "..") 
    )

    # Ensure the script ran without crashing
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Check that some figure output was produced
    output_files = os.listdir(figure_dir)
    assert any(f.endswith(".png") for f in output_files), \
        f"No figure outputs found in {figure_dir}"
