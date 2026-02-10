import os
import shutil
import pytest
from damn.train import train_damn

TEST_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(TEST_FOLDER, "data")
OUTPUT_FOLDER = os.path.join(TEST_FOLDER, "output")
MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "model")
FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, "figure")

@pytest.fixture(autouse=True)
def clean_output():
    """Delete output folder before each test."""
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)
    yield


@pytest.mark.parametrize(
    "organism,file_name,media_file,od_file,cobra_file",
    [
        (
            "putida",
            "putida_OD_81",
            os.path.join(DATA_FOLDER, "putida_media_81.csv"),
            os.path.join(DATA_FOLDER, "putida_OD_81.csv"),
            os.path.join(DATA_FOLDER, "IJN1463EXP_duplicated.xml"),
        )
    ]
)
def test_train_damn_full(organism, file_name, media_file, od_file, cobra_file):
    """Full functional test for DAMN training pipeline."""

    train_damn(
        organism=organism,
        file_name=file_name,
        od_file=od_file,
        media_file=media_file,
        cobra_model_file=cobra_file,
        model_dir=MODEL_FOLDER,
        figure_dir=FIGURE_FOLDER
    )

    # Check model folder exists
    model_folder = os.path.join(OUTPUT_FOLDER, "model")
    assert os.path.exists(model_folder), "Model folder was not created"
    figure_folder = os.path.join(OUTPUT_FOLDER, "figure")
    assert os.path.exists(figure_folder), "Figure folder was not created"

    train_test_split = "medium" if organism == "putida" else "forecast"
    run_name = f"{file_name}_{train_test_split}"

    # Expected validation files
    expected_val_files = [
        f"{model_folder}/{run_name}_val_array.txt",
        f"{model_folder}/{run_name}_val_dev.txt",
        f"{model_folder}/{run_name}_val_ids.txt",
    ]

    for f in expected_val_files:
        assert os.path.exists(f), f"Missing validation file: {f}"

    # Check saved model files
    saved_models = [
        f for f in os.listdir(model_folder)
        if f.startswith(run_name)
    ]

    assert len(saved_models) > 0, "No models were saved"
