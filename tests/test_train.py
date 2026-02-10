import os
import numpy as np
import pytest
from damn.train import train_damn

TEST_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(TEST_FOLDER, "data")
#OUTPUT_FOLDER = os.path.join(TEST_FOLDER, "output")
#MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "model")
#FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, "figure")

#@pytest.fixture(autouse=True)
#def clean_output():
#    """Delete output folder before each test."""
#    if os.path.exists(OUTPUT_FOLDER):
#        shutil.rmtree(OUTPUT_FOLDER)
#    os.makedirs(OUTPUT_FOLDER)
#    yield


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
    """Full functional test for DAMN training pipeline with in-memory outputs."""

    # Run the training and capture all returned variables
    (
        mdl,
        run_name,
        train_array,
        train_dev,
        val_array,
        val_dev,
        val_ids,
        losses_s_v_train,
        losses_neg_v_train,
        losses_c_train,
        losses_drop_c_train,
        losses_s_v_val,
        losses_neg_v_val,
        losses_c_val,
        losses_drop_c_val
    ) = train_damn(
        organism=organism,
        file_name=file_name,
        od_file=od_file,
        media_file=media_file,
        cobra_model_file=cobra_file,
    )

    # Check run_name format
    train_test_split = "medium" if organism == "putida" else "forecast"
    expected_run_name = f"{file_name}_{train_test_split}"
    assert run_name == expected_run_name, "run_name mismatch"

    # Check model object
    assert mdl is not None, "Model object is None"
    assert hasattr(mdl, "save_model"), "Model object missing save_model method"

    # Check arrays
    assert isinstance(train_array, np.ndarray), "train_array should be a numpy array"
    assert isinstance(train_dev, np.ndarray), "train_dev should be a numpy array"
    assert isinstance(val_array, np.ndarray), "val_array should be a numpy array"
    assert isinstance(val_dev, np.ndarray), "val_dev should be a numpy array"
    assert isinstance(val_ids, (list, np.ndarray)), "val_ids should be list or numpy array"

    # Check losses
    loss_vars = [
        losses_s_v_train, losses_neg_v_train, losses_c_train, losses_drop_c_train,
        losses_s_v_val, losses_neg_v_val, losses_c_val, losses_drop_c_val
    ]
    for loss in loss_vars:
        assert isinstance(loss, np.ndarray) or isinstance(loss, list), "Loss should be list or ndarray"
        assert len(loss) > 0, "Loss array is empty"

