import os
import glob
import sklearn
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
import damn
from damn import model
from damn import utils
from damn import plot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_damn(
    organism="putida",

    # INPUT folders
    model_dir="data",

    # Output folders
    #figure_dir="./figure",

    # File & data overrides
    file_name=None,
    metabolite_ids=None,

    # Evaluation params overrides
    train_test_split=None,
    N_iter=3,
    OD=None,
    plot_type=None,

    # Plot params 
    # range you want the plots to be printed
    R2min=0,
    R2max=0.5,
):

    # DEFAULT PRESETS BY ORGANISM
    PRESETS = {
        "putida": {
            "file_name": "putida_OD_81", # 'putida_OD_81' or 'M28_OD_20'
            "metabolite_ids": [],
            "train_test_split": "medium", # 'forecast' or 'medium'
            "N_iter": 3,
            "OD": True, # when True biomass concentration transformed in OD
            "plot_type": "growth", # 'growth' or 'substrate'
        },

        "ecoli": {
            "file_name": "M28_OD_20", # 'putida_OD_81' or 'M28_OD_20'
            "metabolite_ids": [
                'glc__D_e','xyl__D_e','succ_e','ala__L_e','arg__L_e','asn__L_e',
                'asp__L_e','cys__L_e','glu__L_e','gln__L_e','gly_e','his__L_e',
                'ile__L_e','leu__L_e','lys__L_e','met__L_e','phe__L_e','pro__L_e',
                'ser__L_e','thr__L_e','trp__L_e','tyr__L_e','val__L_e',
                'ade_e','gua_e','csn_e','ura_e','thymd_e','BIOMASS'
            ],
            "train_test_split": "medium", # 'forecast' or 'medium'
            "N_iter": 3,
            "OD": True, # when True biomass concentration transformed in OD
            "plot_type": "growth", # 'growth' or 'substrate'
        },

        "custom": {
            # minimal defaults — user must override
            "file_name": None, # 'putida_OD_81' or 'M28_OD_20'
            "metabolite_ids": [],
            "train_test_split": "medium", # 'forecast' or 'medium'
            "N_iter": 1,
            "OD": True, # when True biomass concentration transformed in OD
            "plot_type": "growth", # 'growth' or 'substrate'
        }
    }

    if organism not in PRESETS:
        raise ValueError(f"Organism must be one of {list(PRESETS.keys())}")

    cfg = PRESETS[organism]

    # APPLY PRESETS + OVERRIDES
    file_name = file_name if file_name is not None else cfg["file_name"]
    metabolite_ids = metabolite_ids if metabolite_ids is not None else cfg["metabolite_ids"]
    train_test_split = train_test_split if train_test_split is not None else cfg["train_test_split"]
    N_iter = N_iter if N_iter is not None else cfg["N_iter"]
    OD = OD if OD is not None else cfg["OD"]
    plot_type = plot_type if plot_type is not None else cfg["plot_type"]

    # LOAD DATA
    run_name = f"{file_name}_{train_test_split}"

    val_array = np.loadtxt(os.path.join(model_dir, f"{run_name}_val_array.txt"), dtype=float) 
    val_dev = np.loadtxt(os.path.join(model_dir, f"{run_name}_val_dev.txt"), dtype=float) 
    val_ids = np.loadtxt(os.path.join(model_dir, f"{run_name}_val_ids.txt"), dtype=int) 
    
    if val_array is None: 
        raise ValueError(f"Validation file not found: {run_name}_val_array.txt")

    #val_array_file = glob.glob(os.path.join(model_dir, "*_val_array.txt"))
    #val_dev_file   = glob.glob(os.path.join(model_dir, "*_val_dev.txt"))
    #val_ids_file   = glob.glob(os.path.join(model_dir, "*_val_ids.txt"))

    #val_array = np.loadtxt(val_array_file[0], dtype=float)
    #val_dev   = np.loadtxt(val_dev_file[0], dtype=float)
    #val_ids   = np.loadtxt(val_ids_file[0], dtype=int)


    # PREDICT
    Pred, Ref = {}, {}
    for i in range(N_iter):
        mdl_name = os.path.join(model_dir, f"{run_name}_{i}")

        mdl = model.MetabolicModel.load_model(
            model_name=mdl_name,
            metabolite_ids=metabolite_ids,
            verbose=False
        )

        pred, ref = model.predict_on_val_data(mdl, val_array, verbose=False)
        Pred[i], Ref[i] = np.asarray(pred), np.asarray(ref)

    Pred = np.asarray(list(Pred.values()))
    Ref  = np.asarray(list(Ref.values()))

    # METRICS
    if metabolite_ids is not None:
        R2, R2dev = utils.r2_growth_curve_with_std(Pred, Ref, OD=OD)
    else:
        R2 = utils.r2_growth_curve(Pred, Ref, OD=OD)
        R2dev = None

    print(
        f"Model: {run_name} | "
        f"R2 = {np.mean(R2):.2f} ± {np.std(R2):.2f} | "
        f"Median = {np.median(R2):.2f}"
    )
    times = mdl.times
    return  Pred, Ref, R2, R2dev, times, val_dev, val_ids , mdl,  run_name, train_test_split, plot_type, OD, R2min, R2max

