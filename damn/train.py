def train_damn(
    organism="custom",

    # Output folders (NEW)
    model_dir="./model",
    figure_dir="./figure",

    # File & data overrides
    file_name = None,
    od_file=None,  
    media_file=None,
    cobra_model_file=None,
    biomass_rxn_id=None,

    # Training params overrides
    train_test_split=None,
    seed=None,
    num_epochs=None,
    loss_weight=None,
    loss_decay=None,

    # Architecture defaults (can override)
    hidden_layers_lag=[50],
    hidden_layers_flux=[500],
    dropout_rate=0.2,
    x_fold=3,
    batch_size=10,
    patience=100,
    N_iter=3,
):
    import os
    import numpy as np
    import tensorflow as tf
    from . import model
    from . import plot

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # DEFAULT PRESETS BY ORGANISM
    PRESETS = {
        "putida": {
            "train_test_split": "medium",
            "file_name": "putida_OD_81",
            "od_file": "putida_OD_81.csv",
            "media_file": "putida_media_81.csv",
            "cobra_model_file": "IJN1463EXP_duplicated.xml",
            "biomass_rxn_id": "BIOMASS_KT2440_WT3",
            "seed": 1,
            "num_epochs": 500,
            "loss_weight": [0.1, 0.1, 1, 1.0],
            "loss_decay": [0, 1, 0, 0.5],
        },

        "ecoli": {
            "train_test_split": "forecast",
            "file_name": "M28_OD_20",
            "od_file": "M28_OD_20.csv",
            "media_file": "M28_media.csv",
            "cobra_model_file": "iML1515_duplicated.xml",
            "biomass_rxn_id": "BIOMASS_Ec_iML1515_core_75p37M",
            "seed": 30,
            "num_epochs": 1000,
            "loss_weight": [0.1, 1, 1, 1],
            "loss_decay": [0, 1, 0, 0],
        },

        "custom": {
            "train_test_split": "medium",
            "file_name": "M28_OD_20",
            "media_file": "custom_media.csv",
            "cobra_model_file": "model.xml",
            "biomass_rxn_id": "BIOMASS_RXN",
            "seed": 1,
            "num_epochs": 500,
            "loss_weight": [0.1, 0.1, 1, 1],
            "loss_decay": [0, 1, 0, 0],
        }
    }

    if organism not in PRESETS:
        raise ValueError(f"Organism must be one of {list(PRESETS.keys())}")

    cfg = PRESETS[organism]

    # Apply overrides if provided
    train_test_split = train_test_split or cfg["train_test_split"]
    file_name= file_name or cfg["file_name"]
    od_file = od_file or cfg["od_file"]   
    media_file = media_file or cfg["media_file"]
    cobra_model_file = cobra_model_file or cfg["cobra_model_file"]
    biomass_rxn_id = biomass_rxn_id or cfg["biomass_rxn_id"]
    seed = seed if seed is not None else cfg["seed"]
    num_epochs = num_epochs or cfg["num_epochs"]
    loss_weight = loss_weight or cfg["loss_weight"]
    loss_decay = loss_decay or cfg["loss_decay"]

    media_file = media_file
    od_file = od_file

    run_name = f'{file_name}_{train_test_split}'

    # Seed reproducibility
    np.random.seed(seed)

    # CREATE MODEL
    mdl, train_array, train_dev, val_array, val_dev, val_ids = model.create_model_train_val(
        media_file, od_file,
        cobra_model_file,
        biomass_rxn_id,
        x_fold=x_fold,
        hidden_layers_lag=hidden_layers_lag,
        hidden_layers_flux=hidden_layers_flux,
        dropout_rate=dropout_rate,
        loss_weight=loss_weight,
        loss_decay=loss_decay,
        verbose=False,
        train_test_split=train_test_split
    )

    # Save validation metadata
    os.makedirs(model_dir, exist_ok=True)

    np.savetxt(os.path.join(model_dir, f"{run_name}_val_array.txt"), val_array)
    np.savetxt(os.path.join(model_dir, f"{run_name}_val_dev.txt"), val_dev)
    np.savetxt(os.path.join(model_dir, f"{run_name}_val_ids.txt"), np.asarray(val_ids))

    # PRETRAIN
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=280,
            decay_rate=0.9,
            staircase=True
        )
    (
        (losses_s_v_train, losses_neg_v_train, losses_c_train, losses_drop_c_train),
        (losses_s_v_val, losses_neg_v_val, losses_c_val, losses_drop_c_val)
    ) = model.train_model(
            mdl, train_array, val_array=val_array,
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            num_epochs=num_epochs, batch_size=batch_size, patience=patience,
            verbose=True,
            train_test_split=train_test_split,
            x_fold=x_fold
        )

    # TRAIN MODEL
    for i in range(N_iter):
        (
            (losses_s_v_train, losses_neg_v_train, losses_c_train, losses_drop_c_train),
            (losses_s_v_val, losses_neg_v_val, losses_c_val, losses_drop_c_val)
        ) = model.train_model(
            mdl, train_array, val_array=val_array,
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            num_epochs=num_epochs, batch_size=batch_size, patience=patience,
            verbose=True,
            train_test_split=train_test_split,
            x_fold=x_fold
        )
        mdl_name = os.path.join(model_dir, f"{run_name}_{i}")
        mdl.save_model(model_name=mdl_name, verbose=True)
        

    os.makedirs(figure_dir, exist_ok=True)

    plot.plot_loss('Training S_v', losses_s_v_train, num_epochs, save=figure_dir)
    plot.plot_loss('Training Neg_v', losses_neg_v_train, num_epochs, save=figure_dir)
    plot.plot_loss('Training C', losses_c_train, num_epochs, save=figure_dir)
    plot.plot_loss('Training Drop_c', losses_drop_c_train, num_epochs, save=figure_dir)

    if x_fold > 1:
        plot.plot_loss('Validation S_v', losses_s_v_val, num_epochs, save=figure_dir)
        plot.plot_loss('Validation Neg_v', losses_neg_v_val, num_epochs, save=figure_dir)
        plot.plot_loss('Validation C', losses_c_val, num_epochs, save=figure_dir)
        plot.plot_loss('Validation Drop_c', losses_drop_c_val, num_epochs, save=figure_dir)
