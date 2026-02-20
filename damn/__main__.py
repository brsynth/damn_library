import os
import sys
import shutil
import argparse
import zipfile
import tempfile
import argparse
import numpy as np
from damn.train import train_damn
from damn.predict import test_damn
from damn import plot

def damn_train(argv=None):
    parser = argparse.ArgumentParser(description="DAMN Model Trainer",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--organism", choices=["ecoli", "putida", "custom"], default="custom",
                        help="model type")

    parser.add_argument("--file-name", type=str,
                        help="Training dataset file")
    parser.add_argument("--media-file", type=str,
                        help="Media definition file")
    parser.add_argument("--od-file", type=str,
                        help="OD file")
    parser.add_argument("--cobra-model-file", type=str,
                        help="COBRA metabolic model file")
    parser.add_argument("--biomass-rxn-id", type=str,
                        help="Biomass reaction ID")

    parser.add_argument("--train-test-split", type=str,
                        help="Train/test split ratio (e.g., 0.8)")
    parser.add_argument("--seed", type=int,
                        help="Random seed")
    parser.add_argument("--num-epochs", type=int,
                        help="Number of training epochs")
    parser.add_argument("--loss-weight", nargs=4, type=float,
                        help="Loss weights (4 values)")
    parser.add_argument("--loss-decay", nargs=4, type=float,
                        help="Loss decay values (4 values)")

    parser.add_argument("--hidden-layers-lag", type=int, nargs="+", default=[50],
                        help="Hidden layer sizes for lag network (e.g. 50 50)")
    parser.add_argument("--hidden-layers-flux", type=int, nargs="+", default=[500],
                        help="Hidden layer sizes for flux network (e.g. 500 200)")
    parser.add_argument("--dropout-rate", type=float, default=0.2,
                        help="Dropout Rate")
    parser.add_argument("--x-fold", type=int, default=3,
                        help="Number of folds for cross-validation")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Training batch size")
    parser.add_argument("--patience", type=int, default=100,
                        help="Patience")
    parser.add_argument("--n-iter", type=int, default=3,
                        help="Number of training iterations")

    parser.add_argument("--modul-dir",type=str,default=os.path.join(os.getcwd(), "modul"),
        help="Directory to save outputs (default: ./modul in working directory)") 
    parser.add_argument("--figure-dir",type=str,default=os.path.join(os.getcwd(), "figure"),
        help="Directory to save outputs (default: ./figure in working directory)")

    args = parser.parse_args(argv)

    # create directiries
    model_dir = args.modul_dir
    figure_dir = args.figure_dir
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    (
        mdl,
        models,
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
        organism=args.organism,
        train_test_split=args.train_test_split,
        file_name=args.file_name,
        od_file=args.od_file,
        media_file=args.media_file,
        cobra_model_file=args.cobra_model_file,
        biomass_rxn_id=args.biomass_rxn_id,
        seed=args.seed,
        num_epochs=args.num_epochs,
        loss_weight=args.loss_weight,
        loss_decay=args.loss_decay,
        hidden_layers_lag=args.hidden_layers_lag,
        hidden_layers_flux=args.hidden_layers_flux,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        x_fold=args.x_fold,
        patience=args.patience,
        N_iter=args.n_iter,
    )

    # Save validation arrays
    np.savetxt(os.path.join(model_dir, f"{run_name}_val_array.txt"), val_array)
    np.savetxt(os.path.join(model_dir, f"{run_name}_val_dev.txt"), val_dev)
    np.savetxt(os.path.join(model_dir, f"{run_name}_val_ids.txt"), np.asarray(val_ids))

    # Save model
    for name, mdl in models.items():
        mdl.save_model(
            model_name=os.path.join(model_dir, name),
            verbose=True
        )

    #mdl_name = os.path.join(model_dir, run_name)
    #mdl.save_model(model_name=mdl_name, verbose=True)

    #Zip model
    zip_file_path = f"{model_dir}.zip"
    shutil.make_archive(base_name=model_dir, format='zip', root_dir=model_dir)

    #Plot last run
    if args.organism == "putida":
        if args.num_epochs is not None:
            plot_epochs = args.num_epochs
        else:
            plot_epochs = 500
    elif args.organism == "ecoli":
        if args.num_epochs is not None:
            plot_epochs = args.num_epochs
        else:
            plot_epochs = 1000
    else:
        plot_epochs = 500

    plot.plot_loss('Training S_v', losses_s_v_train, num_epochs=plot_epochs, save=figure_dir)
    plot.plot_loss('Training Neg_v', losses_neg_v_train, num_epochs=plot_epochs, save=figure_dir)
    plot.plot_loss('Training C', losses_c_train, num_epochs=plot_epochs, save=figure_dir)
    plot.plot_loss('Training Drop_c', losses_drop_c_train, num_epochs=plot_epochs, save=figure_dir)
    if args.x_fold > 1:
        plot.plot_loss('Validation S_v', losses_s_v_val, num_epochs=plot_epochs, save=figure_dir)
        plot.plot_loss('Validation Neg_v', losses_neg_v_val, num_epochs=plot_epochs, save=figure_dir)
        plot.plot_loss('Validation C', losses_c_val, num_epochs=plot_epochs, save=figure_dir)
        plot.plot_loss('Validation Drop_c', losses_drop_c_val, num_epochs=plot_epochs, save=figure_dir)

    return zip_file_path , figure_dir


def damn_predict(argv=None):
    parser = argparse.ArgumentParser(description="DAMN Model prediction",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--organism", choices=["ecoli", "putida", "custom"], default="custom",
                        help="model type")

    parser.add_argument("--file-name", type=str,
                        help="Training dataset file")
    parser.add_argument("--metabolite-ids", type=str, nargs="+",
                        help="Metabolites ids")

    parser.add_argument("--train-test-split", type=str,
                        help="Train/test split ratio (e.g., 0.8)")
    parser.add_argument("--n-iter", type=int, default=3,
                        help="Number of training iterations")
    parser.add_argument("--ods", type=lambda x: x.lower() == "true", default=True,
                        help="when True biomass concentration transformed in OD")
    parser.add_argument("--plot-type", choices=["growth", "substrate"], type=str,
                        help="plot type")

    parser.add_argument("--r2min", type=float, default=0,
                        help="range you want the plots to be printed min")
    parser.add_argument("--r2max", type=float, default=0.5,
                        help="range you want the plots to be printed max")

    parser.add_argument("--model-dir", type=str,
                        help="Zip files contains models in h5 and json format") 
    parser.add_argument("--figure-dir",type=str,default=os.path.join(os.getcwd(), "figure"),
                        help="Directory to save outputs (default: ./figure in working directory)")

    args = parser.parse_args(argv)

    #UnZip model
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(args.model_dir) as z:
            z.extractall(tmp_dir)


        (
            Pred, 
            Ref, 
            R2, 
            R2dev, 
            times, 
            val_dev, 
            val_ids , 
            mdl,  
            run_name, 
            train_test_split, 
            plot_type, 
            OD, 
            R2min, 
            R2max
        ) = test_damn(
            organism=args.organism,
            train_test_split=args.train_test_split,
            file_name=args.file_name,
            OD=args.ods,
            plot_type=args.plot_type,
            N_iter=args.n_iter,
            R2min=args.r2min,
            R2max=args.r2max,
            model_dir=tmp_dir
        )


    # Plot last run
    figure_dir = args.figure_dir
    os.makedirs(figure_dir, exist_ok=True)

    title = f"R2 Histogram {train_test_split}"
    plot.plot_similarity_distribution(title, R2, save=figure_dir)

    if plot_type == "growth":
        plot.plot_predicted_reference_growth_curve(
            times=times,
            Pred=Pred,
            Ref=Ref,
            val_dev=val_dev,
            OD=OD,
            R2=R2,
            R2dev=R2dev,
            train_time_steps=getattr(mdl, "train_time_steps", 0),
            experiment_ids=list(val_ids),
            run_name=run_name,
            train_test_split=train_test_split,
            R2min=R2min,
            R2max=R2max,
            save=figure_dir
        )

    elif plot_type == "substrate":
        plot.plot_predicted_biomass_and_substrate(
            times,
            Pred,
            experiment_ids=list(val_ids),
            metabolite_ids=list(mdl.metabolite_ids),
            run_name=run_name,
            train_test_split=train_test_split,
            save=figure_dir
        )

def main():
    parser = argparse.ArgumentParser(prog="damn", description="DAMN CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train DAMN model")
    subparsers.add_parser("predict", help="Prediction DAMN model")
    args = parser.parse_args(sys.argv[1:2])

    # Dispatch
    if args.command == "train":
        damn_train(sys.argv[2:])
    elif args.command == "predict":
        damn_predict(sys.argv[2:])

if __name__ == "__main__":
    main()
