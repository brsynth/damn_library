import argparse
from damn.train import train_damn

def main():
    parser = argparse.ArgumentParser(description="DAMN Model Trainer")

    parser.add_argument("--organism", choices=["ecoli", "putida", "custom"], default="custom")

    parser.add_argument("--train-test-split", type=str)
    parser.add_argument("--file-name", type=str)
    parser.add_argument("--media-file", type=str)
    parser.add_argument("--cobra-model-file", type=str)
    parser.add_argument("--biomass-rxn-id", type=str)

    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-epochs", type=int)

    parser.add_argument("--loss-weight", nargs=4, type=float)
    parser.add_argument("--loss-decay", nargs=4, type=float)

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--x-fold", type=int)
    parser.add_argument("--n-iter", type=int)

    args = parser.parse_args()

    train_damn(
        organism=args.organism,
        train_test_split=args.train_test_split,
        file_name=args.file_name,
        media_file=args.media_file,
        cobra_model_file=args.cobra_model_file,
        biomass_rxn_id=args.biomass_rxn_id,
        seed=args.seed,
        num_epochs=args.num_epochs,
        loss_weight=args.loss_weight,
        loss_decay=args.loss_decay,
        batch_size=args.batch_size,
        x_fold=args.x_fold,
        N_iter=args.n_iter,
    )
