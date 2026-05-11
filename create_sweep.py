"""Create W&B Bayesian sweeps for orthogonalization-iteration ablations."""

import argparse
import math

import wandb

PROJECT = "cifar10-airbench-1ep-sweeps"
ENTITY = "thib-s"
BAYES_RUN_CAP = 500
ORTHOGONALIZE_NUM_ITERS_VALUES = [1, 5, 10, 25, 100]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create one W&B Bayesian sweep per orthogonalize_num_iters value for the "
            "convolutional Muon ablation study."
        )
    )
    parser.add_argument(
        "--orthogonalize-num-iters",
        type=int,
        nargs="+",
        default=ORTHOGONALIZE_NUM_ITERS_VALUES,
        help=(
            "Subset of orthogonalize_num_iters values to create sweeps for. "
            f"Defaults to {ORTHOGONALIZE_NUM_ITERS_VALUES}."
        ),
    )
    parser.add_argument(
        "--project",
        type=str,
        default=PROJECT,
        help=f"W&B project name. Defaults to {PROJECT}.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=ENTITY,
        help=f"W&B entity name. Defaults to {ENTITY}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    created_sweeps = []
    for orthogonalize_num_iters in args.orthogonalize_num_iters:
        sweep_config = {
            "name": f"conv-muon-orth-iters-{orthogonalize_num_iters}",
            "method": "bayes",
            "run_cap": BAYES_RUN_CAP,
            "metric": {"name": "tta_val_acc_mean", "goal": "maximize"},
            "parameters": {
                "orthogonalize_num_iters": {"values": [orthogonalize_num_iters]},
                "muon_lr": {
                    "distribution": "log_uniform_values",
                    "min": 0.01,
                    "max": 2.0,
                },
                "muon_momentum": {
                    "distribution": "uniform",
                    "min": 0.01,
                    "max": 0.99,
                },
                "sgd_momentum": {
                    "distribution": "uniform",
                    "min": 0.01,
                    "max": 0.99,
                },
                "muon_weight_decay_scale": {"values": [0.0]},
                "adam_weight_decay_scale": {"values": [0.0]},
                "head_lr": {
                    "distribution": "log_normal",
                    "mu": math.log(0.67),
                    "sigma": 0.5,
                },
                "bias_lr": {
                    "distribution": "log_normal",
                    "mu": math.log(0.053),
                    "sigma": 0.5,
                },
                "batch_size": {"values": [250]},
                "epochs": {"values": [1]},
                "whitening_epochs": {"values": [1]},
                "no_aug_flip": {"values": [True]},
                "aug_translate": {"values": [0]},
                "orthogonalize_damp": {"values": [0.91]},
                "orthogonalize_epsilon": {"values": [0.09]},
            },
        }

        sweep_id = wandb.sweep(
            sweep_config,
            project=args.project,
            entity=args.entity,
        )
        sweep_path = f"{args.entity}/{args.project}/{sweep_id}"
        created_sweeps.append((orthogonalize_num_iters, sweep_path))
        print(
            "Created sweep for orthogonalize_num_iters="
            f"{orthogonalize_num_iters}: {sweep_path}"
        )

    if created_sweeps:
        print("Run agents with:")
        for i, (orthogonalize_num_iters, sweep_path) in enumerate(created_sweeps):
            print(
                "  "
                f"# orthogonalize_num_iters={orthogonalize_num_iters}\n"
                f"cd /home/thibaut.boissin/projects/muon_conv2/; export CUDA_VISIBLE_DEVICES={i%2}; /home/thibaut.boissin/projects/muon_conv2/.venv/bin/python /home/thibaut.boissin/projects/muon_conv2/airbench94_conv_muon.py --sweep-id {sweep_path} --count {BAYES_RUN_CAP} --batch-size 250 --epochs 1 --whitening-epochs 1 --aug-translate 0 --no-aug-flip"
            )


if __name__ == "__main__":
    main()
