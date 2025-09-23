import argparse
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def centralized_args_parser():
    """
    Parse arguments to define hyperparameter settings for centralized training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed used for train/test splitting (default = 42).",
    )
    parser.add_argument(
        "--test-fraction",
        default=0.2,
        type=float,
        help="Test fraction for train/test splitting (default = 0.2).",
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="Number of training epochs (default = 20).",
    )
    parser.add_argument(
        "--learning-rate",
        default=0.003,
        type=float,
        help="Learning rate (default = 0.003).",
    )
    parser.add_argument(
        "--batch-divisor",
        default=5,
        type=int,
        help="Divisor to determine batch size (default = 5).",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.0001,
        type=float,
        help="Weight decay constant (default = 0.0001).",
    )
    parser.add_argument(
        "--data-partitions-file",
        default=None,
        type=str,
        help=f"Path to the data partitions file (default = {None}).\n"
        "If not used, then the data is split into n_models equal parts."
        "If data-partitions file is provided, "
        "then the number of models (n_models) "
        "to train is equal to the number of "
        "data partitions available in the file. In the data partitions, "
        "each key is a client id and value for each key is a list of "
        "dataset indices to be used for that client.",
    )
    parser.add_argument(
        "--n-models",
        default=1,
        type=int,
        help="Number of models to train (default = 1). "
        "Data is split into n_models equal parts."
        "This is used when the clustered-indices file is not found.",
    )
    args = parser.parse_args()
    return args


def server_args_parser():
    """
    Parse arguments to define hyperparameter settings for centralized training.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-rounds",
        default=3,
        type=int,
        help="Number of training rounds (default = 3).",
    )
    parser.add_argument(
        "--min-fit-clients",
        default=1,
        type=int,
        help="Minimum number of fit clients (default = 1).",
    )
    parser.add_argument(
        "--min-evaluate-clients",
        default=1,
        type=int,
        help="Minimum number of evaluation clients (default = 1).",
    )
    parser.add_argument(
        "--min-available-clients",
        default=1,
        type=int,
        help="Minimum number of available clients (default = 1).",
    )

    args = parser.parse_args()
    return args


def client_args_parser():
    """
    Parse arguments to define experimental settings on client side.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-partitions",
        default=1,
        type=int,
        help="Number of partitions (default = 1).",
    )
    parser.add_argument(
        "--client-id",
        default=0,
        type=int,
        help="client ID used for the current client (default = 0).",
    )
    parser.add_argument(
        "--partitioner-type",
        default="uniform",
        type=str,
        choices=["uniform", "linear", "square", "exponential"],
        help="Partitioner types (default = 'uniform').",
    )
    parser.add_argument(
        "--partition-id",
        default=0,
        type=int,
        help="Partition ID used for the current client (default = 0).",
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="Number of training epochs (default = 20).",
    )
    parser.add_argument(
        "--batch-divisor",
        default=5,
        type=int,
        help="Divisor to determine batch size (default = 5).",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed used for train/test splitting (default = 42).",
    )
    parser.add_argument(
        "--test-fraction",
        default=0.2,
        type=float,
        help="Test fraction for train/test splitting (default = 0.2).",
    )
    parser.add_argument(
        "--learning-rate",
        default=0.003,
        type=float,
        help="Learning rate (default = 0.003).",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.0001,
        type=float,
        help="Weight decay constant (default = 0.0001).",
    )
    parser.add_argument(
        "--data-partitions-file",
        default=None,
        type=str,
        help=f"Path to the data partitions file (default = {None}).\n"
        "If not used, then the data is split into n_models equal parts."
        "If data-partitions file is provided, "
        "then the number of models (n_models) "
        "to train is equal to the number of "
        "data partitions available in the file. In the data partitions, "
        "each key is a client id and value for each key is a list of "
        "dataset indices to be used for that client.",
    )
    parser.add_argument(
        "--n-models",
        default=1,
        type=int,
        help="Number of models to train (default = 1). "
        "Data is split into n_models equal parts."
        "This is used when the clustered-indices file is not found.",
    )

    args = parser.parse_args()
    return args
