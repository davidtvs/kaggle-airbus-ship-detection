from argparse import ArgumentParser


def get_arguments():
    """Defines command-line arguments, and parses them.

    """
    parser = ArgumentParser()

    # Hyperparameters
    parser.add_argument(
        "--batch-size", "-b", type=int, default=10, help="The batch size. Default: 10"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs. Default: 20"
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=0.001,
        help="The learning rate. Default: 0.001",
    )

    # Settings
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. Default: 4",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use for computation. E.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--dataset-info",
        action="store_true",
        help="Prints information about the datasets and shows a random batch of images",
    )

    return parser.parse_args()
