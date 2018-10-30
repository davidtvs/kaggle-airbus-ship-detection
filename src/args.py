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
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help=(
            "Number of epochs with no improvement after which learning rate is reduced."
            " Default: 5"
        ),
    )
    parser.add_argument(
        "--stop-patience",
        type=int,
        default=5,
        help=(
            "(Early stopping) Number of epochs with no improvement after which training"
            " is stopped. Default: 20"
        ),
    )

    # Dataset
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/media/davidtvs/Storage/Datasets/airbus-ship-detection/small",
        help="Path to the root directory of the selected dataset",
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
        type=str,
        default="cuda",
        help="Device to use for computation. E.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--dataset-info",
        action="store_true",
        help="Prints information about the datasets and shows a random batch of images",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="model",
        help="Name given to the model when saving. Default: ENet",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="The directory where models are saved. Default: save",
    )

    return parser.parse_args()
