from argparse import ArgumentParser


def train_classifier_args():
    """Defines command-line arguments for training."""
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="train_classifier.json",
        help="Path to the JSON configuration file. Default: train_classifier.json",
    )
    parser.add_argument(
        "--model-checkpoint",
        "-m",
        type=str,
        help=(
            "Path to the model checkpoint to be loaded (.pth file). When paired with "
            "the --resume flag, weights, optimizer, starting epoch, and metrics are "
            "loaded from the checkpoint. Without --resume, only the weights are "
            "loaded. Default: null"
        ),
    )
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help=(
            "When this flag is specified together with --model_checkpoint, training "
            "is continued from the checkpoint as if it was never stopped. This flag is "
            "ignored is --model-checkpoint is not specified"
        ),
    )

    return parser.parse_args()


def segmentation_dataset_args():
    """Defines command-line arguments for predictions."""
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="segmentation_dataset.json",
        help="Path to the JSON configuration file. Default: segmentation_dataset.json",
    )

    return parser.parse_args()
