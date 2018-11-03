from argparse import ArgumentParser


def get_train_args():
    """Defines command-line arguments for training."""
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="train.json",
        help="Path to the JSON configuration file. Default: train.json",
    )
    parser.add_argument(
        "--model-checkpoint",
        "-m",
        type=str,
        help="Path to the model checkpoint to be loaded. Default: null",
    )

    return parser.parse_args()


def get_predict_args():
    """Defines command-line arguments for predictions."""
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="predict.json",
        help="Path to the JSON configuration file. Default: predict.json",
    )

    return parser.parse_args()
