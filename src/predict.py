import time
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import utils
from tqdm import tqdm
from args import get_predict_args
from data.airbus import AirbusShipDataset
import models.ship_noship as sns


def predict(model, dataloader, device):
    device = torch.device(device)
    model = model.to(device).eval()

    # Get the current time to know how much time it took to make the predictions
    since = time.time()
    predictions = []
    for step, (images, _) in enumerate(tqdm(dataloader)):
        images = images.to(device)

        # We don't want to compute gradients, deactivate the autograd engine, this also
        # saves a lot of memory
        with torch.no_grad():
            # Do a froward pass with the images and apply the sigmoid function to get
            # the prediction
            logits = model(images)
            # Note: Because gradients are not computed there is no need to detach from
            # the graph
            pred = torch.sigmoid(logits).round_().cpu().numpy()

        predictions.append(pred)

    time_elapsed = time.time() - since
    print(
        "Predictions complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return predictions


if __name__ == "__main__":
    # Get arguments from the command-line and json configuration
    args = get_predict_args()
    config = utils.load_config(args.config)

    num_classes = 1
    input_dim = (config["img_h"], config["img_w"])
    checkpoint_path = config["model_checkpoint"]

    if checkpoint_path is None:
        raise ValueError("model checkpoint hasn't been specified")
    if not os.path.isfile(checkpoint_path):
        raise ValueError("the model checkpoint doesn't exist")

    # Compose the image transforms to be applied to the data
    image_transform = transforms.Compose(
        [transforms.Resize(input_dim), transforms.ToTensor()]
    )
    target_transform = transforms.Compose(
        [transforms.Resize(input_dim), transforms.ToTensor()]
    )

    # Initialize the dataset in test mode
    print("Loading training dataset...")
    testset = AirbusShipDataset(
        config["dataset_dir"],
        mode="test",
        transform=image_transform,
        target_transform=target_transform,
    )
    test_loader = data.DataLoader(
        testset, batch_size=config["batch_size"], num_workers=config["workers"]
    )
    if config["dataset_info"]:
        utils.dataloader_info(test_loader)

    # Initialize ship or no-ship detection network and then laod the weigths
    print("Loading ship detection model...")
    snsnet = sns.resnet(config["resnet_size"], num_classes)

    print("Loading model weights from {}...".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    snsnet.load_state_dict(checkpoint["model"])

    print()
    print("Generating predictions...")
    predictions = predict(snsnet, test_loader, config["device"])
