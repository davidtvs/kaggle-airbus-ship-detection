import time
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import utils
from tqdm import tqdm
from args import get_predict_args
from data.airbus import AirbusShipDataset
import models.net as models


def predict(model, dataloader, device):
    device = torch.device(device)
    model = model.to(device).eval()

    # Get the current time to know how much time it took to make the predictions
    since = time.time()
    predictions = []
    for step, (images, _) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        logits = model(images)

        # Apply the sigmoid function to get the prediction from the logits
        pred = torch.sigmoid(logits).detach().round_().cpu().numpy()
        predictions.append(pred)

    time_elapsed = time.time() - since
    print(
        "Predictions complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return predictions


if __name__ == "__main__":
    # Get arguments from the command-line
    args = get_predict_args()

    if args.model_checkpoint is None:
        raise ValueError("specify a model checkpoint with --model-checkpoint or -m")
    if not os.path.isfile(args.model_checkpoint):
        raise ValueError("the model checkpoint doesn't exist")

    num_classes = 1
    input_dim = 224

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
        args.dataset_dir,
        mode="test",
        transform=image_transform,
        target_transform=target_transform,
    )
    test_loader = data.DataLoader(
        testset, batch_size=args.batch_size, num_workers=args.workers
    )
    if args.dataset_info:
        utils.dataloader_info(test_loader)

    # Initialize ship or no-ship detection network and then laod the weigths
    print()
    print("Loading ship detection model...")
    snsnet = models.resnet_snsnet(34, num_classes)
    print(snsnet)

    print()
    print("Loading model weights from {}...".format(args.model_checkpoint))
    checkpoint = torch.load(args.model_checkpoint)
    snsnet.load_state_dict(checkpoint["state_dict"])

    print()
    print("Generating predictions...")
    predictions = predict(snsnet, test_loader, args.device)
