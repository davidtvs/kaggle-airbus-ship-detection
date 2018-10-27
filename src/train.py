import torch.utils.data as data
import torchvision.transforms as transforms
import utils
from data.airbus import AirbusShipDataset

root_dir = "/media/davidtvs/Storage/Datasets/airbus-ship-detection"

# Run only if this module is being run directly
if __name__ == "__main__":
    dataset = AirbusShipDataset(
        root_dir,
        transform=transforms.ToTensor(),
        target_transform=transforms.ToTensor(),
    )
    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for images, targets in dataloader:
        utils.imshow_batch(images, targets)
