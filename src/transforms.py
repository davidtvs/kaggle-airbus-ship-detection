import torch
import torchvision.transforms as transforms


class TargetHasShipTensor(object):
    def __call__(self, image):
        tensor_img = transforms.ToTensor()(image)
        tensor_sum = tensor_img.sum()
        return tensor_sum.gt(1).float().unsqueeze_(-1)


class ToLongTensor(object):
    def __call__(self, image):
        float_tensor = transforms.ToTensor()(image)
        return float_tensor.long().squeeze()


class Threshold(object):
    def __init__(self, threshold=0.5, high=1, low=0):
        self.threshold = torch.Tensor([threshold])
        self.high = torch.Tensor([high])
        self.low = torch.Tensor([low])

    def __call__(self, tensor):
        return torch.where(tensor > self.threshold, self.high, self.low)
