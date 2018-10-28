import torchvision.transforms as transforms


class TargetHasShipTensor(object):
    def __call__(self, image):
        tensor_img = transforms.ToTensor()(image)
        tensor_sum = tensor_img.sum()
        return tensor_sum.gt(1).float().unsqueeze_(-1)
