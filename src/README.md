# ResNet34 Classifier and UNet34, 479th place (0.83313)

The top-scoring model was a combination of two models:

1. ship or no-ship classification ResNet34 replacing the last pooling layer and fully connected layer with an adaptive head based on the [fast.ai ResNet models](https://github.com/fastai/fastai/blob/1ad3caafc123cb35fea8b63fee3b82301310207b/fastai/vision/learner.py#L33) and [AdaptiveConcatPool2d](https://github.com/fastai/fastai/blob/14c02c2009af212e5030ff0f777246826ed4f9dc/fastai/layers.py#L61).

2. segmentation network based on [LinkNet](https://arxiv.org/abs/1707.03718) with the following differences:
    - the ResNet18 encoder was replaced by a ResNet34;
    - the kernel size of the last transposed convolution in the final block was changed from 2 to 3. This made it possible for the network to accept any input image size without changing the output padding.

The ship or no-ship model predicts which images have ships. Images with ships are then passed to the segmentation network that performs pixel-wise ship segmentation.

The classifier model was trained using binary cross entropy loss with logits ([`BCEWithLogitsLoss`](https://pytorch.org/docs/stable/nn.html?highlight=bcewithlogits#torch.nn.BCEWithLogitsLoss)) and the segmentation model was trained with binary cross entropy and dice with logits ([`BCE_BDWithLogitsLoss`](https://github.com/davidtvs/airbus-ship-detection/blob/master/src/models/loss.py#L293))

## Usage and replicating results

The results can be replicated by following these steps:

1. `python train_classifier.py -c config/train_clf_224.json`
2. `python train_classifier.py -c config/train_clf_384.json -m checkpoints/sns34_i224/sns34_i224.pth`
3. `python train_classifier.py -c config/train_clf_768.json -m checkpoints/sns34_i384/sns34_i384.pth`

At this point, the classifier is trained. The next step is to generate the dataset for training with the segmentation network. We could use the classifier during the segmentation training to generate the dataset in real-time but that is computationally expensive and not time efficient. The `segmentation_dataset.py` script will write the training images to a CSV instead.

4. `python segmentation_dataset.py -c config/segmentation_dataset.json`

To train the segmentation network:

5. `python train_seg_binary.py -c config/train_seg_384.json`
6. `python train_seg_binary.py -c config/train_seg_768.json -m checkpoints/linknet34_fscore_bce_bdl_i384/linknet34_fscore_bce_bdl_i384.pth`

Finally, the models are combined with [`ComboNet`](https://github.com/davidtvs/airbus-ship-detection/blob/master/src/models/combonet.py) and the submission is created:

7. `python make_submission.py -c config/make_submission.json`

## What didn't work

- ENet gave lower performance
- Standard LinkNet (with ResNet18) performed slightly worse
- [UNet by lyakaap](https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution) took too much time to train.
- Ground-truth ship segmentations weren't tight, i.e. they were more akin to oriented bounding boxes. Post-processing was applied to predictions to compute the oriented bounding box of each ship and fill it completely. This resulted in much worse performance.
