# Exploring public kernels

This is a compilation of the best public kernels available at this time for the [`Airbus Ship Detection Challenge`](https://www.kaggle.com/c/airbus-ship-detection/overview) competition. The goal is to understand what other competitors have already done and their main findings.

Sorting the list of public kernels by `Best score` we find the following kernels:

* [Unet34 submission (0.89 public LB)](https://www.kaggle.com/iafoss/unet34-submission-0-89-public-lb), [Unet34 (dice 0.87+)](https://www.kaggle.com/iafoss/unet34-dice-0-87), and [Fine-tuning ResNet34 on ship detection](https://www.kaggle.com/iafoss/fine-tuning-resnet34-on-ship-detection) - 89.2%
* [Classification and Segmentation (-FP)](https://www.kaggle.com/hmendonca/classification-and-segmentation-fp) - 85.3%
* [Basic Modeling](https://www.kaggle.com/shubhammank/basic-modeling/output) - 84.7%
* [U-Net Model with submission](https://www.kaggle.com/hmendonca/u-net-model-with-submission) - 84.7%
* [One line Base Model LB:0.847](https://www.kaggle.com/paulorzp/one-line-base-model-lb-0-847) - 84.7%

The top score of public kernels belongs to Iafoss, his series of three public kernels all build on each other and we'll look into them with more detail. Another important kernel is *One line Base Model LB:0.847* by Paulo Pinto; it simply outputs that there are no ships for all images and achieves a 84.7% score. This is a baseline, which means that a model is only useful if it scores higher than 84.7%.

## Unet34 submission (0.89 public LB), Unet34 (dice 0.87+), and Fine-tuning ResNet34 on ship detection

The author, Iafoss, starts by pointing out two types of network architectures that are common ways of solving similar problems - U-net and SSD - and evaluates each of them in terms of advantages and disadvantages.

### Advantages and disadvantages of U-net

:heavy_check_mark: Any state-of-the-art U-net model can be used.  
:heavy_check_mark: Allows usage of pre-trained weights.  
:x: Requires post-processing to mask each ship individually, specially difficult there is overlap.  
:x: The ground-truth bounding boxes are pixelated, i.e., they do not follow the ship contour exactly. The segmentation produced by U-net networks tries to follow the ship contour down to the pixel. This mismatch leads to lower scores.  

### Advantages and disadvantages of SSD

:heavy_check_mark: Outputs bounding boxes around the ship that can be filled yielding pixelated segmentations.  
:heavy_check_mark: Each ship gets an unique bounding box.  
:x: The bounding boxes must be able to rotate, which is not common for this type of network. Therefore, we would have to build a new model with a new loss function and train it from scratch.  

### Model

The author states that an SSD architecture is likely to yield better results but building the network, loss function, and training the model from scratch takes a considerable ammount of effort and time. Thus, he chooses to go with a U-net architecture.

The model is actually comprised of two stacked models:

* The first model is a standard ResNet34 which predicsts if the image has a ship or not
* The second model takes images that contain ships as input and segments each ship. The network is comprised of a ResNet34 encoder and an upsampling decoder with skip connections between downsampling and upsampling layers.

### Loss function

A custom loss function is also employed: $loss = 10 \times focal\_loss - \log(soft\_dice)$. Focal loss is typically used in imbalanced segmentation problems which fits this competition since the ratio of pixels that belong to a ship to total pixels is $\approx$ 1:1000 for images that contain ships and $\approx$ 1:10000 for the whole dataset. The $- \log(soft\_dice)$ is included to increase the loss when ships are not detected correctly and dice is close to zero. It lowers the amount of false positives and incorrect segmentations in images that contain a single ship. To keep both terms within the same order of magnitude, the focal loss is multiplied by 10.

### Training

#### Ship detection

1. Training the head layers on 256$\times$256 images for one epoch yields 93.7% accuracy.
2. Training the entire model on 256$\times$256 images for two more epochs yields 97% accuracy.
3. Training the entire model on 384$\times$384 images for several more epochs further increases the accuracy to 98%.

#### Ship segmentation

1. Training only the decoder on 256$\times$256 images for one epoch yields a dice coefficient of $\approx 0.8$.
2. Training the entire model on 256$\times$256 images for six additional epochs yields a dice coefficient of $\approx$ 0.86.
3. Training the entire model on 384$\times$384 images for 12 additional epochs yields a dice coefficient of $\approx$ 0.89.
4. Training the entire model on 768$\times$768 images for several more epochs yields a dice coefficient of $\approx$ 0.90.
