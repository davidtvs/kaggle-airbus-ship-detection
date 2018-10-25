# ENet

PyTorch implementation of [*ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation*](https://arxiv.org/abs/1606.02147), ported from the lua-torch implementation [ENet-training](https://github.com/e-lab/ENet-training) created by the authors.


## Installation

1. Python 3 and pip.
2. Set up a virtual environment (optional, but recommended).
3. Install dependencies using pip: `pip install -r requirements.txt`.


## Usage

Run `main.py`, the main script file used for training and/or testing the model. The following options are supported:

```
python main.py [-h] [--mode {train,test,full}] [--resume]
               [--batch-size BATCH_SIZE] [--epochs EPOCHS]
               [--learning-rate LEARNING_RATE] [--lr-decay LR_DECAY]
               [--lr-decay-epochs LR_DECAY_EPOCHS]
               [--weight-decay WEIGHT_DECAY] [--dataset {camvid,cityscapes}]
               [--dataset-dir DATASET_DIR] [--height HEIGHT] [--width WIDTH]
               [--weighing {enet,mfb,none}] [--with-unlabeled]
               [--workers WORKERS] [--print-step] [--imshow-batch]
               [--no-cuda CUDA] [--name NAME] [--save-dir SAVE_DIR]
```

For help on the optional arguments run: `python main.py -h`


### Examples: Training

```
python main.py -m train --save-dir save/folder/ --name model_name --dataset name --dataset-dir path/root_directory/
```


### Examples: Resuming training

```
python main.py -m train --resume True --save-dir save/folder/ --name model_name --dataset name --dataset-dir path/root_directory/
```


### Examples: Testing

```
python main.py -m test --save-dir save/folder/ --name model_name --dataset name --dataset-dir path/root_directory/
```


## Project structure

### Folders

- `data`: Contains code to load the supported datasets.
- `metric`: Evaluation-related metrics.
- `models`: ENet model definition.
- `save`: By default, `main.py` will save models in this folder. The pre-trained models can also be found here.

### Files

- `args.py`: Contains all command-line options.
- `main.py`: Main script file used for training and/or testing the model.
- `test.py`: Defines the `Test` class which is responsible for testing the model.
- `train.py`: Defines the `Train` class which is responsible for training the model.
- `transforms.py`: Defines image transformations to convert an RGB image encoding classes to a `torch.LongTensor` and vice versa.
