# UNetSalmonModel 
## Implementation of deep learning framework Unet using Keras

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

Results: `IPD44_Informe_Proyecto.pdf
---

# INSTALLATION:

## Using `virtualenv` in linux or windows

First, create an environment using `virtualenv` once inside the Folder of the repository as follows:

```
 virtualenv env
```

where, `env` is the name of the environment.
Enter the environment as follows:

```
 source env/bin/activate
```
Then install the requirements:

```
pip install -r requirements.txt
```

# DATASET

For this project we use the ROI images from our previous work in [ConvSalmonModel](https://github.com/Juanx65/ConvSalmonModel), in which we extract a section of the tarject salmon where the dots are more likely to be present and detected. You can find the dataset in folder `data/salmones`.

To create the labels for the UNet, we use the program present in the folder `data_preparation`, where we use an adaptative threshold to binarise the dots, and then we make a manual finetune of the binary results.

### Data augmentation

The data for training contains 30 256*256 images, which are far not enough to feed a deep learning neural network. We use a module called ImageDataGenerator in `keras.preprocessing.image` to do data augmentation.

See data.py for detail.

# MODEL

![img/u-net-architecture.png](img/u-net-architecture.png)

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is a 256*256 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.

# TEST
To test the model, use the following example as a guide:
```
python eval.py --data data/salmones/test_all --results /results --weights /weights/best.hdf5
```
Where `--data` is the path to the dataset to test and  `--weights` is the path to the checkpoints (trained weights of the model).

###### `test.py` will display the confusion matrix of a given dataset for the checkpoints of the model.

* Confusion Matrix on the training dataset:
  ![confusion matrix of training dataset.](/imag/conf.png)


# TRAIN

To train the model, use the following example as a guide:

```
python train.py --epochs 10 --step_per_epoch 300 --data data/salmones/train --save weights/best.hdf5
```

Where --save saves best.hdf5 in the weights folder, monitoring the loss of the model.
