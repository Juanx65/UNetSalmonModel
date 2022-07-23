# UNetSalmonModel 
## Implementation of deep learning framework Unet using Keras

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

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

To create the labels for the UNet, we use program present in the folder `data_preparation`, where we use an adaptative threshold to binarise the dots, and then we make a manual finetune of the binary results.

The original dataset is from [isbi challenge](http://brainiac2.mit.edu/isbi_challenge/).

### Data augmentation

The data for training contains 30 512*512 images, which are far not enough to feed a deep learning neural network. We use a module called ImageDataGenerator in `keras.preprocessing.image` to do data augmentation.

See data.py for detail.

# MODEL

![img/u-net-architecture.png](img/u-net-architecture.png)

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is a 512*512 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.

# TEST
To test the model, use the following example as a guide:
```
python test.py --data_dir 'rois_tests/' --weights '/checkpoints/best.ckpt'
```
Where `--data_dir` is the path to the dataset to test and  `--weights` is the path to the checkpoints (trained weights of the model).

###### `test.py` will display the confusion matrix of a given dataset for the checkpoints of the model.

* Confusion Matrix on the training dataset:
  ![confusion matrix of training dataset.](/images_readme/conf_roi_bench.png)

* Confusion Matrix on a testing dataset:
  ![confusion matrix of test dataset.](/images_readme/conf_roi_tests_bench.png)


# TRAIN

The model is trained for 5 epochs.

After 5 epochs, calculated accuracy is about 0.97.

Loss function for the training is basically just a binary crossentropy.
