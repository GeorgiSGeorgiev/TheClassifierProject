"""EfficientNetB0 fine-tuning script

This script directly starts the process of EfficientNetB0 fine-tuning. At the beginning of the script the model is
being initialized, then its pretrained version is being loaded and in the end the fine-tuning takes place.

Can be started directly and doesn't take any arguments. All the settings are directly inside the script.

"""
# Created by: Georgi Stoyanov Georgiev.
# as part of the "Neural network architectures for mobile devices" bachelor thesis

import os   # used to import the checkpoint path to the program
import dataset_loader       # our script that loads our dataset
import graph_plotter        # our script that creates graphs
import model_controller     # our script used just to unfreeze the last layer of the model
import tensorflow as tf     # TensorFlow 2
from init_efficientnetB0 import build_model     # our scripts that builds the basic model
from tensorflow.keras.layers.experimental import preprocessing  # used to build the input augmentation layer

# The buffer size will be dynamically tuned. Can be changed by the programmer.
# Used in the prefetch setting below.
AUTOTUNE = tf.data.AUTOTUNE

tf.random.set_seed(9)   # TF 2 random seed. Used to replicate the training results. Works perfectly.

# Inspiration source:
# https://www.tensorflow.org/tutorials/load_data/images
# This is the main TF tutorial.

# Please set the path to the location where you saved the dataset photos.
# It has to point directly to the folder where are located all the different car types folders.
dataset_path = "D:\\tmp\\cars4\\car_photos_4"

img_dim = 224   # The EfficientNetB0 supports only input images with the size of 224x224. So don't change this value.
# Inside the model_controller class it is guaranteed that when creating an instance the user can not change
# the image dimensions.
IMG_SIZE = (img_dim, img_dim)   # input dimensions of the images
batch_size = 64     # the size of each batch full with dataset images
# Batches are being used to accelerate the training process of the model.

# Load the dataset into two disjoint sets: one training and one validation set
train_ds, val_ds = dataset_loader.get_dataset(dataset_path, IMG_SIZE, batch_size)

# Get the number of classes directly from the dataset folder structure.
NUM_CLASSES = len(train_ds.class_names)

# Prefetch allows to do background tasks while loading the input tensors to the model.
# This increases the training speeds.
train_dataset = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_dataset = val_ds.prefetch(buffer_size=AUTOTUNE)

# Cars dataset is not a very big dataset so data augmentation is recommended to reduce the overfitting effect.
data_augmentation = tf.keras.Sequential(
    [
        preprocessing.RandomFlip('horizontal', seed=9),      # random horizontal flip with seed set to a constant value
        preprocessing.RandomRotation(0.15, seed=9),          # random image rotation with seed set to a const.
        preprocessing.RandomContrast(factor=0.9, seed=9),    # independently adjust the contrast on each channel
        preprocessing.RandomZoom(0.25, seed=9),              # random image zoom
        preprocessing.RandomTranslation(height_factor=0.15, width_factor=0.15, seed=9),
    ],
    name="img_augmentation")   # The name of the layer. Later we can use it as a key to get the layer from the network.

# Build the basic version of the model and add the data augmentation layer to it.
model = build_model(num_classes=NUM_CLASSES, img_dim=img_dim, img_augmentation=data_augmentation)

# Set the training epoch of the last validation layer. It is good to train it as much as possible
# until achieving convergence.
epochs = 30  # @param {type: "slider", min:8, max:80}

# Train the last validation layer of the network.
history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, verbose=1)
# uncomment to see the training graph of the last validation layer
# graph_plotter.plot_hist(history)

# Set the path where the checkpoints will be saved. Can be changed by the user.
# The default setting will save the checkpoints directly to the directory of this script.
# Also specify the format of the checkpoint names (epoch:04d will put the last 4 digits of the epoch index
# at the end of the checkpoint name).
checkpoint_path = "efficientnetB0_training0/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs.
# The saving frequency can be adjusted as well.
saving_frequency = 5
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,  # The logging mode.
    save_weights_only=True,    # Save only the checkpoint, not the whole model. Can be adjusted by the user.
    save_freq=saving_frequency*len(train_ds))

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Unfreeze the last 60 layers of the model. Can be adjusted.
model_controller.unfreeze_model(model, 60)

# Specify the model training epochs. Be careful not to overtrain the model.
epochs = 25  # @param {type: "slider", min:8, max:50}
hist = model.fit(train_dataset, epochs=epochs, callbacks=[cp_callback], validation_data=validation_dataset, verbose=1)
graph_plotter.plot_hist(hist)   # Plot the training process of the model. It is an indicator that process finished.
