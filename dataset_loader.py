# This script only defines some functions which load the requested dataset or its class labels.

# First set all of the random number generator seeds. Some seeds may be calculated using the import time of the
# libraries. That's why we are setting the seeds right at the beginning.
import os
os.environ['PYTHONHASHSEED'] = str(9)
# Set the Python hash seed to a constant. The Python hashing function can be used by other random number generators.

# Set the numpy seed. Numpy library allows to convert tensors to numpy arrays which resemble normal arrays.
from numpy.random import seed
seed(9)
# Import the TensorFlow 2 and set its seed.
import tensorflow as tf
tf.random.set_seed(9)

# Import libraries which work with system paths.
import os
import os.path as pth
import pathlib

# Followed directly this tutorial:  https://www.tensorflow.org/tutorials/load_data/images


def get_dataset(dataset_path, image_dimensions, batch_size):
    """Loads the requested dataset from the given directory into a training set and a validation set.

        Parameters
            ----------
            dataset_path : str
                The system path to the dataset in the form of a string.
            image_dimensions : (int,int)
                Input image dimensions to be loaded into the model.
            batch_size : int
                Size of the batches into which the images will be loaded.
        Returns
            ----------
            (train_ds, val_ds) : tuple `(`tf.data.Dataset`, `tf.data.Dataset`)`
                Each dataset is a tuple `(images, labels)`, where `images` has shape
                `(batch_size, image_size[0], image_size[1], num_channels)`,
                and `labels` are an array of integers from 0 to (number of classes - 1).
    """
    # Get the class names directly from the folder structure of the dataset.
    class_names = list([name for name in os.listdir(dataset_path) if pth.isdir(pth.join(dataset_path, name))])
    # Change the format of the string path to a "pathlib.Path".
    # This is the only path format TF function we are using works with.
    data_dir = pathlib.Path(dataset_path)
    # Get and print the total number of the dataset images. Serves as a control check.
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(f"Total image count: {image_count}")

    # Load the training data.
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,                       # The dataset path in the "pathlib.Path" format.
        validation_split=0.2,           # Training-validation ratio.
        subset="training",              # Type of the currently loaded images.
        label_mode="categorical",       # Important, determines if the output will be binary or not.
        seed=123,                       # Random data split seed used for reproducibility.
        image_size=image_dimensions,    # Desired image dimensions (the EfficientNetB0 and MobileNetV2 default is 224).
        batch_size=batch_size,          # The image batch size we are using. May be set freely by the user.
        labels="inferred",              # Labels are generated accordingly to the dataset folder structure.
        class_names=class_names         # The names of the class labels, in this case they were set directly
                                        # by the dataset folder structure.
    )

    # Load the validation data. Same as above.
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        label_mode="categorical",
        seed=123,
        image_size=image_dimensions,
        batch_size=batch_size,
        labels="inferred",
        class_names=class_names
    )

    # Return the already loaded datasets.
    return train_ds, val_ds


def get_united_dataset(dataset_path, image_dimensions, batch_size):
    """Loads the requested dataset from the given directory just into one dataset containing all the images.

         Parameters
            ----------
            dataset_path : str
                The system path to the dataset in the form of a string.
            image_dimensions : (int,int)
                Input image dimensions to be loaded into the model.
            batch_size : int
                Size of the batches into which the images will be loaded.
         Returns
            ----------
            dataset : `tf.data.Dataset`
                The dataset is a tuple `(images, labels)`, where `images` has shape
                `(batch_size, image_size[0], image_size[1], num_channels)`,
                and `labels` are an array of integers from 0 to (number of classes - 1).
    """
    # Get the class names directly from the folder structure of the dataset.
    class_names = list([name for name in os.listdir(dataset_path) if pth.isdir(pth.join(dataset_path, name))])
    # Change the format of the string path to a "pathlib.Path".
    # This is the only path format TF function we are using works with.
    data_dir = pathlib.Path(dataset_path)
    # Get and print the total number of the dataset images. Serves as a control check.
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(f"Total image count: {image_count}")

    # Load the whole dataset.
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,                       # The dataset path in the "pathlib.Path" format.
        label_mode="categorical",       # Important, determines if the output will be binary or not.
        seed=123,                       # Random data split seed used for reproducibility.
        image_size=image_dimensions,    # Desired image dimensions (the EfficientNetB0 and MobileNetV2 default is 224).
        batch_size=batch_size,          # The image batch size we are using. May be set freely by the user.
        labels="inferred",              # Labels are generated accordingly to the dataset folder structure.
        class_names=class_names         # The names of the class labels, in this case they were set directly
                                        # by the dataset folder structure.
    )
    # Return the already loaded dataset.
    return dataset


def get_labels(dataset_path):
    """Loads the requested dataset from the given directory.
    Warning! The labels are taken directly from the different names of the class folders.

        Parameters
            ----------
            dataset_path : str
                The system path to the dataset in the form of a string.
        Returns
            ----------
            class_names : list of str
                List of strings which contains all of the class names.
    """
    # Get the class names directly from the folder structure of the dataset.
    class_names = list([name for name in os.listdir(dataset_path) if pth.isdir(pth.join(dataset_path, name))])
    print(f"All classes: {class_names}")    # Print all of the loaded class names.
    return class_names
