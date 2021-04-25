import os
import os.path as pth
import tensorflow as tf
import pathlib

# https://www.tensorflow.org/tutorials/load_data/images


def get_dataset(dataset_path, image_dimensions, batch_size):
    """Loads the requested dataset from the given directory"""
    class_names = list([name for name in os.listdir(dataset_path) if pth.isdir(pth.join(dataset_path, name))])
    print(f"All classes: {class_names}")
    data_dir = pathlib.Path(dataset_path)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(f"Total image count: {image_count}")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        label_mode="categorical",   # important, determines if the output will be binary or not
        seed=123,
        image_size=image_dimensions,
        batch_size=batch_size,
        labels="inferred",
        class_names=class_names
    )

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

    return train_ds, val_ds


def get_labels(dataset_path):
    """Loads the requested dataset from the given directory.
    Warning! The labels are taken directly from the different names of the class folders."""

    class_names = list([name for name in os.listdir(dataset_path) if pth.isdir(pth.join(dataset_path, name))])
    print(f"All classes: {class_names}")
    return class_names
