import tensorflow as tf
import pathlib

# https://www.tensorflow.org/tutorials/load_data/images


def get_dataset(dataset_path, image_dimensions, batch_size):
    """Loads the requested dataset from the given directory"""
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
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(f"All classes: {class_names}")

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        label_mode="categorical",
        seed=123,
        image_size=image_dimensions,
        batch_size=batch_size)

    return train_ds, val_ds
