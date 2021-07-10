# Created by: Georgi Stoyanov Georgiev.
# as part of the "Neural network architectures for mobile devices" bachelor thesis

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

import graph_plotter        # Used to draw learning diagrams.
import model_controller     # Used to unfreeze the last layers of the model.
import init_mobilenetV2     # The initialization script of MobileNetV2.
import init_efficientnetB0  # The initialization script of EfficientNetB0.
from tensorflow.keras.layers.experimental import preprocessing  # Used to create the augmentation layer.

# The buffer size will be dynamically tuned. Can be changed by the programmer.
# Used in the prefetch setting below.
AUTOTUNE = tf.data.AUTOTUNE

# Followed directly this tutorial: https://www.tensorflow.org/tutorials/load_data/images


def fine_tune(train_dataset, val_dataset, class_num=8, model_name="MobileNetV2",
              checkpoint_path="model_training_ckpt/cp-{epoch:04d}.ckpt",
              image_dimension=224, epochs_classifier=30, epochs_train=25,
              plot=True, reverse_save_freq=5, active_augmentation=True,
              layers_to_be_trained=53):
    """Fine tunes a selected CNN model via Keras API.

        Parameters
            ----------
            train_dataset : `tf.data.Dataset`
                The already loaded training dataset.
            val_dataset : `tf.data.Dataset`
                The already loaded validation dataset.
            class_num : int, optional
                The number of dataset classes (default is 8).
            model_name : str, optional
                The name of the model to be fine-tuned (default is "MobileNetV2").
                Currently the only supported options are "MobileNetV2" and "EfficientNetB0".
            checkpoint_path : str, optional
                The full system path to the checkpoint to be loaded and fine-tuned
                (default: "model_training_ckpt/cp-{epoch:04d}.ckpt").
            image_dimension : int, optional
                The dimensions of the input images (default: 224). Has to match the input image resolution which the
                given CNN model supports.
            epochs_classifier : int, optional
                Number of epochs which the classification layer will be trained (default: 30).
            epochs_train : int, optional
                Number of epochs which the top neural network layers will be trained (default: 25).
            plot : bool, optional
                Determines whether to draw the training graphs of both the classification layer and the model or not
                (default: True).
            reverse_save_freq : int, optional
                The checkpoint save frequency of the model training process but it is inverted (default: 5).
                This means that the higher the value is, the lower the frequency will be. The default value of 5 means
                that on every 5 model training epochs a new checkpoint will be saved. Value of 10 means that
                on every 10 model training epochs a new checkpoint will be saved. There is always 1 checkpoint saved at
                the very beginning of the training process (at epoch 0) IF THE VALUE OF THIS PARAMETER IS NOT 0.
                WARNING! If set to 0, the checkpoint saving functionality will be turned OFF.
            active_augmentation : bool, optional
                Whether to use augmentation layer during the training process or not (default: True).
            layers_to_be_trained : int, optional
                Determines the number of the top model layers which will be trained. If this value is set to a bigger
                number than the total number of the model layers, then there won't be any errors and the maximum number
                of model layers will be trained.

        Optionally returns
            ----------
            (accuracy, loss) : `(float, float)`
                If the fine-tuning was successful, returns a tuple of two floats, where the first float is the model
                accuracy achieved in the last training epoch and the second float is the model loss achieved in the
                last training epoch as well.
    """
    model = None    # Model placeholder

    # Allows to do background tasks while loading the input tensors to the model.
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    # If augmentation option has been activated then add an input augmentation layer to the beginning of the network.
    # Because cars dataset is not very big data augmentation is recommended to reduce the overfitting.
    if active_augmentation:
        data_augment = tf.keras.Sequential(
            [
                preprocessing.RandomFlip('horizontal'),     # random horizontal flip with seed set to a constant value
                preprocessing.RandomRotation(0.15),         # random image rotation with seed set to a const.
                preprocessing.RandomContrast(factor=0.5),   # independently adjust the contrast on each channel
                preprocessing.RandomZoom(0.15),             # random image zoom
                preprocessing.RandomTranslation(height_factor=0.15, width_factor=0.15),
            ],
            name="img_augmentation")    # The name of the layer. Later we can use it as a key to get the layer.

        # Initialize the models.
        if model_name == "MobileNetV2":
            model = init_mobilenetV2.build_model(num_classes=class_num, img_dim=image_dimension,
                                                 img_augmentation=data_augment)
        elif model_name == "EfficientNetB0":
            model = init_efficientnetB0.build_model(num_classes=class_num, img_dim=image_dimension,
                                                    img_augmentation=data_augment)
    else:
        # No augmentation -> Don't add the augmentation layer and directly initialize the model.
        if model_name == "MobileNetV2":
            model = init_mobilenetV2.build_model(num_classes=class_num, img_dim=image_dimension)
        elif model_name == "EfficientNetB0":
            model = init_efficientnetB0.build_model(num_classes=class_num, img_dim=image_dimension)
    if model is None:   # There was an error in the model initialization.
        print("Model could not be loaded. Check the model name one more time.")
        return

    # Trains only the last layer of the network which is the main classification layer via the train and validation
    # datasets. Verbose just determines the console logging mode (verbose=1 -> progress bar + model accuracy and loss).
    class_history = model.fit(train_dataset, epochs=epochs_classifier, validation_data=validation_dataset, verbose=1)

    if plot:
        graph_plotter.plot_hist(class_history)  # Show a graph containing the full history of the learning process.
    # End of the classification layer training.

    # Now train the requested number of network top layers.
    # If the checkpoint saving option is not turned off then set it.
    if reverse_save_freq != 0:
        # Create a callback that saves the model's weights.
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,  # The logging mode.
            save_weights_only=True,    # Save only the checkpoint, not the whole model. Can be adjusted by the user.
            save_freq=reverse_save_freq * len(train_dataset))

        # Save the weights using the `checkpoint_path` format (the format that was defined with the string inside).
        model.save_weights(checkpoint_path.format(epoch=0))

        # Unfreeze the top layers of the model.
        model_controller.unfreeze_model(model, layers_to_be_trained)

        # Do the training and save the training history.
        # Use the already defined checkpoint callbacks during the training.
        # Verbose is once more the logging mode (verbose=1 -> progress bar + model accuracy and loss).
        hist = model.fit(train_dataset, epochs=epochs_train, callbacks=[cp_callback],
                         validation_data=validation_dataset, verbose=1)
    else:   # Else the checkpoint saver was turned off. Do the training without saving the checkpoints.
        model_controller.unfreeze_model(model, layers_to_be_trained)
        hist = model.fit(train_dataset, epochs=epochs_train, validation_data=validation_dataset,
                         verbose=1)
    if plot:
        graph_plotter.plot_hist(hist)   # Show the whole training history to the user in the form of a graph.

    # Return the accuracy and loss values from the last training iteration.
    return hist.history.get('accuracy')[-1], hist.history.get('loss')[-1]


# Example usage of the fine_tune method:
'''
data_path = "D:\\tmp\\cars4\\car_photos_4"
image_dim = 224
batch_size = 64
image_size = (image_dim, image_dim)
train_ds, val_ds = dataset_loader.get_dataset(data_path, image_size, batch_size)
fine_tune(train_ds, val_ds, model_name="MobileNetV2", image_dimension=image_dim)    # call the training directly
'''
