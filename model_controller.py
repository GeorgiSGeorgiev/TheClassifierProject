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

import numpy as np
import dataset_loader
import graph_plotter
import init_mobilenetV2 as init_mnV2        # MobileNetV2 initializer.
import init_efficientnetB0 as init_enB0     # EfficientNetB0 initializer.

from tensorflow.keras import layers         # Allow to add Batch normalisation. We have written about it in our work.
from keras.preprocessing import image       # Allows image loading and scaling.
from tensorflow.keras.layers.experimental import preprocessing  # Augmentation layer.


def unfreeze_model(in_model, back_layers_count):
    """Unfreeze the last back_layers_count number of layers in the in_model.
    Doesn't modify the classification layers of the model."""
    # We unfreeze the top layers while leaving BatchNorm last layers frozen.
    # This won't modify the classification layer during the training.
    for layer in in_model.layers[-back_layers_count:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    # Set the Adam optimizer. Can be changed to a different one.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    in_model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


class ModelController:
    """Stores a CNN model. Contains multiple functions that evaluate and save the model."""
    available_models = ["MobileNetV2", "EfficientNetB0"]
    model = None    # Variable which stores the resulting model.
    img_dim = 224   # Dimensions of the input images.
    img_size = (img_dim, img_dim)

    # Constructor. Defines the type of the network which will be stored in the new instance of this class.
    # The type can not be changed any more to avoid making setting mistakes which can cause debugging troubles.
    def __init__(self, model_type):
        self.type = model_type
        if self.type is self.available_models[0]:
            self.model_type_inx = 0
            self.img_dim = 224
        elif self.type is self.available_models[1]:
            self.model_type_inx = 1
            self.img_dim = 224

        self.img_size = (self.img_dim, self.img_dim)

    def load_model_from_ckpt(self, ckpt_dir):
        """Gets the checkpoint parent directory via ckpt_dir and then if the model types match builds the model and
         loads the last available checkpoint to it."""
        # the newly initialized base model has to be the same as the model which weights were saved to the checkpoint
        # if the original model had augmentation layer at the beginning, the new model has to have it too
        # Below we add a basic augmentation layer that almost does not change the input. It will be active only
        # if we are training the model but it has to be there to maintain the model structure.
        data_augmentation = tf.keras.Sequential(
            [
                preprocessing.RandomContrast(factor=0.01),
            ],
            name="img_augmentation")

        latest = tf.train.latest_checkpoint(ckpt_dir)   # Get the latest checkpoint.
        new_model = None                                # Reset the model.
        # Build the new model according to the saved type name inside this class.
        if self.type == "MobileNetV2":
            new_model = init_mnV2.build_model(8, self.img_dim, data_augmentation)
        elif self.type == "EfficientNetB0":
            new_model = init_enB0.build_model(8, self.img_dim, data_augmentation)
        # Load the model weights directly from the variable which contains the loaded from the checkpoint weights.
        new_model.load_weights(latest)
        self.model = new_model  # set the model variable
        print("Model successfully loaded.")

    def test_loaded_model(self, test_images, test_labels):
        """Gets a test images and test labels and evaluates the loaded model on them. After that the results
        are written as an standard output."""
        if self.model is None:
            return
        loss, acc = self.model.evaluate(test_images, test_labels, verbose=2)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
        print("Restored model, loss: {:5.2f}%".format(100 * loss))

    def eval_image(self, image_path, dataset_path, predictions_to_be_shown=3, ckpt_path=None, plot=True):
        """Evaluates a specified image. The path to the image can be set in the image_path variable. The
        dataset_path is needed to get the class names from the dataset file structure. The input parameter
        "predictions_to_be_shown" determines the number of predictions to be shown to the user. The input parameter
        "ckpt_path" has to be set if the user wants to load a different checkpoint to this instance and
        the current model variable is set to NONE. The "plot" boolean turns on or off the graph plotter which shows
        the evaluation results in a form of a column diagram."""

        class_names = dataset_loader.get_labels(dataset_path)

        # Convert the input image to the right input format
        img = image.load_img(image_path, target_size=self.img_size)
        orig_img = img
        img = image.img_to_array(img)   # Convert the image to a Numpy array which will be later fed to the network.
        img = np.expand_dims(img, axis=0)   # Add one more 0 axis to the image to match the input format of the model.

        # If model is set to None and ckpt_path is not None then try to load the model.
        self.check_and_load(ckpt_path)
        # If no model available after the loading attempt, then:
        if self.model is None:
            print("Error! Model couldn't be loaded.")
            return

        # Model available, so evaluate it and make image predictions.
        print("Evaluating the model")
        predictions = self.model(img)
        predictions_np = predictions.numpy()    # Convert the predictions to a numpy array.
        print('Predictions:\n', predictions_np)     # Write the numpy array on the console.

        # Get the top n predictions and all the labels.
        pred_prob, pred_labels = graph_plotter.get_top_values(predictions_np, class_names, predictions_to_be_shown)
        # If set to plotting, show the top predictions to the user in a column diagram.
        if plot:
            graph_plotter.show_image_and_graph(orig_img, pred_prob, pred_labels)
        # Return the predictions.
        return pred_prob, pred_labels

    def eval_validation_batch(self, dataset_path, batch_index=0, ckpt_path=None, plot=True):
        """Evaluates a specified batch of images. The dataset_path is needed to get the class names from the dataset
        file structure AND to load the whole dataset. From the validation part there is chosen the batch with index of
        batch_index. The batch size is set to 9 and can not be changed. This gives exactly 9 predictions.
        The input parameter "ckpt_path" has to be set if the user wants to load a different checkpoint to this
        instance and the current model variable is set to NONE. The "plot" boolean turns on or off the graph plotter
        which shows the evaluation results in a form of a column diagram."""

        # Load the whole dataset from the given path into batches of size 9.
        train_ds, val_ds = dataset_loader.get_dataset(dataset_path, self.img_size, 9)

        # If model is set to None and ckpt_path is not None then try to load the model.
        self.check_and_load(ckpt_path)
        # If no model available after the loading attempt, then:
        if self.model is None:
            print("Error! Model couldn't be loaded.")
            return

        # Iterate through the batches until the requested one is returned. The dataset type itself does not support
        # indexing but can be accessed via a numpy iterator.
        for i in range(batch_index):
            val_ds.as_numpy_iterator().next()
        image_batch, label_batch = val_ds.as_numpy_iterator().next()    # the iteration has stopped one step earlier
        predictions = self.model.predict_on_batch(image_batch).flatten()
        # the result is a list of lists which has only one list inside, so use flatten to get only that list as a result

        # Apply a sigmoid since our model returns logits which have to be represented as a percent values.
        predictions = tf.nn.softmax(predictions)

        print('Predictions:\n', predictions)
        print('Labels:\n', label_batch)
        # If set to true, show the plot.
        if plot:
            graph_plotter.show_9_image_predictions(image_batch, label_batch, val_ds.class_names)

    def check_and_load(self, ckpt_path):
        """If model is set to None and ckpt_path is not None then try to load the model from the last checkpoint."""
        if self.model is None:
            if ckpt_path is None:
                print("Error! No model present nor valid checkpoint path provided")
                return
            # load the model from the latest checkpoint available
            self.load_model_from_ckpt(ckpt_path)

    def save_as_saved_model(self, destination_path=None, ckpt_path=None):
        """Convert the loaded model to a SavedModel format. If destination_path is set to None then save the model to
        the local destination of this script. The variable "ckpt_path" is used only if the internal model equals None
        and the "ckpt_path" is not None. Then it serves as a destination of the checkpoint to be loaded."""
        # If model is set to None and ckpt_path is not None then try to load the model.
        self.check_and_load(ckpt_path)
        # If still no model available:
        if self.model is None:
            print("Error! Model couldn't be loaded.")
            return

        # Starting the conversion process.
        print("Saving the model...")
        # Save the entire model as a SavedModel.
        # If "destination_path" not set use the default one which is the directory of this project.
        if destination_path is None:
            self.model.save('saved_model/my_model')
        else:
            self.model.save(destination_path, include_optimizer=True)
        print("Model saved successfully.")

    def load_saved_model(self, model_path=None):
        """Loading a model from the SavedModel format directly to this instance. If "model_path" is None then
        load from the default path which is 'saved_model/my_model'. Otherwise use "model_path" as a destination path
        of the desired SavedModel."""
        print("Loading the model into the controller...")
        if model_path is None:
            self.model = tf.keras.models.load_model('saved_model/my_model')
        else:
            self.model = tf.keras.models.load_model(model_path)
        print()
        print("IMPORTANT: The restored model will not have additional training information.\n"
              "If you are restoring a model that has training information, you don't need the saved optimizer values.\n"
              "Additional warnings may appear.\n")
        print("Model loaded successfully.")

    def reset_model(self):
        """Allows to set the inside model to None, which resets it."""
        self.model = None

    def model_available(self):
        """Checks whether the inside model is None or not."""
        if self.model is None:
            return False
        return True
