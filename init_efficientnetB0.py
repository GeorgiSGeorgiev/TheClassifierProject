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

# Import the actual model structure and the preloaded model checkpoints
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
# Layers part contains the different types of layers which we can use to create the new tail of the network.


# Followed directly this tutorial: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
def build_model(num_classes, img_dim=224, img_augmentation=None):
    """Initializes a new EfficientNetB0 model.

    Parameters
        ----------
        num_classes : int
            The number of dataset classes.
        img_dim : int, optional
            Input image dimensions (default is 224). In our case we don't include the top of the model so
            image dimensions are fully adjustable.
        img_augmentation : tf.keras.Sequential, optional
            The Sequential layers which form the first new input augmentation layer (default is None)
    """
    # This layer sets all of the input values in the interval [-1,1].
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    # img_dim X img_dim X 3 (RGB channels, so there are 3 color channels)
    input_shape = (img_dim, img_dim, 3)
    inputs = layers.Input(shape=input_shape)    # Create a new input tensor.
    # If available, add data augmentation to the input tensor. The augmentation is a new first layer of the network.
    if img_augmentation is not None:
        x = img_augmentation(inputs)
        x = preprocess_input(x)
    else:
        x = preprocess_input(inputs)

    # Don't include the top categorization layer, we'll build it on our own.
    model = EfficientNetB0(input_shape=input_shape, include_top=False, input_tensor=x, weights="imagenet")
    # So we get the original pretrained EfficientNetB0 model from the loaded inside Keras checkpoints which were trained
    # on the ImageNet dataset. Via the input_tensor variable we add the augmentation to the final model.
    # The include_top False value says that we don't want to include the top layer from the checkpoint to the new model.
    # The pretrained model can classify different objects than our model, so we have to create a new top which will
    # allow us to train the model to classify new objects.
    # Input_shape is the size of the input images.

    # Freeze the pretrained weights. We don't want to modify them by accident.
    model.trainable = False

    print(f"Original model output: {model.output}")
    # Rebuild the top of the model.
    # First add a new pooling layer to make the dimensions of the output we are classifying smaller.
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    print(f"Output after average pooling: {x}")
    # Add a batch normalisation layer to increase the stability and the speed of the result model..
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2  # Determines the rate of the parameter zeroing.
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)     # Add a top dropout layer.
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)
    # Determine the output format at the end. First specify the number of classes to be classified.
    # Then the softmax function converts the output logits to the probabilities we desire.

    # Compile the model.
    model = tf.keras.Model(inputs, outputs, name="EfficientNetB0")
    # Specify the output, the input size and the type of the model to be loaded.
    # Then add the Adam optimizer to the network.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # The basic model is already ready to be returned.
    return model
