#!/usr/bin/python

"""Script that evaluates just one image

This script directly loads the requested model and then evaluates it on one image. This is a script which was created
only with testing purposes.


Can be started directly. It takes 2 (x)or 5 arguments:

I.)  1.) Path to the dataset from which the dataset labels (class names) will be taken.
     2.) Path to the image which has to be evaluated.

II.) 1.) The name of the model (MobileNetV2 or EfficientNetB0).
     2.) The format of the model (can be set to "ckpt" which means loading from the checkpoints, or to anything else
         which will load the model from the keras SavedModel format).
     3.) The path to the checkpoints to be loaded.
     4.) Path to the dataset from which the dataset labels (class names) will be taken.
     5.) Path to the image which has to be evaluated.
"""

import sys
from model_controller import ModelController

model_name = "EfficientNetB0"                       # The name of the model (can be set by the user).
model_format = "ckpt"                               # The format of the model checkpoints (can be set by the user).
model_path = "efficientnetB0_training_2_0"          # The path to the checkpoints to be loaded (can be set by the user).

# Path to the dataset from which the dataset labels (class names) will be taken. It must be set as an argument.
dataset_path = None
# Path to the image which has to be evaluated.
image_path = None

# Read the arguments.
if len(sys.argv) == 2:
    dataset_path = sys.argv[0]
    image_path = sys.argv[1]
elif len(sys.argv) == 5:
    model_name = sys.argv[0]
    model_format = sys.argv[1]
    model_path = sys.argv[2]
    dataset_path = sys.argv[3]
    image_path = sys.argv[4]
else:
    sys.exit("ERROR! Invalid number of arguments...")

# Initialize the Model Controller.
model_ctrl = ModelController(model_name)

# Load the model from checkpoints or from the SavedModel format.
if model_format == "ckpt":
    model_ctrl.load_model_from_ckpt(model_path)
else:
    model_ctrl.load_saved_model(model_path)

# Evaluate the image.
model_ctrl.eval_image(image_path, dataset_path)

