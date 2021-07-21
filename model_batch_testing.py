"""Batch evaluation script

This script allows directly to evaluate the model on a batch of 9 images.

Can be started directly and doesn't take any arguments. All the settings are directly inside the script.

"""

# Created by: Georgi Stoyanov Georgiev.
# as part of the "Neural network architectures for mobile devices" bachelor thesis
from model_controller import ModelController

# Path to the checkpoint to be loaded to the model.
ckpt_path = "efficientnetB0_training_2_4"
# The dataset path from which the dataset will be taken.
dataset_path = "D:\\tmp\\cars4\\car_photos_4"

# Create a new model controller. The model type has to match the checkpoint type.
model_ctrl = ModelController("EfficientNetB0")
# Load the checkpoints.
model_ctrl.load_model_from_ckpt(ckpt_path)

# Evaluate the batch of 9 images.
model_ctrl.eval_validation_batch(dataset_path)
