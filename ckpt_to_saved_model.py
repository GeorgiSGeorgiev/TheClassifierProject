"""Convert model saved as checkpoints to SavedModel

This script is meant to be run directly. It converts the model saved as checkpoints to the SavedModel format.
The main advantage of the SavedModel format is that it contains the whole model structure inside. So it does not
require the model to be fully initialized before loading in its weight values.

This script requires from the user to have `tensorflow` installed within the user's Python environment. Also
the model_controller script is needed which is part of this project.

The script doesn't take any input arguments. The user only has to adjust the first three script variables.
It is meant to show an example usage of the conversion to SavedModel functionality of the ModelController class.

"""

from model_controller import ModelController

# The path of the checkpoint which will be loaded. The first variable to be adjusted by the user.
ckpt_path = "efficientnetB0_training_2_4"
# The path to the saving location of the new SavedModel file. It is the second variable to be adjusted by the user.
saved_path = "D:\\Gogi\\UK\\MFF_UK_2020_2021\\TheProject\\New_Progression\\efficientnetB0_SavedModel_2_4"
# Create an instance of the ModelController class. User has to adjust the name of the model according to the loaded
# checkpoints. Incompatible models will result into an error.
model_ctrl = ModelController("EfficientNetB0")

print("Starting conversion")
# Load the model from checkpoint.
model_ctrl.load_model_from_ckpt(ckpt_path)
# Save as saved model.
model_ctrl.save_as_saved_model(saved_path)

print("Conversion successfully ended")
