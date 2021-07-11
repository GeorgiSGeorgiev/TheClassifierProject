import tensorflow as tf
import tensorflow_model_optimization as tfmot
from model_controller import ModelController

tf.random.set_seed(9)

ckpt_path = "efficientnetB0_training_2_7"
# saved_path = "D:\\Gogi\\UK\\MFF_UK_2020_2021\\TheProject\\New_Progression\\efficientnetB0_SavedModel_2_0"

# model_ctrl = ModelController("MobileNetV2")
model_ctrl = ModelController("EfficientNetB0")

model_ctrl.load_model_from_ckpt(ckpt_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model_ctrl.model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as file:
    file.write(tflite_model)
