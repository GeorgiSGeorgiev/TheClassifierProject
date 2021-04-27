import tensorflow as tf

saved_path = "D:\\Gogi\\UK\\MFF_UK_2020_2021\\TheProject\\New_Progression\\efficientnetB0_SavedModel"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)    # path to the SavedModel directory
print("Converting to TensorFlow Lite...")
tflite_model = converter.convert()

print("Writing to file...")
# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

print("Model conversion finished.")
