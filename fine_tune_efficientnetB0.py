import os
import dataset_loader
import graph_plotter
import model_controller
import tensorflow as tf
from init_efficientnetB0 import build_model
from tensorflow.keras.layers.experimental import preprocessing

AUTOTUNE = tf.data.AUTOTUNE

# https://www.tensorflow.org/tutorials/load_data/images

dataset_path = "D:\\tmp\\cars3\\car_photos_3"
img_dim = 224
IMG_SIZE = (img_dim, img_dim)
batch_size = 64

train_ds, val_ds = dataset_loader.get_dataset(dataset_path, IMG_SIZE, batch_size)

NUM_CLASSES = len(train_ds.class_names)

# Allows to do background tasks while loading the input tensors to the model.
train_dataset = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_dataset = val_ds.prefetch(buffer_size=AUTOTUNE)
# test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# because cars dataset is not very big data augmentation is recommended to reduce the overfitting
data_augmentation = tf.keras.Sequential(
    [
        preprocessing.RandomFlip('horizontal'),
        preprocessing.RandomRotation(0.15),
        preprocessing.RandomContrast(factor=0.5),
        preprocessing.RandomZoom(0.15),
        preprocessing.RandomTranslation(height_factor=0.15, width_factor=0.15),
    ],
    name="img_augmentation")

model = build_model(num_classes=NUM_CLASSES, img_dim=img_dim, img_augmentation=data_augmentation)

epochs = 30  # @param {type: "slider", min:8, max:80}

history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, verbose=1)
graph_plotter.plot_hist(history)

# Training
checkpoint_path = "efficientnetB0_training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=5*len(train_ds))

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# unfreeze the last 60 layers of the model
model_controller.unfreeze_model(model, 60)

epochs = 25  # @param {type: "slider", min:8, max:50}
hist = model.fit(train_dataset, epochs=epochs, callbacks=[cp_callback], validation_data=validation_dataset, verbose=1)
graph_plotter.plot_hist(hist)
