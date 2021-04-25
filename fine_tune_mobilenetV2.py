import os
import tensorflow as tf
import matplotlib.pyplot as plt
import load_dataset
from mobilenetV2_init import build_model
from tensorflow.keras import layers

from tensorflow.keras.layers.experimental import preprocessing

AUTOTUNE = tf.data.AUTOTUNE

# https://www.tensorflow.org/tutorials/load_data/images

dataset_path = "D:\\tmp\\cars2\\car_photos_2"
img_dim = 224
IMG_SIZE = (img_dim, img_dim)
batch_size = 32

train_ds, val_ds = load_dataset.get_dataset(dataset_path, IMG_SIZE, batch_size)

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
        preprocessing.RandomContrast(factor=0.4),
        preprocessing.RandomZoom(0.1),
        preprocessing.RandomTranslation(height_factor=0.15, width_factor=0.15),
    ],
    name="img_augmentation")


def plot_hist(training_data):
    plt.plot(training_data.history["accuracy"])
    plt.plot(training_data.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


model = build_model(num_classes=NUM_CLASSES, img_dim=img_dim, img_augmentation=data_augmentation)

epochs = 30  # @param {type: "slider", min:8, max:80}

history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, verbose=2)
plot_hist(history)


# Training
checkpoint_path = "mobilenetV2_training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=5*batch_size)

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))


def unfreeze_model(in_model):
    # We unfreeze the top 60 layers while leaving BatchNorm layers frozen
    for layer in in_model.layers[-60:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    in_model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


unfreeze_model(model)

epochs = 25  # @param {type: "slider", min:8, max:50}
hist = model.fit(train_dataset, epochs=epochs, callbacks=[cp_callback], validation_data=validation_dataset, verbose=2)
plot_hist(hist)
