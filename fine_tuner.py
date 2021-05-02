import dataset_loader
import graph_plotter
import model_controller
import tensorflow as tf
import init_mobilenetV2
import init_efficientnetB0
from tensorflow.keras.layers.experimental import preprocessing

AUTOTUNE = tf.data.AUTOTUNE

# https://www.tensorflow.org/tutorials/load_data/images


def fine_tune(train_dataset, val_dataset, class_num=8, model_name="MobileNetV2",
              checkpoint_path="model_training_ckpt/cp-{epoch:04d}.ckpt", image_dimension=224, epochs_classifier=30,
              epochs_train=25, plot=True, reverse_save_freq=5, active_augmentation=True, layers_to_be_trained=53):
    model = None    # Model placeholder

    # Allows to do background tasks while loading the input tensors to the model.
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    # If augmentation option has been activated then add an input augmentation layer to the beginning of the network.
    # Because cars dataset is not very big data augmentation is recommended to reduce the overfitting.
    if active_augmentation:
        data_augment = tf.keras.Sequential(
            [
                preprocessing.RandomFlip('horizontal'),
                preprocessing.RandomRotation(0.15),
                preprocessing.RandomContrast(factor=0.5),
                preprocessing.RandomZoom(0.15),
                preprocessing.RandomTranslation(height_factor=0.15, width_factor=0.15),
            ],
            name="img_augmentation")
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
    if model is None:
        print("Model could not be loaded. Check the model name one more time.")
        return

    # Trains only the last layer of the network which is the main classification layer.
    class_history = model.fit(train_dataset, epochs=epochs_classifier, validation_data=validation_dataset, verbose=1)

    if plot:
        graph_plotter.plot_hist(class_history)  # Show a graph containing the full history of the learning process.

    # Trains the requested number of layers.
    if reverse_save_freq != 0:
        # Create a callback that saves the model's weights.
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq=reverse_save_freq * len(train_dataset))

        # Save the weights using the `checkpoint_path` format.
        model.save_weights(checkpoint_path.format(epoch=0))

        # Unfreeze the last layers of the model.
        model_controller.unfreeze_model(model, layers_to_be_trained)

        # Do the training and save the training history.
        hist = model.fit(train_dataset, epochs=epochs_train, callbacks=[cp_callback], validation_data=validation_dataset,
                         verbose=1)
    else:   # The checkpoint saver was turned off. Do the training without saving the checkpoints.
        model_controller.unfreeze_model(model, layers_to_be_trained)
        hist = model.fit(train_dataset, epochs=epochs_train, validation_data=validation_dataset,
                         verbose=1)
    if plot:
        graph_plotter.plot_hist(hist)   # Show the whole training history to the user in the form of a graph.

    # Return the accuracy and lost values from the last training iteration.
    return hist.history.get('accuracy')[-1], hist.history.get('loss')[-1]


# Example usage of the fine_tune method:
'''
data_path = "D:\\tmp\\cars3\\car_photos_3"
image_dim = 224
batch_size = 64
image_size = (image_dim, image_dim)
train_ds, val_ds = dataset_loader.get_dataset(data_path, image_size, batch_size)
fine_tune(train_ds, val_ds, model_name="MobileNetV2", image_dimension=image_dim)    # call the training directly
# fine_tune(train_ds, val_ds, model_name="EfficientNetB0", image_dimensions=image_dim)
'''
