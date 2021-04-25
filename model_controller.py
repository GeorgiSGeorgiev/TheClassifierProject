import numpy as np
import load_dataset
import graph_plotter
import tensorflow as tf
import mobilenetV2_init as init

from tensorflow.keras import layers
from keras.preprocessing import image
from tensorflow.keras.layers.experimental import preprocessing


def unfreeze_model(in_model, back_layers_count):
    # We unfreeze the top 60 layers while leaving BatchNorm layers frozen
    for layer in in_model.layers[-back_layers_count:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    in_model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


# UNDER CONSTRUCTION
class ModelController:
    model = None

    def __init__(self, model_type):
        self.type = model_type

    # TODO: EfficientNet Implementation

    def load_model_from_ckpt(self, ckpt_dir):
        data_augmentation = tf.keras.Sequential(
            [
                preprocessing.RandomContrast(factor=0.01),
            ],
            name="img_augmentation")

        latest = tf.train.latest_checkpoint(ckpt_dir)
        # if model name is MobileNetV2 then
        new_model = init.build_model(8, 224, data_augmentation)
        # otherwise load the efficientnet weights
        new_model.load_weights(latest)
        self.model = new_model
        print("Model successfully loaded.")

    def test_loaded_model(self, test_images, test_labels):
        if self.model is None:
            return
        loss, acc = self.model.evaluate(test_images, test_labels, verbose=2)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
        print("Restored model, loss: {:5.2f}%".format(100 * loss))

    def eval_image(self, image_path, dataset_path, predictions_to_be_shown=3, ckpt_path=None, plot=True):
        # determined by the model!!!!!!!!!!!!
        img_dim = 224
        img_size = (img_dim, img_dim)

        class_names = load_dataset.get_labels(dataset_path)

        # convert the input image to the right input format
        img = image.load_img(image_path, target_size=img_size)
        orig_img = img
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        if self.model is None:
            if ckpt_path is None:
                print("Error! No model present nor valid checkpoint path provided")
                return
            # load the model from the latest checkpoint available
            self.load_model_from_ckpt(ckpt_path)
        # still no model available
        if self.model is None:
            print("Error! Model couldn't be loaded.")
            return

        # make image predictions
        print("Evaluating the model")
        predictions = self.model(img)
        predictions_np = predictions.numpy()
        print('Predictions:\n', predictions_np)

        # get the top n predictions and all the labels
        pred_prob, pred_labels = graph_plotter.get_top_values(predictions_np, class_names, predictions_to_be_shown)
        if plot:
            graph_plotter.show_image_and_graph(orig_img, pred_prob, pred_labels)

        return pred_prob, pred_labels
