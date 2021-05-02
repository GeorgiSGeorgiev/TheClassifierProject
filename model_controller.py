import numpy as np
import dataset_loader
import graph_plotter
import tensorflow as tf
import init_mobilenetV2 as init_mnV2
import init_efficientnetB0 as init_enB0

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
    """Stores a CNN model. Contains multiple functions that evaluate and save the model."""
    available_models = ["MobileNetV2", "EfficientNetB0"]
    model = None
    img_dim = 224
    img_size = (img_dim, img_dim)

    def __init__(self, model_type):
        self.type = model_type
        if self.type is self.available_models[0]:
            self.model_type_inx = 0
            self.img_dim = 224
        elif self.type is self.available_models[1]:
            self.model_type_inx = 1
            self.img_dim = 224

        self.img_size = (self.img_dim, self.img_dim)

    def load_model_from_ckpt(self, ckpt_dir):
        # the newly initialized base model has to be the same as the model which weights were saved to the checkpoint
        # if the original model had augmentation layer at the beginning, the new model has to have it too
        # Basic augmentation layer that almost does not change the input:
        data_augmentation = tf.keras.Sequential(
            [
                preprocessing.RandomContrast(factor=0.01),
            ],
            name="img_augmentation")

        latest = tf.train.latest_checkpoint(ckpt_dir)
        new_model = None
        if self.type == "MobileNetV2":
            new_model = init_mnV2.build_model(8, self.img_dim, data_augmentation)
        elif self.type == "EfficientNetB0":
            new_model = init_enB0.build_model(8, self.img_dim, data_augmentation)
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

        class_names = dataset_loader.get_labels(dataset_path)

        # convert the input image to the right input format
        img = image.load_img(image_path, target_size=self.img_size)
        orig_img = img
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        
        self.check_and_load(ckpt_path)
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

    def eval_validation_batch(self, dataset_path, batch_index=0, ckpt_path=None, plot=True):
        train_ds, val_ds = dataset_loader.get_dataset(dataset_path, self.img_size, 9)

        self.check_and_load(ckpt_path)
        # still no model available
        if self.model is None:
            print("Error! Model couldn't be loaded.")
            return

        for i in range(batch_index):
            val_ds.as_numpy_iterator().next()
        image_batch, label_batch = val_ds.as_numpy_iterator().next()
        predictions = self.model.predict_on_batch(image_batch).flatten()

        # Apply a sigmoid since our model returns logits
        predictions = tf.nn.softmax(predictions)

        print('Predictions:\n', predictions)
        print('Labels:\n', label_batch)
        if plot:
            graph_plotter.show_9_image_predictions(image_batch, label_batch, val_ds.class_names)

    def check_and_load(self, ckpt_path):
        if self.model is None:
            if ckpt_path is None:
                print("Error! No model present nor valid checkpoint path provided")
                return
            # load the model from the latest checkpoint available
            self.load_model_from_ckpt(ckpt_path)

    def save_as_saved_model(self, destination_path=None, ckpt_path=None):
        self.check_and_load(ckpt_path)
        # still no model available
        if self.model is None:
            print("Error! Model couldn't be loaded.")
            return

        print("Saving the model...")
        # Save the entire model as a SavedModel.
        if destination_path is None:
            self.model.save('saved_model/my_model')
        else:
            self.model.save(destination_path, include_optimizer=True)
        print("Model saved successfully.")

    def load_saved_model(self, model_path=None):
        print("Loading the model into the controller...")
        if model_path is None:
            self.model = tf.keras.models.load_model('saved_model/my_model')
        else:
            self.model = tf.keras.models.load_model(model_path)
        print()
        # print("IMPORTANT NOTES: The restored model will not have additional training information.\n"
        #     "If you are restoring a model that has training information, you don't need the saved optimizer values.\n"
        #     "Additional warnings may appear.\n")
        print("Model loaded successfully.")
