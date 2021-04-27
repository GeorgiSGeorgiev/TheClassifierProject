import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers


# Followed directly this tutorial: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
def build_model(num_classes, img_dim, img_augmentation=None):
    # This layer sets all of the input values in the interval [-1,1]. This is required by MobileNetV2.
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    input_shape = (img_dim, img_dim, 3)
    inputs = layers.Input(shape=input_shape)  # 3 color channels
    if img_augmentation is not None:
        x = img_augmentation(inputs)
        x = preprocess_input(x)
    else:
        x = preprocess_input(inputs)

    # Don't include the top categorization  layer, we'll build it on our own
    model = EfficientNetB0(input_shape=input_shape, include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    print(f"Original model output: {model.output}")
    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    print(f"Output after average pooling: {x}")
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNetB0")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
