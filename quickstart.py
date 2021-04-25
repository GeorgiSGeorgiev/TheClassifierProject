import tensorflow as tf
import load_dataset
import load_mobilenetV2_from_ckpt as mnV2_ckpt_init
import matplotlib.pyplot as plt
import numpy as np

ckpt_path = "mobilenetV2_training"
dataset_path = "D:\\tmp\\cars2\\car_photos_2"
img_dim = 224
IMG_SIZE = (img_dim, img_dim)
batch_size = 32

train_ds, val_ds = load_dataset.get_dataset(dataset_path, IMG_SIZE, batch_size)

NUM_CLASSES = len(train_ds.class_names)

model = mnV2_ckpt_init.get_model_from_ckpt(ckpt_path)

image_batch, label_batch = val_ds.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.softmax(predictions)

print('Predictions:\n', predictions)
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    predicted_label = np.argmax(label_batch[i])
    plt.title(val_ds.class_names[predicted_label])
    plt.axis("off")
plt.show()
