import load_dataset
import load_mobilenetV2_from_ckpt as mnV2_ckpt_init
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np

ckpt_path = "mobilenetV2_training"
image_path = "D:\\Gogi\\Photos\\Renny3\\20200809_191305.jpg"
dataset_path = "D:\\tmp\\cars2\\car_photos_2"

img_dim = 224
IMG_SIZE = (img_dim, img_dim)
batch_size = 32

train_ds, val_ds = load_dataset.get_dataset(dataset_path, IMG_SIZE, batch_size)
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

img = image.load_img(image_path, target_size=IMG_SIZE)
orig_img = img
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

model = mnV2_ckpt_init.get_model_from_ckpt(ckpt_path)

predictions = model(img)

predictions_np = predictions.numpy()
print('Predictions:\n', predictions_np)

# The k-th element will be in its final sorted position and all smaller elements will be moved before it
biggest_ind = np.argpartition(predictions_np[0], -3)[-3:]
biggest_pred = [predictions_np[0][i] for i in biggest_ind]
labeled_map = list(zip(biggest_pred, biggest_ind))
labeled_map.sort(key=lambda x: x[0], reverse=True)
pred_prob = [el[0] for el in labeled_map]
pred_labels = [class_names[el[1]] for el in labeled_map]


def plot_values(probabilities, labels):
    length = len(probabilities)
    plt.grid(False)
    plt.xticks(np.arange(length), labels)
    a_list = list(range(0, 10))
    a_list[:] = [x / 10 for x in a_list]

    plt.yticks(a_list)
    thisplot = plt.bar(range(length), probabilities, color='red')
    plt.ylim([0, 1])

    thisplot[0].set_color('blue')


plt.figure(figsize=(10, 6))
ax = plt.subplot(1, 2, 1)  # divide into two parts- left and right
plt.imshow(orig_img)
plt.title(pred_labels[0])
plt.subplot(1, 2, 2)
plot_values(pred_prob, pred_labels)
plt.show()
