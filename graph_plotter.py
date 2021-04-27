import numpy as np
import matplotlib.pyplot as plt


def get_top_values(predictions_np, string_class_names, top_k_el):
    if top_k_el > len(string_class_names):
        top_k_el = 3
    # The k-th element will be in its final sorted position and all smaller elements will be moved before it
    biggest_ind = np.argpartition(predictions_np[0], -top_k_el)[-top_k_el:]
    biggest_pred = [predictions_np[0][i] for i in biggest_ind]
    labeled_map = list(zip(biggest_pred, biggest_ind))
    labeled_map.sort(key=lambda x: x[0], reverse=True)
    pred_prob = [el[0] for el in labeled_map]
    pred_labels = [string_class_names[el[1]] for el in labeled_map]
    return pred_prob, pred_labels


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


def show_image_and_graph(image, pred_prob, pred_labels):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)  # divide into two parts- left and right
    plt.imshow(image)
    plt.title(pred_labels[0] + ", {:5.2f}%".format(100 * pred_prob[0]))
    plt.subplot(1, 2, 2)
    plot_values(pred_prob, pred_labels)
    plt.show()


def plot_hist(training_data):
    plt.plot(training_data.history["accuracy"])
    plt.plot(training_data.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


def show_9_image_predictions(images, labels, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype("uint8"))
        predicted_label = np.argmax(labels[i])
        plt.title(class_names[predicted_label])
        plt.axis("off")
    plt.show()
