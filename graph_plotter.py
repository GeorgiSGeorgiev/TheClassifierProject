import numpy as np
import matplotlib.pyplot as plt


def get_top_values(predictions_np, string_class_names, top_k_el):
    """Fine tunes a selected CNN model via Keras API.

        Parameters
            ----------
            predictions_np : numpy.array of array of float
                Numpy array containing all the result predictions saved as floats. Our model returns an array of
                arrays which has only one element inside. That one element is the array of predictions we want.
            string_class_names : list of strings
                List containing the class names to be binded with the probabilities.
            top_k_el : int
                Number of predictions to be shown on the diagram.

        Returns
            ----------
            (pred_prob, pred_labels) : `(array of floats, array of strings)`
                Return the k highest probabilities (sorted) and their class names (labels).
    """
    # Corner case: the specified value top_k_el is bigger than the size of the class names list.
    if top_k_el > len(string_class_names):
        top_k_el = len(string_class_names) - 1

    # predictions_np[0] contains the % predictions we desire. predictions_np actually contains only one item because we
    # make only one set of predictions
    # The argpartition call below ensures that the k-th (-top_k_el) element will be in its final sorted position,
    # all smaller elements will be moved before it and all bigger after it. Then we get those indices of the
    # elements bigger than the element at the index of (-top_k_el).
    # So in the end we get the indices of the top k elements in an unordered fashion.
    biggest_ind = np.argpartition(predictions_np[0], -top_k_el)[-top_k_el:]
    # Get the biggest prediction values according to the indices in the biggest_ind array.
    biggest_pred = [predictions_np[0][i] for i in biggest_ind]
    # From the both new arrays create a list of tuples: (biggest_pred, biggest_ind)
    labeled_map = list(zip(biggest_pred, biggest_ind))
    # Then sort the list according to the prediction values in an reversed order.
    labeled_map.sort(key=lambda x: x[0], reverse=True)
    # The first element of each tuple is the prediction value we desire.
    pred_prob = [el[0] for el in labeled_map]
    # Use the indices from the second element of each tuple to get the right class name.
    pred_labels = [string_class_names[el[1]] for el in labeled_map]
    # Return the top k probabilities and their class names in the right order.
    return pred_prob, pred_labels


def plot_values(probabilities, labels):
    """Initializes a basic bar diagram but doesn't show it.
    The first bar will be blue and the rest of them red.
    This function can be used to show explicitly the best value bar.

        Parameters
            ----------
            probabilities : np.array of float
                Numpy array containing all the desired predictions saved as floats.
            labels : list of strings
                List containing the class names which have to be shown on the diagram.
    """
    length = len(probabilities)
    plt.grid(False)     # no grid on the diagram
    plt.xticks(np.arange(length), labels)   # the labels will be  shown on the x-axis
    a_list = list(range(0, 10))
    a_list[:] = [x / 10 for x in a_list]    # generate the possible percent values

    plt.yticks(a_list)  # the y-axis will contain percents
    thisplot = plt.bar(range(length), probabilities, color='red')   # create a red bar for each class
    # The bar height is determined by the corresponding probability value.
    plt.ylim([0, 1])    # set the limits of the y-axis

    thisplot[0].set_color('blue')


# Uses the previous function
def show_image_and_graph(image, pred_prob, pred_labels):
    """Creates a diagram which has a picture on its left side and a bars on its right side.
    The bars represent the input probabilities and their labels are the class names from the input.
    Uses the plot_values function to create the bars on the right side.

        Parameters
            ----------
            image : A PIL Image instance.
                The image to be shown on the left side of the diagram.
            pred_prob : np.array of float
                Numpy array containing all the desired predictions saved as floats.
            pred_labels : list of strings
                List containing the class names which have to be shown on the diagram.
    """
    plt.figure(figsize=(10, 6))     # Whole diagram size.
    plt.subplot(1, 2, 1)            # Divide into two parts- left and right. Choose the left side at the same moment.
    plt.imshow(image)               # Show the image on the left side.
    # The diagram title will be the best prediction value.
    plt.title(pred_labels[0] + ", {:5.2f}%".format(100 * pred_prob[0]))
    plt.subplot(1, 2, 2)            # Select the right side.
    plot_values(pred_prob, pred_labels)     # Create the bar diagram on the right side.
    plt.show()      # Show the whole diagram.


def plot_hist(training_data):
    plt.plot(training_data.history["accuracy"])
    plt.plot(training_data.history["val_accuracy"])
    plt.title("Přesnost modelu")
    plt.ylabel("přesnost")
    plt.xlabel("epochy")
    plt.legend(["trénování", "validace"], loc="upper left")
    plt.show()


def plot_two_models_training(accuracy1, accuracy2, model1_name, model2_name, y_label=""):
    bar_width = 0.35
    number_of_items = len(accuracy1)
    plt.subplots(figsize=(12, 8))

    # Set X-axis bar positions
    bar1 = np.arange(number_of_items)
    bar2 = [x + bar_width for x in bar1]

    cyber_blue = (0, 0.875, 1, 1)   # (R, G, B, alpha)
    cyber_purple = (0.42, 0, 0.7, 1)

    # Add the actual graph bars
    plt.bar(bar1, accuracy1, color=cyber_blue, width=bar_width,
            edgecolor='black', label=model1_name)
    plt.bar(bar2, accuracy2, color=cyber_purple, width=bar_width,
            edgecolor='black', label=model2_name)

    # Add the X and Y axis description
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel(y_label, fontsize=13)
    plt.xticks([r + bar_width / 2 for r in range(number_of_items)], range(number_of_items))

    # Show graph legend
    plt.legend()
    # Show the whole graph
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
