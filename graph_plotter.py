import numpy as np                  # defines the Numpy arrays we are using below (and in the other scripts)
import matplotlib.pyplot as plt     # main Python plotting library

# This script contains only functions that create different graphs used by the other scripts.


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
            probabilities : np.array of floats
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
    """Creates a graph which shows the whole training process measured in training and validation accuracy
    throughout all epochs.

        Parameters
            ----------
            training_data : `History` object
                The training history returned directly from the model `fit` function which does the training process.
    """
    plt.plot(training_data.history["accuracy"])             # plot the training accuracy
    plt.plot(training_data.history["val_accuracy"])         # plot the validation accuracy
    plt.title("Model accuracy")                             # title of the graph
    plt.ylabel("accuracy (%)")                              # the label of the y-axis
    plt.xlabel("epochs")                                    # the label of the x-axis
    plt.legend(["train", "validation"], loc="upper left")   # the legend of the graph
    plt.show()                                              # show the graph to the user


def plot_two_models_training(values1, values2, model1_name, model2_name, x_label="Iteration", y_label=""):
    """Creates a bar diagram which compares data taken from 2 different neural network models. The shown data may be
    arbitrary.

        Parameters
            ----------
            values1 : np.array of floats
                Data gathered from the first model training (top-1 accuracy or top-1 loss as an example).
            values2 : np.array of floats
                Data gathered from the second model training (top-1 accuracy or top-1 loss as an example).
            model1_name : str
                The actual name of the first model (will be shown in the legend).
            model2_name : str
                The actual name of the second model (will be shown in the legend).
            x_label : str, optional
                The label of the x-axis (default: "Iteration").
            y_label : str, optional
                The label of the y-axis (default: "").
    """
    bar_width = 0.35                        # the width of the different bars
    number_of_items = len(values1)          # get the total number of values
    if number_of_items != len(values2):     # if the number of samples in the both input arrays doesn't match, return
        return
    plt.figure(figsize=(12, 8))           # determine the size of the whole diagram

    # Set all X-axis bar positions
    bars1 = np.arange(number_of_items)      # bars corresponding to the data from the first model
    bars2 = [x + bar_width for x in bars1]  # bars corresponding to the data from the second model

    # Defining the bar colors.
    cyber_blue = (0, 0.875, 1, 1)           # (R, G, B, alpha)
    cyber_purple = (0.42, 0, 0.7, 1)

    # Create the actual graph bars via the already defined variables.
    plt.bar(bars1, values1, color=cyber_blue, width=bar_width,
            edgecolor='black', label=model1_name)
    plt.bar(bars2, values2, color=cyber_purple, width=bar_width,
            edgecolor='black', label=model2_name)

    # Add the X and Y axis description and set its font size.
    plt.xlabel(x_label, fontsize=13)
    plt.ylabel(y_label, fontsize=13)
    # Set the label frequency of the x-axis values.
    plt.xticks([val_index + bar_width / 2 for val_index in range(number_of_items)], range(number_of_items))

    # Show graph legend.
    plt.legend()
    # Show the whole graph.
    plt.show()


def show_9_image_predictions(images, predictions, labels, class_names):
    """Shows 9 images. Above each image is shown its prediction by the model and its true label.

        Parameters
            ----------
            images : np.array of Image instances
                9 images which have been evaluated. A batch of 9 images can be created to make the evaluation faster.
            predictions : np.array of np.arrays of floats
                Prediction data generated directly by the model. There is one array of predictions for each image.
            labels : np.array of np.arrays of int
                Labels directly from the dataset. Each image label is in the form of an array which has a 1 on the index
                corresponding to the right class. All other values are 0.
            class_names : np.array of str
                The actual names of the classes which will be shown as part of the image titles.
    """
    plt.figure(figsize=(10, 10))    # size of the figure
    for i in range(9):              # 9 images, cycle of 9 parts
        plt.subplot(3, 3, i + 1)    # partition into the whole plot into 9 parts and select the (i+1)-st
        plt.imshow(images[i].astype("uint8"))               # show the i-th image
        best_prediction_inx = np.argmax(predictions[i])     # get the index of the best prediction
        right_label = np.argmax(labels[i])                  # get the true label
        # Format the title. First there will be the predicted value and then there will be the right label.
        plt.title("Predicted: " + class_names[best_prediction_inx] + '\n' + "Right: " + class_names[right_label])
        plt.axis("off")                                     # Turn off axis lines and labels.
    plt.show()  # Show the plot.
