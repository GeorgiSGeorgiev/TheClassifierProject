"""Text file graph training plotter

This script directly reads a text file in a specific format containing training data of two models and creates a
bar diagram which contains all the read data. The text file has to contain two columns of floats, each representing
the specified training data of one model (e.g. top-1 accuracy or top-1 loss). The columns have to be separated by the
", " delimiter. The first row of the text file is considered as informational (can contain the names of the models)
and is skipped. The training data from the fine-tuning scripts are saved directly into this format so don't try to
change it unless you have serious reasons to do it.

Can be started directly and doesn't take any arguments. All the settings are directly inside the script.

"""

# Created by: Georgi Stoyanov Georgiev.
# as part of the "Neural network architectures for mobile devices" bachelor thesis

import numpy as np      # The data are represented as Numpy arrays inside the memory.
import graph_plotter    # The graph plotter script from this project.

# The whole path to the text file, to be specified by the user.
data_file_path = "accuracy_rates_4_3.txt"
# The data type, also to be specified by the user.
data_type = "Accuracy (%)"

# Load the text file, skip the first row and use as delimiter
data = np.loadtxt(data_file_path, skiprows=1, delimiter=", ")
x = data[:, 0]  # get the first column
y = data[:, 1]  # get the second column

# Plot the data.
graph_plotter.plot_two_models_training(x, y, "MobileNetV2", "EfficientNetB0", y_label=data_type)
