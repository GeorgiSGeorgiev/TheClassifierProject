"""Compare two CNN models via the cross-validation algorithm

This script allows the user to directly compare the MobileNetV2 and the EfficientNetB0 models. As a comparison tool
the cross-validation algorithm was chosen.

The script is activated directly and doesn't take any arguments. The reason for the decision is the great number of
adjustable parameters that the algorithm has. Although the code is well commented and every step inside is described.

To run the script there have to be installed the libraries `numpy` and `TensorFlow` within the Python
environment you are running this script in. The script depends on the `graph_plotter`, `dataset_loader` and `fine_tuner`
scripts as well. They are part of the project and are inside the project package.

"""

# Created by: Georgi Stoyanov Georgiev.
# as part of the "Neural network architectures for mobile devices" bachelor thesis

# First set all of the random number generator seeds. Some seeds may be calculated using the import time of the
# libraries. That's why we are setting the seeds right at the beginning.
import os
os.environ['PYTHONHASHSEED'] = str(9)
# Set the Python hash seed to a constant. The Python hashing function can be used by other random number generators.

# Set the numpy seed. Numpy library allows to convert tensors to numpy arrays which resemble normal arrays.
from numpy.random import seed
seed(9)
# Import the TensorFlow 2 and set its seed.
import tensorflow as tf
tf.random.set_seed(9)


import graph_plotter    # Used to draw learning diagrams.
import dataset_loader   # Used to load the dataset.
import fine_tuner       # Used to fine-tune the models as part of the cross-validation algorithm.

# The following 2 scripts were used to measure the delta time the algorithm has been running.
import datetime
import time

print('Starting the script.')

print('Starting the timer.')
start = time.perf_counter()

# Setting the models and the input format. If you change the model input shapes in the initialization process, you
# will have to change the value of image_dim as well because they have to match.
model_name_1 = "MobileNetV2"
model_name_2 = "EfficientNetB0"
data_path = "D:\\tmp\\cars4\\car_photos_4"
image_dim = 224
batch_size = 64
image_size = (image_dim, image_dim)

# Initialization of the result containers.
accuracy_list_1 = []
accuracy_list_2 = []
loss_list_1 = []
loss_list_2 = []
accuracy_diff = []
loss_diff = []

print('Preprocessing the dataset...')
# Load the whole dataset at once.
dataset = dataset_loader.get_united_dataset(data_path, image_size, batch_size)
# Save the total number of batches to the total_batch_number variable.
total_batch_number = tf.data.experimental.cardinality(dataset)
result = []
interval_count = 5  # Number of the data intervals. One of the intervals will serve as validation data in every step
# of the cross-validation algorithm.

for i in range(interval_count):
    # (total_batch_number // interval_count) gets how many batches are there in one interval.
    # The row below appends the batches from the first interval of batches to the result list variable.
    # The whole interval is taken as one independent list.
    result.append(dataset.take(total_batch_number // interval_count))
    # We have to remove the batches (skip them) that are already in the result list from the original dataset.
    dataset = dataset.skip(total_batch_number // interval_count)
# So the for cycle above breaks the dataset into many segments (intervals) and stores these
# segments into the `result` list.

print('Dataset preprocessing finished successfully.')
train = None
validation = None

print()
print('STARTING THE ALGORITHM')
print()

# for each interval:
for i in range(interval_count):
    print()
    print('*********************')
    print('STARTING ITERATION %s' % i)
    print('---------------------')
    train = None
    validation = result[i]    # the i-th interval will be the validation dataset
    for j in range(interval_count):
        if i == j:  # Don't add the validation dataset to the training dataset
            continue
        if train is None:
            train = result[j]   # The train dataset is still empty, so initialize it.
        else:
            # Else the train dataset is not empty, so add new dataset segments to it.
            # Concatenate can merge two different lists into one.
            train = train.concatenate(result[j])
    # First fine-tune the MobileNetV2 model.
    # IMPORTANT! User can change the epochs_classifier , epochs_train, plot and the layers_to_be_trained variables.
    # `epochs_classifier` determines for how many epochs the top classification layer will be trained.
    # `epochs_train` determines for how many epochs the top layers of the network will be trained.
    # `plot` determines whether to draw training diagrams of the both classification layer and the top layers training
    # `plot` doesn't have to be True because at the end of the algorithm there is created a general comparison diagram
    # `layers_to_be_trained` determines the number of top layers of the network to be trained
    # Important detail is that MobileNetV2 originally has totally 53 trainable layers.
    acc1, loss1 = fine_tuner.fine_tune(train, validation, model_name="MobileNetV2", image_dimension=image_dim,
                                       epochs_classifier=15, epochs_train=15, reverse_save_freq=0, plot=False,
                                       layers_to_be_trained=53)
    print('MobileNetV2 successfully trained, proceeding with the training of EfficientNetB0.')
    # Then fine-tune the EfficientNetB0 model.
    # The settings are the same as above.
    # Important detail is that EfficientNetB0 originally has totally 237 trainable layers.
    acc2, loss2 = fine_tuner.fine_tune(train, validation, model_name="EfficientNetB0", image_dimension=image_dim,
                                       epochs_classifier=15, epochs_train=15, reverse_save_freq=0, plot=False,
                                       layers_to_be_trained=237)
    print('EfficientNetB0 successfully trained.')
    print('Comparing and saving the results to the internal memory.')
    accuracy_list_1.append(acc1)    # Add the accuracy measured in the last MobileNetV2 epoch to the accuracy_list_1.
    accuracy_list_2.append(acc2)    # Add the accuracy measured in the last EfficientNetB0 epoch to the accuracy_list_2.
    loss_list_1.append(loss1)       # Add the loss measured in the last MobileNetV2 epoch to the loss_list_1.
    loss_list_2.append(loss2)       # Add the loss measured in the last EfficientNetB0 epoch to the loss_list_2.

    # Count the differences between the values from above. Used as a direct comparison tool.
    accuracy_diff.append(acc2 - acc1)
    loss_diff.append(loss1 - loss2)

# The cross-validation algorithm has ended. Count the time which the algorithm has been running.
elapsed_seconds = time.perf_counter() - start
# Change the measuring format from seconds to hours:minutes:seconds:milliseconds.
delta_time = str(datetime.timedelta(seconds=elapsed_seconds))

print()
print('THE ALGORITHM HAS FINISHED')
print('..........................')
print('Plotting the results')

# Creating the different comparison graphs.
# The first of them shows the accuracy of the both models in each iteration of the algorithm.
graph_plotter.plot_two_models_training(accuracy_list_1, accuracy_list_2, model_name_1, model_name_2, "Accuracy")
# The second of them shows the loss of the both models in each iteration of the algorithm.
graph_plotter.plot_two_models_training(loss_list_1, loss_list_2, model_name_1, model_name_2, "Loss")
# The third of them shows the accuracy and the loss difference of the both models in each iteration of the algorithm.
graph_plotter.plot_two_models_training(accuracy_diff, loss_diff,
                                       "Accuracy difference", "Loss difference", "Differences")
# Next the counted results are saved to the specified below files. The user can set the file paths on his own.
print('Saving the results to the specified location.')
with open('accuracy_rates_6.txt', 'w') as file:
    # The data are saved into 2 columns. Each column is for one of the models.
    # Every row represents the data from one iteration.
    file.write(f"{model_name_1}, {model_name_2}\n")
    for i in range(len(accuracy_list_1)):
        file.write(f"{accuracy_list_1[i]}, {accuracy_list_2[i]}\n")

with open('loss_rates_6.txt', 'w') as file:
    file.write(f"{model_name_1}, {model_name_2}\n")
    for i in range(len(loss_list_1)):
        file.write(f"{loss_list_1[i]}, {loss_list_2[i]}\n")

with open('direct_comparisons_6.txt', 'w') as file:
    file.write("Accuracy difference" + " Loss difference\n")
    for i in range(len(accuracy_diff)):
        file.write(f"{accuracy_diff[i]}, {loss_diff[i]}\n")

# Final words.
print()
print('THE SCRIPT HAS FINISHED')
print('Elapsed %.3f seconds.' % elapsed_seconds)
print('Elapsed %s ' % delta_time)

# THE END
