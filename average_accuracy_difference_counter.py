"""Average accuracy counter

Script that can read model accuracy difference data from several text files and count the average value which is
written on the standard console output.

This script is meant to be run directly and it requires only the NumPy library. It takes no input arguments.
All the settings can be modified directly in the script. The script was meant to be only a helping tool.

The text file has to contain two columns of floats. First of them are the accuracy differences and the second one are
the loss differences. The columns have to be separated by the ", " delimiter. The first row of the text file is
considered as informational (can contain the names of the columns or the names of the models) and is skipped.
The training data from the fine-tuning scripts are saved directly into this format so don't try to change it unless
you have serious reasons to do it.

"""

import numpy as np

# Paths to the files which will be used as sources for the model data.
data_file_path = ["direct_comparisons_3_1.txt", "direct_comparisons_3_2.txt", "direct_comparisons_3_3.txt",
                  "direct_comparisons_3_4.txt"]

file_number = len(data_file_path)

number_of_samples = 0
data = []
data_columns = []

# For each file do:
for i in range(file_number):
    # Get the both columns from the file and skp the first row because it contains only the column names.
    data.append(np.loadtxt(data_file_path[i], skiprows=1, delimiter=", "))
    # We are considering only the accuracy difference, so we need just the first column. Add the column to the list
    # which contains all such columns from the different files.
    data_columns.append(data[i][:, 0])
    if i == 0:
        number_of_samples = len(data[i][:, 0])  # Get the number of items in the columns.
print(number_of_samples)
result = 0

# Sum all the data together.
for i in range(file_number):
    for j in range(number_of_samples):
        result += data_columns[i][j]
print(result)
# Count the average we are looking for.
result = result / (number_of_samples * file_number)

print('Average: %s' % result)
