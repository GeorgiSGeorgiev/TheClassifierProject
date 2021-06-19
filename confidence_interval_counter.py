import math
import numpy as np
import matplotlib.pyplot as plt

k = 5
t = 2.776
data_file_path = "D:\\Gogi\\UK\\MFF_UK_2020_2021\\TheProject\\Data\\Comparison\\loss_rates_25x25_50x100.txt"
data = np.loadtxt(data_file_path, skiprows=1, delimiter=", ")

x1 = data[:, 0]
y1 = data[:, 1]

err_difference = []
total_difference = 0
for i in range(k):
    err_difference.append(x1[i] - y1[i])
    total_difference += err_difference[i]

mean_err_difference = total_difference / k

tmp_variance_var = 0
for i in range(k):
    tmp_variance_var = (err_difference[i] - mean_err_difference) ** 2

variance = tmp_variance_var / k
deg_of_freedom = math.sqrt(variance / (k-1))

conf_interval = [mean_err_difference - deg_of_freedom * t, mean_err_difference + deg_of_freedom * t]

print(conf_interval)
