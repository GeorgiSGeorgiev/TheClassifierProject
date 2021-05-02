import tensorflow as tf
from model_controller import ModelController
import graph_plotter

ckpt_path = "efficientnetB0_training"
saved_path = "D:\\Gogi\\UK\\MFF_UK_2020_2021\\TheProject\\New_Progression\\efficientnetB0_SavedModel"

name1 = "MobileNetV2"
name2 = "EfficientNetB0"

accuracy1 = [0.70, 0.75, 0.81, 0.84, 0.65]
accuracy2 = [0.45, 0.45, 0.65, 0.76, 0.87]
loss1 = []
loss2 = []

# there will be for cycle which will iterate through different data and save the different values to the array
graph_plotter.plot_two_models_training(accuracy1, accuracy2, "MobileNetV2", "EfficientNetB0")
'''
with open('accuracy_rates.txt', 'w') as file:
    file.write(f"{name1}, {name2}\n")
    for i in range(len(accuracy1)):
        file.write(f"{accuracy1[i]}, {accuracy2[i]}\n")

with open('accuracy_rates.txt', 'w') as file:
    file.write(f"{name1}, {name2}\n")
    for i in range(len(loss1)):
        file.write(f"{loss1[i]}, {loss2[i]}\n")
'''