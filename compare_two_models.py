import tensorflow as tf
import graph_plotter
import dataset_loader
import fine_tuner

model_name_1 = "MobileNetV2"
model_name_2 = "EfficientNetB0"
data_path = "D:\\tmp\\cars3\\car_photos_3"
image_dim = 224
batch_size = 64
image_size = (image_dim, image_dim)

# Initialization of the result containers.
accuracy_list_1 = []
accuracy_list_2 = []
loss_list_1 = []
loss_list_2 = []

dataset = dataset_loader.get_united_dataset(data_path, image_size, batch_size)
val_batches = tf.data.experimental.cardinality(dataset)
result = []
interval_count = 5
for i in range(interval_count):
    result.append(dataset.take(val_batches // interval_count))
    dataset = dataset.skip(val_batches // interval_count)
# break the dataset into many segments and store them into a list

train = None
validation = None
for i in range(interval_count):
    train = None
    validation = result[i]    # i-th interval will be the testing dataset
    for j in range(interval_count):
        if i == j:  # don't add the testing dataset to the training dataset
            continue
        if train is None:
            train = result[j]   # train is empty, initialize it
        else:   # train is not empty, so add new dataset segments to it
            train = train.concatenate(result[j])
    acc1, loss1 = fine_tuner.fine_tune(train, validation, model_name="MobileNetV2", image_dimension=image_dim,
                                       epochs_classifier=2, epochs_train=2, reverse_save_freq=0)
    acc2, loss2 = fine_tuner.fine_tune(train, validation, model_name="EfficientNetB0", image_dimension=image_dim,
                                       epochs_classifier=2, epochs_train=2, reverse_save_freq=0)
    accuracy_list_1.append(acc1)
    accuracy_list_2.append(acc2)
    loss_list_1.append(loss1)
    loss_list_2.append(loss2)

# there will be for cycle which will iterate through different data and save the different values to the array
graph_plotter.plot_two_models_training(accuracy_list_1, accuracy_list_2, model_name_1, model_name_2)
graph_plotter.plot_two_models_training(loss_list_1, loss_list_2, model_name_1, model_name_2)
# TODO: add accuracy/loss label changer
# TODO: count the accuracy/loss difference

'''
with open('accuracy_rates.txt', 'w') as file:
    file.write(f"{model_name_1}, {model_name_2}\n")
    for i in range(len(accuracy1)):
        file.write(f"{accuracy1[i]}, {accuracy2[i]}\n")

with open('accuracy_rates.txt', 'w') as file:
    file.write(f"{model_name_1}, {model_name_2}\n")
    for i in range(len(loss1)):
        file.write(f"{loss1[i]}, {loss2[i]}\n")
'''