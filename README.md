# The server project

## Introduction

Contains different Python scripts meant to train, evaluate, convert and compare CNN models. Uses directly TensorFlow 2 and its Keras API. The main TensorFlow 2 installation links are placed below. There are also links to some official documentations which have been used.

This code is part of my (Georgi S. Georgiev's) bachelor thesis "Neural network architectures for mobile devices" so with each usage it has to be cited. The thesis itself is in czech and I have written it as a student of the Charles University in Prague. Permanent link to the thesis can be found here: http://hdl.handle.net/20.500.11956/148274. If there are enough requests, I may try to translate the thesis to English. For more information see the English Abstract of the thesis which can be found in the link above as well.



## Structure

* average_accuracy_difference_counter : Script that can read model accuracy difference data from several text files and count the average value which is
  written on the standard console output.
  
* ckpt_to_saved_model : Convert model saved as checkpoints to SavedModel.

  This script is meant to be run directly. It converts the model saved as checkpoints to the SavedModel format. The main advantage of the SavedModel format is that it contains the whole model structure inside. So it does not require the model to be fully initialized before loading in its weight values.

* compare_two_models : Compare two CNN models via the cross-validation algorithm.

  This script allows the user to directly compare the MobileNetV2 and the EfficientNetB0 models. As a comparison tool the cross-validation algorithm was chosen.

* dataset_loader : This script only defines some functions which load the requested dataset or its class labels.

* eval_one_image : Script that evaluates just one image.

  This script directly loads the requested model and then evaluates it on one image. This is a script which was created only with testing purposes.

* fine_tune_efficientnetB0 : This script directly starts the process of EfficientNetB0 fine-tuning. At the beginning of the script the model is being initialized, then its pretrained version is being loaded and in the end the fine-tuning takes place.

* fine_tuner : This script defines the main fine-tuning method which can fine-tune both MobileNetV2 and EfficientNetB0.

* flask_quickstart : This script directly starts the flask server which runs the neural network model and evaluates it as well. The script serves as a communication tool between the application and the model. It uses the HTTP communication protocol.

* graph_plotter : This script contains only functions that create different graphs used by the other scripts.

* init_efficientnetB0 : Defines only one function which initialises the base EfficientNetB0 model.

* init_mobilenetV2 : Defines only one function which initialises the base MobileNetV2 model.

* model_batch_testing : This script allows directly to evaluate the model on a batch of 9 images.

* model_controller : Definition of the Model Controller class. This script is not meant to be run directly. It defines the whole Model Controller class which stores a CNN model. Contains multiple functions that evaluate and save the model as well.

* txt_to_graph_plotter : This script directly reads a text file in a specific format containing training data of two models and creates a bar diagram which contains all the read data.



## Installation Links (last updated: 20.07.2021)

* TensorFlow 2 installation guide : https://www.tensorflow.org/install
* Nvidia CUDA installation guide : https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
* Python 3 download and installation : https://www.python.org/downloads/
* PyCharm IDE installation guide : https://www.jetbrains.com/help/pycharm/installation-guide.html



## Documentation links (last updated: 20.07.2021)

* Official Keras documentation (part of TensorFlow 2) : https://keras.io/
* TensorFlow 2 quickstart : https://www.tensorflow.org/tutorials
* The plotting tool : https://matplotlib.org/
* Flask server official : https://flask.palletsprojects.com/en/2.0.x/

