from model_controller import ModelController
# Flask and werkzeug represent the server on which the model will be run.
import flask
import werkzeug

# The whole algorithm inspiration was taken from the following source:
# https://heartbeat.fritz.ai/uploading-images-from-android-to-a-python-based-flask-server-691e4092a95e

# The path to the checkpoints folder. The last checkpoint will be automatically found and loaded.
# Can be adjusted by the user. Although it is important that the checkpoints match the model type from the controller.
ckpt_path = "efficientnetB0_training_2_0"
# Path to the dataset. The algorithms gets the labels from the structure of the dataset.
dataset_path = "D:\\tmp\\cars4\\car_photos_4"

# Create a model controller for EfficientNetB0. Can be adjusted by the user to MobileNetV2.
model_ctrl = ModelController("EfficientNetB0")
# Load the checkpoints from the ckpt_path.
model_ctrl.load_model_from_ckpt(ckpt_path)
# Start the Flask server. The parameter says which package the server has to use. "__name__" is the default setting
# and it is sufficient in our case because because we aren't using any external packages.
app = flask.Flask(__name__)


# The function below is triggered by a specific URL and the body of the function handles the remote request.
# The route() decorator tells Flask what URL should trigger our function. It searches for GET and POST methods.
@app.route('/', methods=['GET', 'POST'])
def handle_request():
    """Handles a remote request get by the Flask server. In our case it triggers the model evaluation.

        Returns
            ----------
            res_str : str
                The result of the model evaluation. It is saved as a string which is already in its final format so that
                the other side (in our case the mobile application) doesn't have to reformat the string which may be
                a costly operation on a device with limited computational power.
    """
    # Request the image file from Flask.
    image_file = flask.request.files['image']
    # Get a secure version of the original image filename.
    filename = werkzeug.utils.secure_filename(image_file.filename)
    print("\nReceived image File name : " + image_file.filename)
    # Save locally the image file. The file will be saved to the directory where is located this script.
    image_file.save(filename)

    res_str = ""    # init the result string
    to_be_shown = 3     # best predictions to be shown, CAN be adjusted by the user
    # Evaluate the model. Its input will be the saved image. Don't show a diagram with the results,
    # because may reduce the speed of the server application and normally has to be closed by the user before proceeding
    # and sending the result to the other side.
    pred_prob, pred_labels = model_ctrl.eval_image(filename, dataset_path, to_be_shown, plot=False)

    # Format the result string the required way. May be changed by the user.
    for i in range(to_be_shown):
        # On every row of the result string show: prediction class + a float value in the range [0;100]% rounded to
        # two decimal positions.
        res_str += pred_labels[i] + ", {:5.2f}%".format(100 * pred_prob[i]) + "\n"
    return res_str


# Run the server on port 5000. '0.0.0.0' means that the server will listen on all registered IP addresses of the device.
app.run(host="0.0.0.0", port=5000)
