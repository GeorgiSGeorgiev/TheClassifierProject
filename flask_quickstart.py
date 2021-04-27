from model_controller import ModelController
import flask
import werkzeug

# https://heartbeat.fritz.ai/uploading-images-from-android-to-a-python-based-flask-server-691e4092a95e

ckpt_path = "efficientnetB0_training"
dataset_path = "D:\\tmp\\cars3\\car_photos_3"

model_ctrl = ModelController("EfficientNetB0")
model_ctrl.load_model_from_ckpt(ckpt_path)
app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def handle_request():
    image_file = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(image_file.filename)
    print("\nReceived image File name : " + image_file.filename)
    image_file.save(filename)

    res_str = ""
    to_be_shown = 3
    pred_prob, pred_labels = model_ctrl.eval_image(filename, dataset_path, to_be_shown, plot=False)
    for i in range(to_be_shown):
        res_str += pred_labels[i] + ", {:5.2f}%".format(100 * pred_prob[i]) + "\n"
    return res_str


app.run(host="0.0.0.0", port=5000)
