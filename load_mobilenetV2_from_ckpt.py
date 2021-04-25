import tensorflow as tf
import mobilenetV2_init as init


def get_model_from_ckpt(ckpt_dir):
    latest = tf.train.latest_checkpoint(ckpt_dir)
    model = init.build_model(8, 224)
    model.load_weights(latest)
    return model


def test_loaded_model(model, test_images, test_labels):
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    print("Restored model, loss: {:5.2f}%".format(100 * loss))
