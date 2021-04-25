from model_controller import ModelController

image_path = "D:\\Gogi\\Photos\\Cars\\golf3.jpeg"
dataset_path = "D:\\tmp\\cars3\\car_photos_3"
ckpt_path = "mobilenetV2_training"

model_ctrl = ModelController("MobileNetV2")
model_ctrl.load_model_from_ckpt(ckpt_path)
model_ctrl.eval_image(image_path, dataset_path)
