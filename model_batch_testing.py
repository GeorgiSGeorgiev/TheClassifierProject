from model_controller import ModelController

ckpt_path = "mobilenetV2_training"
dataset_path = "D:\\tmp\\cars3\\car_photos_3"

model_ctrl = ModelController("MobileNetV2")
model_ctrl.load_model_from_ckpt(ckpt_path)

model_ctrl.eval_validation_batch(dataset_path)
