from model_controller import ModelController

# ckpt_path = "mobilenetV2_training"
# saved_path = "D:\\Gogi\\UK\\MFF_UK_2020_2021\\TheProject\\New_Progression\\mobilenetV2_SavedModel_3"
ckpt_path = "efficientnetB0_training"
saved_path = "D:\\Gogi\\UK\\MFF_UK_2020_2021\\TheProject\\New_Progression\\efficientnetB0_SavedModel"

# model_ctrl = ModelController("MobileNetV2")
model_ctrl = ModelController("EfficientNetB0")

model_ctrl.load_model_from_ckpt(ckpt_path)
model_ctrl.save_as_saved_model(saved_path)
