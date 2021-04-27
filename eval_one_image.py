#!/usr/bin/python

import sys
from model_controller import ModelController

model_name = "EfficientNetB0"
model_format = "ckpt"
model_path = "efficientnetB0_training"
dataset_path = None     # add label reading
image_path = None

if len(sys.argv) == 2:
    dataset_path = sys.argv[0]
    image_path = sys.argv[1]
if len(sys.argv) == 5:
    model_name = sys.argv[0]
    model_format = sys.argv[1]
    model_path = sys.argv[2]
    dataset_path = sys.argv[3]  # add label reading
    image_path = sys.argv[4]

# ckpt_path = "mobilenetV2_training"
# ckpt_path = "efficientnetB0_training"
# saved_path = "D:\\Gogi\\UK\\MFF_UK_2020_2021\\TheProject\\New_Progression\\mobilenetV2_SavedModel"
# saved_path = "D:\\Gogi\\UK\\MFF_UK_2020_2021\\TheProject\\New_Progression\\efficientnetB0_SavedModel"

model_ctrl = ModelController(model_name)
# model_ctrl.load_model_from_ckpt(ckpt_path)
if model_format == "ckpt":
    model_ctrl.load_model_from_ckpt(model_path)
else:
    model_ctrl.load_saved_model(model_path)

model_ctrl.eval_image(image_path, dataset_path)

