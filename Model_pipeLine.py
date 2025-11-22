from ultralytics import YOLO
from utli import DamageClassification, ImageQuality, DamageDetection, checkReason
import cv2
from PIL import Image as PILImage
from transformers import ViTForImageClassification
import segmentation_models_pytorch_3branch as smp_3b
import torch
from crackSeg_utils import load_unet_vgg16

# initial setting
image_path = 'modelTest/test.JPG'
detection_type = 'column' 

# wall result
# cost: 690.5744173177083 9554.762852044754 0 
# classification result: ['Web', 'Web_large', 'Spalling', 'Spalling', 'Spalling', 'Spalling', 'Spalling']
# Class A Spalling and Large Web-like crack which might contain shear or huge spalling crack detected.

# column result
# cost: 31627.8392578125 9554.762852044754 0
# classification result: ['Diagonal', 'Vertical', 'Spalling', 'Spalling', 'Spalling', 'Spalling', 'Spalling']
# Class A Spalling and Diagonal crack detected.
costEstimation = True
ratio = 0.3125
device = 'cuda'


# read test image
image_cv = cv2.imread(image_path)
image_PIL = PILImage.open(image_path).convert("RGB")


# Load models
model_yolo = YOLO('Models/best.pt')

model_crackSeg = load_unet_vgg16('Models/crackSeg.pt')

model_spallingSeg = smp_3b.create_model(arch='MAnet',encoder_name='efficientnet-b6', encoder_weights='imagenet'
                            , classes=2, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to(device)
model_spallingSeg.load_state_dict(torch.load('Models/spallingSeg.pt'))
model_spallingSeg.eval()

def load_model_damageClassification(detectionType):
    try:
        model_path = f'Models/damageClassify/{detectionType}/'  # Adjust this path as necessary
        model = ViTForImageClassification.from_pretrained(model_path)
        return model
    except Exception as e:
        print(e)
        model = None
models_classification = {
    'wall': None,
    'column': None,
    'beam': None
}
models_classification['wall'] = load_model_damageClassification('wall')
models_classification['column'] = load_model_damageClassification('column')
models_classification['beam'] = load_model_damageClassification('beam')

def load_model_crackClassification(detectionType):
    try:
        model_path = f'Models/crackClassify/{detectionType}/'  # Adjust this path as necessary
        model = ViTForImageClassification.from_pretrained(model_path)
        return model
    except Exception as e:
        #app.logger.error(traceback.format_exc())
        model = None
models_classification_crack = {
    'wall': None,
    'column': None,
    'beam': None
}
models_classification_crack['wall'] = load_model_crackClassification('wall')
models_classification_crack['column'] = load_model_crackClassification('column')
models_classification_crack['beam'] = load_model_crackClassification('beam')


# inference
classify_result = DamageClassification(image_PIL, models_classification[detection_type])

total_result, original_img, detected_img,total_cost_crack, total_cost_spalling, total_cost_rebar = DamageDetection(image_PIL,image_cv, model_yolo,models_classification_crack[detection_type],detection_type,costEstimation,ratio, model_crackSeg, model_spallingSeg)

classType, reason = checkReason(detection_type,classify_result,total_result)


print(total_cost_crack, total_cost_spalling, total_cost_rebar)
print(classType, reason)
# add total_cost_rebar
# how app receive the return predicted cost and present it?
# add crack Classify model
# add damage Classify model(what's for?)
# add instruction about how to provide ratio for user(measure actual length of image / lengh of image)
