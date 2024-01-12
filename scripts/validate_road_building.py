import urllib.request
import matplotlib.pyplot as plt
from PIL import Image
import json
import torch
import os
import cv2
import numpy as np
import struct
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

import utils.utils as utils
import utils.dataset_utils as dutils
import utils.Massachusetts_utils as mutils
from utils.dataset_utils import Single_Dataset

config = json.loads(open('config.json').read())
project_dir = config["PROJECT_DIR"]
sample_data_path = project_dir + 'sample_data/'

## Get streamed image
json_file_path = project_dir + 'sample_data/map_patch.json'
map_patch_json = utils.read_json(json_file_path)

map_size = map_patch_json['mapSize'].split(',')
map_size = [int(map_size[0]), int(map_size[1])]

output = utils.get_out_filename(map_patch_json)
output = project_dir + 'sample_data/' + output
I = Image.open(output)

# Read image & gt masks
# img = cv2.cvtColor(cv2.imread(output), cv2.COLOR_BGR2RGB)
img = utils.contrast_img_file(output)
road_mask = cv2.cvtColor(cv2.imread(sample_data_path + 'road_gt.png'), cv2.COLOR_BGR2RGB)
building_mask = cv2.cvtColor(cv2.imread(sample_data_path + 'building_gt.png'), cv2.COLOR_BGR2RGB)

# Get RGB values for each class
m_helper = mutils.Massachusetts_Helper()
m_helper.initialize(utils.ROAD_DIR)
road_class_rgb_values = m_helper.select_classes(utils.ROAD_CLASSES)
m_helper.initialize(utils.BUILDING_DIR)
building_class_rgb_values = m_helper.select_classes(utils.BUILDING_CLASSES)

# Preprocess dataset & loader
preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50', 'imagenet')
test_road_dataset = dutils.Single_Dataset(img, road_mask,
                                          class_rgb_values=road_class_rgb_values,
                                          augmentation=mutils.get_validation_augmentation(),
                                          preprocessing=dutils.get_preprocessing(preprocessing_fn))
test_building_dataset = dutils.Single_Dataset(img, building_mask,
                                              class_rgb_values=building_class_rgb_values,
                                              augmentation=mutils.get_validation_augmentation(),
                                              preprocessing=dutils.get_preprocessing(preprocessing_fn))

test_road_loader = DataLoader(test_road_dataset, batch_size=1, shuffle=False, num_workers=4)
test_building_loader = DataLoader(test_building_dataset, batch_size=1, shuffle=False, num_workers=4)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS = utils.LOSSES['dice_loss']
THRESHOLD = utils.THRESHOLDS[0]
LR = utils.LRS[0]
pretrained_model_path = project_dir + 'pretrained_models/'

for EPOCH in utils.EPOCHS:
    metrics = [smp.utils.metrics.IoU(threshold=THRESHOLD), ]
    road_model_index = f'dice_loss_ep{EPOCH}_th{THRESHOLD}_lr{LR}.pth'
    building_model_index = f'dice_loss_ep{EPOCH}_th{THRESHOLD}_lr{LR}.pth'

    if not os.path.exists(pretrained_model_path + 'road/road_' + road_model_index):
        print(pretrained_model_path + 'road/road_' + road_model_index + ' does not exist')
        continue
    if not os.path.exists(pretrained_model_path + 'building/building_' + building_model_index):
        print(pretrained_model_path + 'building/building_' + building_model_index + ' does not exist')
        continue

    road_model = torch.load(pretrained_model_path + 'road/road_' + road_model_index, map_location=DEVICE)
    building_model = torch.load(pretrained_model_path + 'building/building_' + building_model_index, map_location=DEVICE)

    road_model_epoch = smp.utils.train.ValidEpoch(road_model, LOSS, metrics, DEVICE, verbose=True)
    building_model_epoch = smp.utils.train.ValidEpoch(building_model, LOSS, metrics, DEVICE, verbose=True)

    road_logs = road_model_epoch.run(test_road_loader)
    building_logs = building_model_epoch.run(test_building_loader)

    road_iou = road_logs['iou_score']
    building_iou = building_logs['iou_score']

    # print with .2 precision
    print(f'EPOCH {EPOCH}, road_iou: {road_iou:.2f}, building_iou: {building_iou:.2f}')

