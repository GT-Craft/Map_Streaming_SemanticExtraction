import os, sys
import json
import pandas as pd
import numpy as np
import cv2, PIL
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp # https://github.com/qubvel/segmentation_models.pytorch

import utils.Massachusetts_utils as mutils
import utils.utils as utils

config = json.loads(open('config.json').read())
project_dir = config["PROJECT_DIR"]
model_dir = project_dir + 'pretrained_models/road/'
dataset_dir = config["DATASET_DIR"]
road_dataset = dataset_dir + 'Roads/'
CLASSES = ['background', 'road']

m_helper = mutils.Massachusetts_Helper()
m_helper.initialize(road_dataset)
selected_class_rgb_values = m_helper.select_classes(CLASSES)


#######################  Training #########################
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'

model = smp.Unet(encoder_name=ENCODER,
                 encoder_weights=ENCODER_WEIGHTS,
                 classes=len(CLASSES),
                 activation=ACTIVATION)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


train_dataset = mutils.Massachusetts_Dataset(m_helper.x_train_dir, m_helper.y_train_dir,
                                             augmentation=mutils.get_training_augmentation(),
                                             preprocessing=mutils.get_preprocessing(preprocessing_fn),
                                             class_rgb_values=selected_class_rgb_values)
valid_dataset = mutils.Massachusetts_Dataset(m_helper.x_valid_dir, m_helper.y_valid_dir,
                                             augmentation=mutils.get_validation_augmentation(),
                                             preprocessing=mutils.get_preprocessing(preprocessing_fn),
                                             class_rgb_values=selected_class_rgb_values)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)


TRAINING = True
# TRAINING = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for loss in utils.LOSSES:
    for EPOCH in utils.EPOCHS:
        for THRESHOLD in utils.THRESHOLDS:
            for LR in utils.LRS:
                metrics = [smp.utils.metrics.IoU(threshold=THRESHOLD), ]
                optimizer = torch.optim.Adam([
                    dict(params=model.parameters(), lr=LR),
                ])

                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5)

                model_name = f'road_{loss}_ep{EPOCH}_th{THRESHOLD}_lr{LR}.pth'

                if os.path.exists(model_dir + model_name):
                    model = torch.load(model_dir + model_name, map_location=DEVICE)

                train_epoch = smp.utils.train.TrainEpoch(model, utils.LOSSES[loss], metrics, optimizer, DEVICE, verbose=True)
                valid_epoch = smp.utils.train.ValidEpoch(model, utils.LOSSES[loss], metrics, DEVICE, verbose=True)

                if TRAINING:
                    best_iou_score = 0.0
                    train_logs, valid_logs = [], []

                    for i in range(0, EPOCH):
                        print(f'\n EPOCH {i}')
                        train_log = train_epoch.run(train_loader)
                        valid_log = valid_epoch.run(valid_loader)
                        train_logs.append(train_log)
                        valid_logs.append(valid_log)

                        if best_iou_score < valid_log['iou_score']:
                            best_iou_score = valid_log['iou_score']
                            torch.save(model, model_dir + model_name)
                            print(f"Model Saved! {model_dir}{model_name}")


                ####################### Testing #########################
                test_model = torch.load(model_dir + model_name, map_location=DEVICE)
                print(f"Testing {model_dir}{model_name}")

                test_dataset = mutils.Massachusetts_Dataset(dataset_dir + "test/", dataset_dir + 'test',
                                                            augmentation=mutils.get_validation_augmentation(),
                                                            preprocessing=mutils.get_preprocessing(preprocessing_fn),
                                                            class_rgb_values=selected_class_rgb_values)
                test_dataloader = DataLoader(test_dataset)

                test_idx = 0
                for test_img, test_mask in test_dataset:
                    test_tensor = torch.from_numpy(test_img).to(DEVICE).unsqueeze(0)
                    pred_mask = test_model(test_tensor)
                    pred_mask = pred_mask.detach().squeeze().cpu().numpy()

                    pred_img = np.transpose(pred_mask, (1, 2, 0))
                    pred_mask = mutils.colour_code_segmentation(mutils.reverse_one_hot(pred_img), selected_class_rgb_values)

                    # Save image to a file along the model
                    cv2.imwrite(f'{project_dir}results/{test_idx}_' + model_name[:-4] + '.png', pred_mask)
                    test_idx += 1

