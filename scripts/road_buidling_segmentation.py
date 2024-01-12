import urllib.request
import matplotlib.pyplot as plt
from PIL import Image
import json
import torch
import os
import cv2
import numpy as np
import struct

import utils.utils as utils
import utils.Massachusetts_utils as mutils

config = json.loads(open('config.json').read())
project_dir = config["PROJECT_DIR"]

json_file_path = project_dir + 'sample_data/map_patch.json'
map_patch_json = utils.read_json(json_file_path)
map_size = map_patch_json['mapSize'].split(',')
map_size = [int(map_size[0]), int(map_size[1])]


output = utils.get_out_filename(map_patch_json)
output = project_dir + 'sample_data/' + output
I = Image.open(output)
# plt.imshow(I)
# plt.show()

# enhanced_img = utils.contrast_img_file(output)
# plt.imshow(enhanced_img)
# plt.show()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m_helper = mutils.Massachusetts_Helper()
m_helper.initialize(utils.ROAD_DIR)
road_class_rgb_values = m_helper.select_classes(utils.ROAD_CLASSES)

m_helper.initialize(utils.BUILDING_DIR)
building_class_rgb_values = m_helper.select_classes(utils.BUILDING_CLASSES)

ms_map_processor = mutils.Ms_Map_Processor()
img_to_seg, preprocessed_img, _ = ms_map_processor.process_map_patch(output, road_class_rgb_values)
padded_size = list(preprocessed_img.shape)[1]
pad_size = (padded_size - map_size[0]) // 2


# Parameter Grid Search
for loss in utils.LOSSES:
    for EPOCH in utils.EPOCHS:
        for THRESHOLD in utils.THRESHOLDS:
            for LR in utils.LRS:
                print('loss: ', loss, 'epoch: ', EPOCH, 'threshold: ', THRESHOLD, 'lr: ', LR)

                model_index = f'{loss}_ep{EPOCH}_th{THRESHOLD}_lr{LR}.pth'

                road_model_path = project_dir + 'pretrained_models/road/road_' + model_index
                building_model_path = project_dir + 'pretrained_models/building/building_' + model_index

                if not os.path.exists(road_model_path):
                    print(f'\t{road_model_path} does not exist')
                    continue

                if not os.path.exists(building_model_path):
                    print(f'{building_model_path} does not exist')
                    continue

                road_model = torch.load(road_model_path, map_location=DEVICE)
                building_model = torch.load(building_model_path, map_location=DEVICE)

                img_tensor = torch.from_numpy(preprocessed_img).to(DEVICE).unsqueeze(0)

                road_pred = road_model(img_tensor)
                road_pred = road_pred.detach().squeeze().cpu().numpy()
                road_pred = np.transpose(road_pred, (1, 2, 0))
                road_pred = mutils.colour_code_segmentation(mutils.reverse_one_hot(road_pred), road_class_rgb_values)

                building_pred = building_model(img_tensor)
                building_pred = building_pred.detach().squeeze().cpu().numpy()
                building_pred = np.transpose(building_pred, (1, 2, 0))
                building_pred = mutils.colour_code_segmentation(mutils.reverse_one_hot(building_pred), building_class_rgb_values)

                # remove padding from the image
                road_pred = road_pred[pad_size:map_size[0]+pad_size, pad_size:map_size[1]+pad_size, :]
                building_pred = building_pred[pad_size:map_size[0]+pad_size, pad_size:map_size[1]+pad_size, :]


                ## Each semantic mask separately stored, but exculsive to each other
                road_prior_mask = utils.generate_priority_mask(road_pred, 255)
                building_prior_mask = utils.generate_priority_mask(building_pred, 255)

                # save mask images for visualization
                cv2.imwrite(project_dir + f'results/road_mask_{EPOCH}_{THRESHOLD}_{LR}.png', road_prior_mask)

                utils.visualize_mask2D(road_prior_mask)
                utils.visualize_mask2D(building_prior_mask)

                road_mask_list = utils.convert_2Dmask_1Dlist(road_prior_mask)
                building_mask_list = utils.convert_2Dmask_1Dlist(building_prior_mask)

                out_road_mask_path = project_dir + f'results/road_mask_{EPOCH}.bin'
                out_building_mask_path = project_dir + f'results/building_mask_{EPOCH}.bin'

                with open(out_road_mask_path, 'wb') as bin_file:
                    for value in road_mask_list:
                        bin_file.write(struct.pack('B', value))

                with open(out_building_mask_path, 'wb') as bin_file:
                    for value in building_mask_list:
                        bin_file.write(struct.pack('B', value))

