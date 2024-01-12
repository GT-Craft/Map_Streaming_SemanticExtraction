import urllib.request
import matplotlib.pyplot as plt
from PIL import Image
import utils
import Massachusetts_utils as mutils
import numpy as np
import struct

road_file = "./road_gt.png"
building_file = "./building_gt.png"

road_img = Image.open(road_file)
building_img = Image.open(building_file)

# PIL image -> numpy array
road_img = np.array(road_img)
building_img = np.array(building_img)

road_img = utils.resize_img(road_img, road_img.shape[0]//10, road_img.shape[1]//10)
building_img = utils.resize_img(building_img, building_img.shape[0]//10, building_img.shape[1]//10)
utils.visualize_mask2D(road_img)
utils.visualize_mask2D(building_img)


road_prior_mask = utils.generate_priority_mask(road_img, 127)
building_prior_mask = utils.generate_priority_mask(building_img, 127)

road_mask_list = utils.convert_2Dmask_1Dlist(road_prior_mask)
building_mask_list = utils.convert_2Dmask_1Dlist(building_prior_mask)

with open(f'road_mask.bin', 'wb') as bin_file:
    for value in road_mask_list:
        bin_file.write(struct.pack('B', value))

with open(f'building_mask.bin', 'wb') as bin_file:
    for value in building_mask_list:
        bin_file.write(struct.pack('B', value))
