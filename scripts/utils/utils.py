import urllib.request
import matplotlib.pyplot as plt
from PIL import Image
import json
import math
import segmentation_models_pytorch as smp
import numpy as np

EPOCHS = [3, 6, 9, 12, 15]
THRESHOLDS = [0.3, 0.5, 0.7]
LRS = [0.00008, 0.001]
# LOSSES = {'cross_entropy_loss': smp.utils.losses.CrossEntropyLoss(), 'dice_loss': smp.utils.losses.DiceLoss()}
LOSSES = {'dice_loss': smp.utils.losses.DiceLoss()}
ROAD_DIR = "/home/jin/mnt/Data/Massachusett_Dataset/Roads/"
BUILDING_DIR = "/home/jin/mnt/Data/Massachusett_Dataset/Buildings/"
ROAD_CLASSES = ['background', 'road']
BUILDING_CLASSES = ['background', 'building']

ROAD_PRIORITY = 1
BUILDING_PRIORITY = 2


def read_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        json_dict = json.load(json_file)
        return json_dict


# return map resolution in meters/pixel
def get_map_resolution(lat_deg, zoomlevel):
    lat_rad = lat_deg * math.pi / 180.0
    mpp = 156543.04 * math.cos(lat_rad) / math.pow(2, zoomlevel)
    return mpp


# url = "https://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/33.741836,-84.416501/16?mapSize=1000,1000&mapLayer=Basemap&format=png&mapMetadata=0&key=" + map_key
def get_map_api(map_patch_json, map_key):
    api_url = ""
    api_url += map_patch_json['base']
    api_url += map_patch_json['latitude'] + ',' + map_patch_json['longitude'] + '/'
    api_url += str(map_patch_json['zoom']) + '?'
    api_url += 'mapSize=' + map_patch_json['mapSize'] + '&'
    api_url += 'mapLayer=' + map_patch_json['mapLayer'] + '&'
    api_url += 'format=' + map_patch_json['format'] + '&'
    api_url += 'mapMetadata=' + map_patch_json['mapMetadata'] + '&'
    api_url += 'key=' + map_key
    return api_url


def get_out_filename(map_patch_json):
    output_mapfile = ''
    output_mapfile += map_patch_json['latitude'] + '_' + map_patch_json['longitude'] + '_' + str(map_patch_json['zoom']) + '.' + map_patch_json['format']
    return output_mapfile


import cv2
def contrast_img_file(img_file):
    image = cv2.imread(img_file)

    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab) # converting to LAB color space
    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))
    # Converting image from LAB Color model to BGR color spcae
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB) # BGR2RGB

    return image


# sampe shape for both lowp_mask and hihgp_mask
def mask_priority(lowp_mask, highp_mask):
    for y in range(highp_mask.shape[0]):
        for x in range(highp_mask.shape[1]):
            if highp_mask[y, x] != 0:
                lowp_mask[y, x] = 0
    return lowp_mask


def generate_priority_mask(mask, priority):
    generated_mask = np.zeros([int(mask.shape[0]), int(mask.shape[1])], dtype=np.uint8)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x].shape.__len__() >= 1:
                if mask[y, x][0] > 0:
                    generated_mask[y, x] = priority
            else:
                if mask[y, x] > 0:
                    generated_mask[y, x] = priority
    return generated_mask


def convert_2Dmask_1Dlist(mask):
    mask_list = []
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            mask_list.append(mask[y, x])
    return mask_list


def convert_1Dlist_2Dmask(list_1d, x, y):
    mask = np.zeros([y, x], dtype=np.uint8)
    for i in range(len(list_1d)):
        yy = i // x
        xx = i % x
        mask[yy, xx] = list_1d[i]
    return mask

def visualize_mask2D(mask):
    plt.imshow(mask)
    plt.show()
