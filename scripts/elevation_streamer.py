import urllib.request
import matplotlib.pyplot as plt
from PIL import Image
import os
import pymap3d
import json
import sys
from tqdm import tqdm

import utils.utils as utils
import utils.Massachusetts_utils as mutils

# python elevation_streamer.py 0 # start to 0
# python elevation_streamer.py 1000 # start to 1000

# Input argument: y_checkpoint
if len(sys.argv) == 2:
    y_checkpoint = sys.argv[1]
else:
    y_checkpoint = 0
y_checkpoint = int(y_checkpoint)
print(f"Start to y checkpoint {y_checkpoint}")

config = json.loads(open('config.json', 'r').read())
project_dir = config['PROJECT_DIR']

#############  Get the information of the target region #############
map_info_path = project_dir + 'sample_data/map_patch.json'
map_patch_json = utils.read_json(map_info_path)

map_size = map_patch_json['mapSize'].split(',')
map_size = [int(map_size[0]), int(map_size[1])] # x, y
print(f"\tMap size: {map_size}")

center_geodetic = [float(map_patch_json['longitude']), float(map_patch_json['latitude'])]
center_offset = [map_size[0] // 2, (map_size[1] // 2)-1]
mpp = utils.get_map_resolution(center_geodetic[1], int(map_patch_json['zoom']))

print(f'\tCenter geodetic: {center_geodetic}')
print(f'\tCenter pixel offset: {center_offset}')
print(f'\tMap resolution (Meters per pixel): {mpp:.2f}')

output =  project_dir + "results/temp.json"
map_key = ""
with open(project_dir + 'MapSessionConfig.txt', 'r') as f:
    map_key = f.readline()


# dataframe format x, y, long, lat, elevation, zoomlevel of elevation
elevation_map = []

start_y = 0
step_size = 10

#############  Check if there is existing elevation matrix to continue #############
elevation_matrix_json = project_dir + 'results/elevation.json'
if os.path.exists(elevation_matrix_json):
    with open(elevation_matrix_json, 'r') as json_file:
        elevation_map = json.load(json_file)
        if elevation_map:
            last_point = elevation_map[-1]
            start_y = last_point['y'] + step_size
            print("FOUND existing elevation matrix")
            print("\tLast point from checkpoint: ", last_point)
            print(f"\tTotal number of elevation: {len(elevation_map)} measurements")


#############  Get the elevation of the target region #############
print(f"Start from {start_y} to {y_checkpoint}")
for y in tqdm(range(start_y, min(map_size[1], y_checkpoint), step_size)):
    for x in tqdm(range(0, map_size[0], step_size)):
        # seems the scale factor is not correct...
        enu_x = round((x - center_offset[0]) * mpp, 3)
        enu_y = round((-y + center_offset[1]) * mpp, 3)

        # e: x, n: y
        target_lat, target_lon, _ = pymap3d.enu2geodetic(enu_x, enu_y, 0, center_geodetic[1], center_geodetic[0], 0, ell=None, deg=True)
        target_lat = round(target_lat, 6)
        target_lon = round(target_lon, 6)

        url = f"http://dev.virtualearth.net/REST/v1/Elevation/List?points={target_lat},{target_lon}&key={map_key}"
        urllib.request.urlretrieve(url, output)
        json_data = open(output, "r")
        json_res = json.load(json_data)
        elevation = json_res['resourceSets'][0]['resources'][0]['elevations'][0]
        elevation_map.append({'x': x, 'y': y, 'elevation': elevation})

# save the elevation list to json file if it is not empty
if y_checkpoint > start_y:
    with open(elevation_matrix_json, 'w') as json_file:
        print(f"\tSave elevation matrix to {elevation_matrix_json}")
        json.dump(elevation_map, json_file)

