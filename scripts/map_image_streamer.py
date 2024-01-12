# Purpose: Fetch a map image from Bing Maps
import urllib.request
import matplotlib.pyplot as plt
from PIL import Image
import json
import utils.utils as utils

config = json.loads(open('config.json').read())
project_dir = config["PROJECT_DIR"]

json_file_path = project_dir + 'sample_data/map_patch.json'
map_patch_json = utils.read_json(json_file_path)

map_key = ""
with open(project_dir + 'MapSessionConfig.txt', 'r') as f:
    map_key = f.readline()

url = utils.get_map_api(map_patch_json, map_key)
output = project_dir + "results/" + utils.get_out_filename(map_patch_json)

urllib.request.urlretrieve(url, output)
I = Image.open(output)
plt.imshow(I)
plt.show()
