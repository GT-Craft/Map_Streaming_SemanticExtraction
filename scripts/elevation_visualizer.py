import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import json
from matplotlib import cm
from scipy.ndimage.interpolation import map_coordinates
from matplotlib.ticker import LinearLocator
import numpy as np
import struct

config = json.loads(open('config.json', 'r').read())
project_dir = config['PROJECT_DIR']
elevation_matrix_json = project_dir + 'sample_data/elevation.json'

with open(elevation_matrix_json, 'r') as json_file:
    elevation_map = json.load(json_file)

# json -> 150x150 -> 1500x1500 float elevation map
elevation_2d = np.zeros((150, 150), dtype=np.float32)
for elem in elevation_map:
    x = elem['x'] // 10
    y = elem['y'] // 10
    elevation_2d[y][x] = elem['elevation']

new_dims = []
for original_length, new_length in zip(elevation_2d.shape, (1500,1500)):
    new_dims.append(np.linspace(0, original_length-1, new_length))

coords = np.meshgrid(*new_dims, indexing='ij')
elevation_2d_full = map_coordinates(elevation_2d, coords)

elevation_bytes = elevation_2d_full.tobytes()
# print(len(elevation_bytes))
with open(project_dir + '/sample_data/elevation.bin', 'wb') as binary_file:
    binary_file.write(elevation_bytes)

plt.imshow(elevation_2d_full)
plt.show()

# print(elevation_2d_full[0][0])
# print(elevation_2d_full[1499][0])
# print(elevation_2d_full[0][1499])
# print(elevation_2d_full[1499][1499])


X = []
Y = []
Z = []
ZZ = []
prev_y = -1
for elem in elevation_map:
    x = elem['x']
    y = elem['y']

    if prev_y != y:
        prev_y = y
        if ZZ:
            Z.append(ZZ)
            ZZ = []


    X.append(x)
    Y.append(y)
    ZZ.append(elem['elevation'])

if ZZ:
    Z.append(ZZ)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = range(0, 1500, 10)
Y = range(0, 1500, 10)
X, Y = np.meshgrid(X, Y)

Z = np.array(Z)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)

# Customize the z axis.
ax.set_zlim(200, 340)
# ax.set_zticks(np.arange(280, 340, 5))

# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.0f}')
# Set X and Y labels
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
# ax.set_zlabel('Elevation')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.4, label='Elevation (m)')

plt.show()

