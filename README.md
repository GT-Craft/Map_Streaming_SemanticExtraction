# GT-Craft: Map Streaming and Semantic Segmentation

GT-Craft is a framework for fast-prototyping geospatial-based digital twins in Unity 3D.
It uses a streamed satellite map image and elevation data to create a virtual scene in Unity.
From the map image, the object semantic is extracted by using DNN models, and this repo includes the map data streaming and segmentation phases of GT-Craft. For more detail, please check [our paper]().


-----------------------------------------------------------------------------------------------------------------------
## Setup
The current implementation is developed and test on Ubuntu 22.04.

#### Getting MS Bing Maps API Key
You need to get a key to use MS Bing Maps API from [here](https://www.bingmapsportal.com/?_gl=1*1vo9bx5*_gcl_au*NjgxOTQwNjk1LjE3MDUwOTI0OTQ).
Then, save your key in `MapSessionConfig.txt` under the repo directory.

```
Map_Streaming_SemanticExtraction
 |- MapSessionConfig.txt
 |- scripts
 |-  ...
```

#### Install Dependencies
```bash
$ cd Map_Streaming_SemanticExtraction

# Pytorch Install or visit https://pytorch.org/get-started/locally/ and install pytorch by yourself
Map_Streaming_SemanticExtraction $ ./setup_cpu.sh   # Pytorch without GPU
Map_Streaming_SemanticExtraction $ ./setup_cu118.sh # Pytorch with NVIDIA GPU and CUDA

# Other Libraries
Map_Streaming_SemanticExtraction $ pip install -r requirements.txt
```

#### Download Massachusetts Building and Road Dataset
The datasets to train the semantic segmentation models for buildings and roads can be downloaded from the below links.

[Massachusetts Roads Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset/download?datasetVersionNumber=1)

[Massachusetts Buildings Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset/download?datasetVersionNumber=2)


-----------------------------------------------------------------------------------------------------------------------
## Streaming, Training, and Validating

#### Update `config.json` for the project and dataset directories
```
Map_Streaming_SemanticExtraction
 |- MapSessionConfig.txt
 |- scripts
     |- config.json
```

#### Setting the target region for building the digital twins
You can set your target region by modifying the json file `sample_data/map_patch.json`.


#### 1. Getting the map image and elevation matrix
```
cd scripts

python map_image_streamer.py      # streaming map image

# streaming elevation data for 1500x1500 image size
# as it takes too long, use the prepared elevation data `sample_data/elevation.json/bin` for first exploration.
# python elevation_streamer.py 1500

python elevation_visualizer.py
```

#### 2. Training and validating segmentation models
You dataset directory should have the following structure. For testing, you can put your own map image under `test` directory.
```
YOUR_DATASET_PATH
  |- Buildings
      |- clabel_class_dict.csv
      |- metadata.csv
      |- png
      |- tiff
  |- Roads
      |- label_class_dict.csv
      |- metadata.csv
      |- tiff
  |- test
```

Then, you can run the following scripts for training the segmentation models.
```python
cd scripts
python train_road.py
python train_building.py
```

After all models are prepared, you can validate the models' performance on your target region.
```
python validate_road_building.py
```


#### 3. Segmentation on your region
You can get the mask images of the object classes (road and building) in your target area.
```
python road_building_segmentation.py
```

