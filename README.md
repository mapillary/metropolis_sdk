# Mapillary Metropolis SDK

This SDK is intended as a collection of examples to get you started with understanding
the Mapillary Metropolis dataset, e.g. by visualizing annotations or performing
common queries on the data.

## Overview
- [Setting up the Metropolis SDK](#setting-up-the-metropolis-sdk)
- [Getting started with the Metropolis SDK](#getting-started-with-the-metropolis-sdk)
- [License](#license)
- [Changelog](#changelog)

## Setting up the Mapillary Metropolis SDK

The Mapillary Metropolis SDK can be installed directly using pip:
```bash
pip install git+https://github.com/mapillary/metropolis_sdk.git
```
or from source:
```bash
git clone https://github.com/mapillary/metropolis_sdk.git
cd metropolis
pip install -r requirements.txt
python setup.py install
```

As an optional dependency, the GDAL library enables accessing geo-referenced aerial
images. The easiest way to do this is with Anaconda:
```bash
conda install -c conda-forge gdal
```

### Preparing the point cloud data

We distribute our point cloud data as three PLY files, each containing the full extent
of one data modality (lidar, multi-view stereo etc.). The SDK, however, assumes that
the point clouds are stored as separate "crop" files, each containing a local slice
of the full point cloud centered around a specific spatial location. The advantages
of this format are two-fold:

1. Local point cloud data is much easier and faster to access.
2. The slices can be micro-aligned with each other, providing a more accurate local
    representation of the scene.

However, the slice-based representation is extremely redundant and storage-intensive,
so we give the users full control on which slices to generate. This is achieved with
the `metropolis_crop_point_clouds` script:

```bash
$ crop_point_clouds --help
usage: metropolis_crop_point_clouds
    [-h] [--metroplois_root_dir METROPLOIS_ROOT_DIR]
    [--sensor_string SENSOR_STRING]
    [--pointcloud_folder POINTCLOUD_FOLDER]
    [--sample_data_folder SAMPLE_DATA_FOLDER]
    [--list_of_sample_keys LIST_OF_SAMPLE_KEYS]
    [--set_sequence SET_SEQUENCE [SET_SEQUENCE ...]]
    [--max_nr_to_process MAX_NR_TO_PROCESS] [--save_ply]

optional arguments:
  -h, --help            show this help message and exit
  --metroplois_root_dir METROPLOIS_ROOT_DIR
                        Directory containing the train/val/test folder
  --sensor_string SENSOR_STRING
                        Sensor ID from the pointcloud modality used for the
                        crops. Can be one of the following: LIDAR_MX2,
                        LIDAR_AERIAL, MVS
  --pointcloud_folder POINTCLOUD_FOLDER
                        Folder containing the big MVS and LiDAR pointclouds
  --sample_data_folder SAMPLE_DATA_FOLDER
                        Crops will be put into this folder. By default, it
                        will be put into 'sample_data', relative to the
                        Metropolis folder
  --list_of_sample_keys LIST_OF_SAMPLE_KEYS
                        Filename of list of sample keys in json format to
                        process, if not specified, all samples will be
                        processed
  --set_sequence SET_SEQUENCE [SET_SEQUENCE ...]
                        List of sets to process, e.g.: train, train val
  --max_nr_to_process MAX_NR_TO_PROCESS
                        Maximum number of crops per set, default=-1
                        (unlimited)
  --save_ply            if not specified saves *.npz format
```

Assuming that the Metropolis dataset is in `~/metropolis` and the point clouds have
been unpacked in `~/metropolis/point_clouds`, we can generate all the slices for the
`MVS` modality with:
```bash
$ metropolis_crop_point_clouds --metroplois_root_dir ~/metropolis --sensor_string MVS --pointcloud_folder ~/metropolis/point_clouds  --set_sequence train val
```

**Note:** this process will take a few hours to complete, and require several terabytes
of storage space if all slices are to be generated.

## Getting started with the Metropolis SDK

The entry point to the SDK is the [Metropolis](metropolis/metropolis.py#L49) class:
```python
from metropolis import Metropolis
metropolis = Metropolis(
    "train", # Name of the split we want to load
    "/home/user/metropolis_data", # Path to the root directory of the dataset
)
```
Annotations and meta-data are stored as a relational database with multiple tables,
with specific entries identified by unique "token" strings:
```python
# Get a sample_data (i.e. image or point cloud meta-data) from the dataset:
sample_data_record = metropolis.sample_data[0]
assert sample_data_record == metropolis.get("sample_data", sample_data_record["token"])

# Find the corresponding sample (i.e. group of images and point clouds captured
# at the same point in time and space)
sample_record = metropolis.get("sample", sample_data_record["sample_token"])
```
Data items (e.g. images, point clouds) are stored as discrete files, each of which
has a corresponding entry in the `sample_data` table:
```python
# Get a sample_data given its token
sample_data_record = metropolis.get("sample_data", "{sample_data_token}")

# Get the path to its actual data file
sample_data_path = metropolis.get_sample_data_path("{sample_data_token}")
```

The SDK also offers a variety of rendering functions to easily visualize different
parts of the dataset, e.g. we can view a `sample_data` entry with all its available
annotations with:
```python
metropolis.render_sample_data(
    "{sample_data_token}",
    out_path="/path/to/output/image.png",
)
```
or render a point cloud onto an image with:
```python
metropolis.render_pointcloud_in_image(
    "{sample_token}",
    pointsensor_channel="MVS",
    camera_channel="CAM_EQUIRECTANGULAR",
    out_path="/path/to/output/image.png",
)
```

### Data format

The Mapillary Metropolis data format is mostly compatible with the NuScenes format
described [here](https://www.nuscenes.org/nuscenes#data-format). For a full
description of our changes and additions, as well as the coordinate systems used in
Metropolis and their transformations, please refer to [this document](FORMAT.md).

## License
The Mapillary Metropolis SDK is Apache 2.0 licensed, as found in the LICENSE file.

## Changelog

TBD - Initial Release
