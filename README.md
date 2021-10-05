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
pip install git+https://github.com/mapillary/metropolis.git
```
or from source:
```bash
git clone https://github.com/mapillary/metropolis.git
cd metropolis
pip install -r requirements.txt
python setup.py install
```

As an optional dependency, the GDAL library enables accessing geo-referenced aerial
images. The easiest way to do this is with Anaconda:
```bash
conda install -c conda-forge gdal
```

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
    pointsensor_channel="MVS_PANO",
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
