# The Mapillary Metropolis data

## General description

The Mapillary Metropolis dataset is organized as a collection of separate data files,
each containing a specific sensor recording (e.g. images, point clouds, lidar sweeps),
plus metadata (e.g. geo-locations, annotations) stored in a relational database. The
database itself is stored as a set of json files, each containing one of the tables.
Records in the database are uniquely identified by a string "token", which is also
used for cross-referencing between tables.

The dataset's folder structure is as follows:

    root/
    ├── train/
    │   ├── db_file1.json
    │   ├── db_file2.json
    │   ...
    ├── test/
    │   ├── db_file1.json
    │   ├── db_file2.json
    │   ...
    ├── val/
    │   ├── db_file1.json
    │   ├── db_file2.json
    │   ...
    ├── aerial/
    ├── panoptic/
    ├── samples/
    └── sweeps/

where `train`, `test`, and `val` contain separate database tables for the dataset
splits, while the remaining folders contain the data files for the whole dataset.

__Note__: while this data format is strongly influenced by [NuScenes](https://www.nuscenes.org/nuscenes#data-format),
we __do not__ guarantee full compatibility with it.

## Coordinate systems and conventions

We define four main categories of coordinate systems:

* __World coordinates__: this is a fixed frame, anchored to a specific position
    in the 3D world which is shared across the whole dataset. X, Y world coordinates
    can be directly translated to geo-referenced EPSG:6498 coordinates by adding a
    specific offset vector stored in `aerial.json`.
* __Vehicle coordinates__: this is a moving frame, anchored to the vehicle that
    captured the sensor data. The X, Y and Z axes point right, forward and up,
    respectively. Vehicle coordinates are stored in `ego_pose.json`.
* __Sensor coordinates__: these are sensor-specific moving frames, anchored to the
    vehicle that captured the sensor data, but generally different from the vehicle
    coordinates. Transformations between sensor coordinates and vehicle coordinates
    are stored in `calibrated_sensor.json`.
    * For cameras, the sensor coordinate system follows the "OpenCV" convention,
        i.e. X, Y and Z point right, bottom and forward, respectively.
    * For point clouds, there's no general convention. The user should always
        interpret the stored point cloud files in the context of their specific
        sensor frame defined in `calibrated_sensor.json`.
* __Object coordinates__: these are object specific frames, used to represent 3D
    bounding box annotations, stored in `sample_annotation.json`. Each bounding box
    is defined by its corners `[±W/2, ±L/2, ±H/2]` in object coordinates, where W, L
    and H are the box's dimensions.

### Coordinate transformations

Transformations between coordinate systems are given as quaternion-vector pairs ![f1],
and always represent the transformation from the local frame to a more global frame
i.e. from object to world, from sensor to vehicle, from vehicle to world.

Given a roto-translation ![f1] from frame `A` to frame `B`, we can transform points in
`A` coordinates ![f2] to points in `B` coordinates ![f3] as:

![f4]

where the rotation matrix ![f5] for a quaternion ![f6] is given by:

![f7]

![f8]
![f9]

where ![f10] is the bottom-right `R x C` sub-matrix of `M`.

[f1]: https://chart.apis.google.com/chart?cht=tx&chl=(q%2C%20t)
[f2]: https://chart.apis.google.com/chart?cht=tx&chl=p%5EA
[f3]: https://chart.apis.google.com/chart?cht=tx&chl=p%5EB
[f4]: https://chart.apis.google.com/chart?cht=tx&chl=p%5EB%20%3D%20R(q)%20p%5EA%20%2B%20t
[f5]: https://chart.apis.google.com/chart?cht=tx&chl=R(q)
[f6]: https://chart.apis.google.com/chart?cht=tx&chl=q%3D(w%2C%20x%2C%20y%2C%20z)
[f7]: https://chart.apis.google.com/chart?cht=tx&chl=R(q)%3D%5Ctext%7BBR%7D_%7B3%5Ctimes%203%7D%5BQ(q)%5Cbar%7BQ%7D(q)%5E%5Ctop%5D%2C
[f8]: https://chart.apis.google.com/chart?cht=tx&chl=Q(q)%3D%5Cleft%5B%5Cbegin%7Bmatrix%7Dw%26-x%26-y%26-z%5C%5Cx%26w%26-z%26y%5C%5Cy%26z%26w%26-x%5C%5Cz%26-y%26x%26w%5C%5C%5Cend%7Bmatrix%7D%5Cright%5D%2C%5Cquad
[f9]: https://chart.apis.google.com/chart?cht=tx&chl=%5Cbar%7BQ%7D(q)%3D%5Cleft%5B%5Cbegin%7Bmatrix%7Dw%26-x%26-y%26-z%5C%5Cx%26w%26z%26-y%5C%5Cy%26-z%26w%26x%5C%5Cz%26y%26-x%26w%5C%5C%5Cend%7Bmatrix%7D%5Cright%5D%2C
[f10]: https://chart.apis.google.com/chart?cht=tx&chl=%5Ctext%7BBR%7D_%7BR%5Ctimes%20C%7D(M)

## Database schema

### aerial.json
Information about the aerial data and the world coordinate.
```python
{
    "offset": List[float],  # Offset in X, Y from world coordinates to geo-referenced coordinates
    "srs": str,             # Name of the geo-referenced coordinate system
    "filename": str,        # Path to the aerial data file, relative to root
}
```

### attribute.json
Definitions of the attributes sample annotations can possess.
```python
{
    "token": str,
    "name": str,        # Name of this attribute
    "description": str, # Text description of this attribute
}
```

### calibrated_sensor.json
Meta-data associated with a specific sensor instance, relating it to the vehicle.
Rotation and translation transform from the sensor frame to the vehicle frame.
```python
{
    "token": str,
    "camera_intrinsic": List[List[float]],  # 3 x 3 matrix of camera intrinsic parameters, only valid for image sensors
    "rotation": List[float],                # Rotation quaternion [q_w, q_x, q_y, q_z]
    "translation": List[float],             # Translation vector [t_x, t_y, t_z]
    "sensor_token": str,                    # Foreign key to the sensor category this sensor instance belongs to
}
```

### category.json
Semantic categories.
```python
{
    "token": str,
    "name": str,        # Name of this category
    "description": str, # Text description of this category
}
```

### ego_pose.json
A vehicle position in time and space. Rotation and translation transform from
the vehicle frame to the world frame.
```python
{
    "token": str,
    "rotation": List[float],    # Rotation quaternion [q_w, q_x, q_y, q_z]
    "translation": List[float], # Translation vector [t_x, t_y, t_z]
    "timestamp": int,           # Timestamp in Unix time
}
```

### instance.json
Object instances, annotated across samples. These generally comprise multiple
annotations, stored in `sample_annotation.json` and `sample_annotation_2d.json`.
Note that:
* Instances are only annotated at the scene level, meaning that if the same
    physical object is visible from different scenes, it will give rise to multiple
    entries in `instance.json`.
* The `first_` and `last_annotation_token`s refer to `sample_annotation_2d.json`.
```python
{
    "token": str,
    "category_token": str,                  # Foreign key to the category this object belongs to
    "first_annotation_token": str,          # Foreign key to the first annotation for this object
    "last_annotation_token": str,           # Foreign key to the last annotation for this object
    "nbr_annotations": int,                 # Number of annotations belonging to this object
    "geo_location": Optional[List[float]],  # Geo-location as [lon, lat]
    "geo_location_aerial": bool,            # True if the geo-location was annotated based on the aerial views, False if it's reconstructed from street-level views
}
```

### panoptic.json
Meta-data for the machine-generated panoptic masks. Note that:
* Panoptic masks are only given for the panoramic 360-image in each sample, but can
    be projected to the "virtual" perspective images using the `get_panoptic_mask()`
    function in the SDK.
* `instance_tokens` and `category_tokens` are maps from segment ids (see the
    panoptic data format specification below) to tokens.
```python
{
    "token": str,
    "sample_token": str,                    # Foreign key to the sample this mask belongs to
    "instance_tokens": List[Optional[str]], # Foreign keys to instances, or null if a segment belongs to a "stuff" category
    "category_tokens": List[Optional[str]], # Foreign keys to categories, or null if a segment has undefined category
    "filename": str,                        # Path to the file the mask is stored in, relative to root
}
```

### sample.json
Samples, i.e. collections of sensor recordings captured at a specific location in
time and space. These are grouped into sequential "scenes" (see `scene.json`).
```python
{
    "token": str,
    "timestamp": int,       # Timestamp in Unix time
    "scene_token": str,     # Foreign key to the scene this sample belongs to
    "previous_sample": str, # Token of the previous sample in this scene
    "next_sample": str,     # Token of the next sample in this scene
}
```

### sample_annotation.json
3D object annotations, defined as 3D bounding boxes in the world. Rotation and
translation transform from the object frame to the world frame. Note that the
`sample_token` field points to the sample where this 3D object was annotated, but
the same object could also be visible from other samples. In the SDK, one can
easily retrieve all 3D boxes that are potentially visible from a certain sample
with `Metropolis.get_boxes(..., get_all_visible=True)`.
```python
{
    "token": str,
    "rotation": List[float],    # Rotation quaternion [q_w, q_x, q_y, q_z]
    "translation": List[float], # Translation vector [t_x, t_y, t_z]
    "size": List[float],        # Bounding box size [w, l, h], i.e. its extent along the Y, X and Z axes, respectively
    "instance_token": str,      # Foreign key to the instance this annotation belongs to
    "sample_token": str,        # Foreign key to the sample where this object is annotated
}
```

### sample_annotation_2d.json
2D object annotations, defined as 2D bounding boxes on the panoramic 360-image of a
sample. Note that, since 360-images have a spherical topology, objects can "wrap
around" the images' sides. When this happens, the left side of the bounding box will
be close to the right image edge, while the right side of the object will be close
to the left image edge. This is reflected in the horizontal box coordinates `x0, x1`
which will be such that `x1 < x0`, instead of `x0 < x1`.
```python
{
    "token": str,
    "bounding_box": List[float],                    # Coordinates [x0, y0, x1, y1] of the bounding box, representing its left, top, right and bottom sides
    "extreme_points": Optional[List[List[float]]],  # Optional list of extreme points [..., [x_i, y_i], ...]
    "instance_token": str,                          # Foreign key to the instance this annotation belongs to
    "sample_token": str,                            # Foreign key to the sample where this object is annotated
    "attribute_tokens": List[str],                  # List of foreign keys to attributes for this annotation
    "next_sample_annotation": str,                  # Token of the next annotation in time for the same instance this annotation belongs to
    "previous_sample_annotation": str,              # Token of the previous annotation in time for the same instance this annotation belongs to
}
```

### sample_data.json
Meta-data associated with specific sensor recordings (e.g images, point clouds).
```python
{
    "token": str,
    "fileformat": str,              # Format of the raw data file (e.g. "jpg" for images, "bin" for point clouds)
    "filename": str,                # Path to the raw data file
    "width": Optional[int],         # Image width (or null for other sensors)
    "height": Optional[int],        # Image height (or null for other sensors)
    "timestamp": int,               # Timestamp in Unix time
    "sample_token": str,            # Foreign key to the sample this belongs to
    "ego_pose_token": str,          # Foreign key to the vehicle position at the time this was taken
    "calibrated_sensor_token": str, # Foreign key to the sensor instance this was taken from
    "next_sample_data": str,        # Token of the next sample data in time from the same sensor
    "previous_sample_data": str,    # Token of the previous sample data in time of the same sensor
}
```

### scene.json
Sequences of consecutive sensor recordings, captured by the same vehicle.
```python
{
    "token": str,
    "name": str,                # Scene name
    "description": str,         # Short text description
    "first_sample_token": str,  # Token of the first sample in the scene
    "last_sample_token": str,   # Token of the last sample in the scene
    "nbr_samples": int,         # Number of samples in the scene
}
```

### sensor.json
Meta data associated with each specific sensor used in the capturing process.
```python
{
    "token": str,
    "modality": str,    # Sensor modality, e.g. "camera" or "lidar"
    "channel": str,     # Sensor channel, e.g. "CAM_FRONT" or "LIDAR_PANO"
}
```

## Sensor data formats

### Images
All images are stored in `JPG` format in the `sweeps` folder. Currently, each sample
in the dataset includes a 360 degrees equirectangular image and four perspective
images pointing in the four cardinal directions w.r.t. the vehicle. The equirectangular
image is to be considered as the "source of truth", and all 2D annotations (detections,
panoptic segmentations) are defined on it.

### Point clouds
Point clouds are stored as numpy binary data files in the `samples` folder, and can
be decoded using the following code snippet:
```python
import numpy as np

with open(POINT_CLOUD_PATH, "rb") as fid:
    data = np.frombuffer(fid.read(), dtype=np.float32)
    data = data.reshape((-1, 5))[:, :3]
```
The resulting `data` will be an `N x 3` matrix, containing a point `[x, y, z]` in
each row.

### Aerial images
Aerial images are stored in the `aerial` folder, using the GDAL `VRT` format. Note
that GDAL is not listed as a requirement of the SDK, and must be installed
separately as explained in the README.
