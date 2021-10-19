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
    └── sample_data/

where `train`, `test`, and `val` contain separate database tables for the dataset
splits, while the remaining folders contain the data files for the whole dataset.

__Note__: while this data format is strongly influenced by [NuScenes](https://www.nuscenes.org/nuscenes#data-format),
we __do not__ guarantee full compatibility with it.

__Note__: for a full description of the coordinate system conventions used in
Metropolis, please refer to [SENSORS.md](SENSORS.md).

## Database schema

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
    "name": str,            # Name of this category
    "description": str,     # Text description of this category
    "has_instances": bool,  # Whether this category can have instances (i.e. is a "thing" category)
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

### geo.json
Meta data about the geo-referenced coordinate systems and the aerial images.
```python
{
    "reference": {
        "lat": float,       # Reference latitude used to convert from cartesian coordinates to geo-referenced coordinates
        "lon": float,       # Reference longitude used to convert from cartesian coordinates to geo-referenced coordinates
        "alt": float,       # Reference altitude used to convert from cartesian coordinates to geo-referenced coordinates
    },
    "aerial": {
        "filename": str,    # Path to the aerial data file
    }
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

### points.json
Point-based annotations, i.e. human-annotated image-to-image correspondences. Each
annotation spans multiple images (within a scene), and has a corresponding 3D point
in world coordinates. Note that:
* The 3D points have been determined by cross-referencing the images with the CAD
    models, which only represent a rough representation of the environment. Because
    of this, these 3D points generally do not exactly reproject to their corresponding
    2D points in the images.
* As for `sample_annotation_2d.json`, the points are annotated on the 360-images.
```python
{
    "token": str,
    "scene_token": str,                 # Foreign key to the scene this annotation belongs to
    "point_3d": List[float],            # 3D position in world coordinates [x, y, z]
    "annotations": [                    # List of image points annotated to be in correspondence
        {
            "sample_token": str,        # Foreign key to a sample
            "point_2d": List[float],    # Position of the point in pixels as [x, y]
        },
        ...
    ]
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
    "size": List[float],        # Bounding box size [l, w, h], i.e. its extent along the Y, X and Z axes of the object frame
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
