# Sensor data in Mapillary Metropolis

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
    bounding box annotations, stored in `sample_annotation.json`. For objects with
    a well-defined orientation (e.g. cars), the X, Y and Z axes point right, forward
    and up, respectively. The Z axis always points up, even for objects that have
    some form of central symmetry (e.g. support poles). The bounding box corners
    have coordinates `[±W/2, ±L/2, ±H/2]` in the object's frame of reference.

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

## Sensors

The `sensor.json` table provides meta-data about the sensors used in Mapillary
Metropolis. In particular, different sensors are described by their `modality`
and `channel`. In the following we provide additional information on each modality,
and least all channels available for each.

### Modality: camera

These sensors produce RGB images, stored as JPG in the `sweeps` folder. Each sample
is guaranteed to have one equirectangular image, which should be regarded as the
main source of truth for annotations. Possible channels are:

* `CAM_EQUIRECTANGULAR`: the main equirectangular image.
* `CAM_LEFT`, `CAM_RIGHT`, `CAM_FRONT`, `CAM_BACK`: optional perspective images,
    pointing in the four cardinal directions w.r.t. to the ego-vehicle. These
    are obtained by warping the equirectangular image.

### Modality: depth

These sensors produce depth maps, stored as 16-bit PNGs in the `samples` folder.
These are obtained by re-projecting the multi-view stereo reconstruction. Possible
channels are:

* `DEPTH_LEFT`, `DEPTH_RIGHT`, `DEPTH_FRONT`, `DEPTH_BACK`: depth maps corresponding
    to the perspective images defined in the previous section.

### Modality: multi-view stereo

This sensor produces point clouds, stored as NPY files in the `samples` folder.
Each sensor reading contains a slice of a large MVS reconstruction, centered
around the corresponding ego-vehicle location. The only channel for this sensor
is named `MVS`.

### Modality: lidar

This sensor produces point clouds, stored as NPY files in the `samples` folder.
Each sensor reading contains a slice of a large lidar scan, centered around the
corresponding ego-vehicle location and re-aligned to match the corresponding
multi-view stereo slice. Possible channels are:

* `LIDAR_MX2`: ground-level lidar data, captured by the same vehicle that collected
    the equirectangular images.
* `LIDAR_AERIAL`: aerial lidar data.
