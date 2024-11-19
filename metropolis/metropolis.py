# Copyright (c) Facebook, Inc. and its affiliates.

# Original copyright notice:
# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2018.
import json
import sys
import time
from os import path
from typing import Tuple, List, Optional, Dict, Any, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import (  # @manual=fbsource//third-party/pypi/matplotlib:matplotlib
    Axes,
)
from PIL import Image
from pyquaternion import Quaternion
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries
from skimage.transform import warp

from .utils import pathmgr
from .utils.color_map import get_colormap, plot_deph_normalized_colormap
from .utils.data_classes import LidarPointCloud, Box, Box2d, EquiBox2d
from .utils.geo import TopocentricConverter
from .utils.geometry_utils import (
    view_points,
    view_points_eq,
    transform_matrix,
    box_in_image,
    inverse_map_eq,
)

# GDAL import is optional
try:
    from osgeo import gdal
except ModuleNotFoundError:
    gdal = None

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("Metropolis dev-kit only supports Python version 3.")


EYE4 = np.eye(4)


class Metropolis:
    def __init__(
        self,
        split: str,
        dataroot: str,
        verbose: bool = True,
        map_resolution: float = 0.1,
    ):
        self.split = split
        self.dataroot = dataroot
        self.verbose = verbose
        self.table_names = [
            "category",
            "attribute",
            "instance",
            "sensor",
            "calibrated_sensor",
            "ego_pose",
            "panoptic",
            "scene",
            "sample",
            "sample_data",
            "sample_annotation",
            "sample_annotation_2d",
        ]

        assert pathmgr.exists(
            self.table_root
        ), f"Database version not found: {self.table_root}"

        start_time = time.time()
        if verbose:
            print(f"======\nLoading Metropolis tables for split {self.split}...")

        # Explicitly assign tables to help the IDE determine valid class members.
        self.category = self.__load_table__("category")
        self.attribute = self.__load_table__("attribute")
        self.instance = self.__load_table__("instance")
        self.sensor = self.__load_table__("sensor")
        self.calibrated_sensor = self.__load_table__("calibrated_sensor")
        self.ego_pose = self.__load_table__("ego_pose")
        self.panoptic = self.__load_table__("panoptic")
        self.scene = self.__load_table__("scene")
        self.sample = self.__load_table__("sample")
        self.sample_data = self.__load_table__("sample_data")
        self.sample_annotation = self.__load_table__("sample_annotation")
        self.sample_annotation_2d = self.__load_table__("sample_annotation_2d")
        self.geo = self.__load_table__("geo")

        # Initialize the colormap which maps from class names to RGB values.
        self.colormap = get_colormap()

        if verbose:
            for table in self.table_names:
                print(f"{len(getattr(self, table))} {table},")
            print(f"Done loading in {time.time() - start_time:.3f} seconds.\n======")

        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

    @property
    def table_root(self) -> str:
        """Returns the folder where the tables are stored for the relevant version."""
        return path.join(self.dataroot, self.split)

    def __load_table__(self, table_name: str) -> List[Dict[str, Any]]:
        """Loads a table."""
        with pathmgr.open(path.join(self.table_root, f"{table_name}.json")) as f:
            table = json.load(f)
        return table

    def __make_reverse_index__(self, verbose: bool) -> None:  # noqa C901
        """De-normalizes database to create reverse indices for common cases.

        Args:
            verbose: Whether to print outputs.
        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = {}
        for table in self.table_names:
            self._token2ind[table] = {}

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member["token"]] = ind

        # Decorate (adds short-cut) sample_annotation table with for category name.
        for record in self.sample_annotation:
            inst = self.get("instance", record["instance_token"])
            record["category_name"] = self.get("category", inst["category_token"])[
                "name"
            ]

        # Do the same for sample_annotation_2d
        for record in self.sample_annotation_2d:
            inst = self.get("instance", record["instance_token"])
            record["category_name"] = self.get("category", inst["category_token"])[
                "name"
            ]

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get("calibrated_sensor", record["calibrated_sensor_token"])
            sensor_record = self.get("sensor", cs_record["sensor_token"])
            record["sensor_modality"] = sensor_record["modality"]
            record["channel"] = sensor_record["channel"]

        # Reverse-index samples with sample_data, annotations and panoptic
        for record in self.sample:
            record["data"] = {}
            record["anns"] = []
            record["anns_2d"] = []

        for record in self.sample_data:
            sample_record = self.get("sample", record["sample_token"])
            sample_record["data"][record["channel"]] = record["token"]

        for ann_record in self.sample_annotation:
            sample_record = self.get("sample", ann_record["sample_token"])
            sample_record["anns"].append(ann_record["token"])

        for ann_record_2d in self.sample_annotation_2d:
            sample_record = self.get("sample", ann_record_2d["sample_token"])
            sample_record["anns_2d"].append(ann_record_2d["token"])

        for pano_record in self.panoptic:
            sample_record = self.get("sample", pano_record["sample_token"])
            sample_record["panoptic_token"] = pano_record["token"]

        if verbose:
            print(
                f"Done reverse indexing in {time.time() - start_time:.1f} seconds.\n======"
            )

    def get(self, table_name: str, token: str) -> Dict[str, Any]:
        """Returns a record from table in constant runtime.

        Args:
            table_name: Table name.
            token: Token of the record.

        Returns:
            Table record. See README.md for record details for each table.
        """
        assert table_name in self.table_names, f"Table {table_name} not found"

        return getattr(self, table_name)[self.getind(table_name, token)]

    def getind(self, table_name: str, token: str) -> int:
        """This returns the index of the record in a table in constant runtime.

        Args:
            table_name: Table name.
            token: Token of the record.

        Returns:
            The index of the record in table, table is an array.
        """
        return self._token2ind[table_name][token]

    def field2token(self, table_name: str, field: str, query: Any) -> List[str]:
        """This function queries all records for a certain field value, and returns
        the tokens for the matching records.

        Warning: this runs in linear time.

        Args:
            table_name: Table name.
            field: Field name. See README.md for details.
            query: Query to match against. Needs to type match the content of the
                query field.

        Returns:
            List of tokens for the matching records.
        """
        matches = []
        for member in getattr(self, table_name):
            if member[field] == query:
                matches.append(member["token"])
        return matches

    def get_box(self, sample_annotation_token: str) -> Box:
        """Instantiates a Box class from a sample annotation record.

        Args:
            sample_annotation_token: Unique sample_annotation identifier.
        """
        record = self.get("sample_annotation", sample_annotation_token)
        return Box(
            record["translation"],
            record["size"],
            Quaternion(record["rotation"]),
            name=record["category_name"],
            token=record["token"],
        )

    def get_box_2d(self, sample_annotation_2d_token: str) -> Box2d:
        """Instantiates a Box2d class from a 2D sample annotation record.

        Args:
            sample_annotation_token: Unique sample_annotation identifier.

        Returns:
            The box object.
        """
        record = self.get("sample_annotation_2d", sample_annotation_2d_token)
        return Box2d(
            record["bounding_box"], name=record["category_name"], token=record["token"]
        )

    def get_boxes(
        self, sample_data_token: str, get_all_visible: bool = False
    ) -> List[Box]:
        """Instantiates Boxes for all annotation for a particular sample_data record

        Args:
            sample_data_token: Unique sample_data identifier.
            get_all_visible: If true, retrieve annotations for all objects that are
                potentially visible from this sample_data (i.e. those that have a
                2D annotation in the corresponding 360 image). Otherwise, only
                return objects that have been annotated in 3D directly on this sample.

        Return:
            A list of boxes.
        """
        sd_record = self.get("sample_data", sample_data_token)
        curr_sample_record = self.get("sample", sd_record["sample_token"])

        if get_all_visible:
            instance_tokens = {
                self.get("sample_annotation_2d", sa_2d_token)["instance_token"]
                for sa_2d_token in curr_sample_record["anns_2d"]
            }
            return list(
                map(
                    self.get_box,
                    (
                        sa["token"]
                        for sa in self.sample_annotation
                        if sa["instance_token"] in instance_tokens
                    ),
                )
            )
        else:
            return list(map(self.get_box, curr_sample_record["anns"]))

    def get_boxes_2d(self, sample_data_token: str) -> List[Box2d]:
        """Instantiates 2D Boxes for all annotation for a particular sample_data record

        Args:
            sample_data_token: Unique sample_data identifier.

        Return:
            A list of 2D Boxes.
        """
        sd_record = self.get("sample_data", sample_data_token)
        curr_sample_record = self.get("sample", sd_record["sample_token"])
        return list(map(self.get_box_2d, curr_sample_record["anns_2d"]))

    def get_color(self, category_name: str) -> Tuple[int, int, int]:
        """Provides the default colors based on the category names.

        Args:
            category_name: Name of the category.

        Returns:
            Color for the category, or (0, 0, 0) if the category is not found.
        """
        return self.colormap.get(category_name, (0, 0, 0))

    def get_sample_data_path(self, sample_data_token: str) -> str:
        """Returns the path to a sample_data."""
        sd_record = self.get("sample_data", sample_data_token)
        return path.join(self.dataroot, sd_record["filename"])

    def get_sample_data(
        self,
        sample_data_token: str,
        selected_anntokens: Optional[List[str]] = None,
        selected_2d_anntokens: Optional[List[str]] = None,
        use_flat_vehicle_coordinates: bool = False,
        get_all_visible_boxes: bool = False,
    # pyre-fixme[11]: Annotation `array` is not defined as a type.
    ) -> Tuple[
        str,
        List[Box],
        Optional[Union[List[Box2d], List[EquiBox2d]]],
        Optional[np.array],
    ]:
        """Returns the data path as well as all annotations related to that sample_data.

        Note that the boxes are transformed into the current sensor's coordinate frame.

        Args:
            sample_data_token: Sample_data token.
            selected_anntokens: If provided only return the selected 3D annotation.
            selected_2d_anntokens: If provided only return the selected 2D annotation.
            use_flat_vehicle_coordinates: Instead of the current sensor's coordinate
                frame, use ego frame which is aligned to z-plane in the world.
            get_all_visible_boxes: If true, retrieve 3D boxes for all objects that are
                potentially visible from this sample_data (i.e. those that have a
                2D annotation in the corresponding 360 image). Otherwise, only
                return boxes that have been annotated in 3D directly on this sample.

        Returns:
            data_path: Path to the data file.
            boxes: 3D bounding boxes.
            boxes_2d: 2D bounding boxes (only returned for cameras).
            camera_intrinsic: Camera intrinsics.
        """

        # Retrieve sensor & pose records
        sd_record = self.get("sample_data", sample_data_token)
        cs_record = self.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        sensor_record = self.get("sensor", cs_record["sensor_token"])
        pose_record = self.get("ego_pose", sd_record["ego_pose_token"])

        is_camera_like = (
            sensor_record["modality"] == "depth"
            or sensor_record["modality"] == "camera"
        )

        data_path = self.get_sample_data_path(sample_data_token)

        if (
            sensor_record["modality"] == "camera"
            or sensor_record["modality"] == "depth"
        ):
            cam_intrinsic = np.array(cs_record["camera_intrinsic"])
            imsize = (sd_record["width"], sd_record["height"])
        else:
            cam_intrinsic = None
            imsize = None

        #### 3D annotations ####

        # Retrieve all sample annotations and map to sensor coordinate system.
        if selected_anntokens is not None:
            boxes = list(map(self.get_box, selected_anntokens))
        else:
            boxes = self.get_boxes(sample_data_token, get_all_visible_boxes)

        # Make list of Box objects including coord system transforms.
        box_list = []
        for box in boxes:
            if use_flat_vehicle_coordinates:
                # Move box to ego vehicle coord system parallel to world z plane.
                yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]
                box.translate(-np.array(pose_record["translation"]))
                box.rotate(
                    Quaternion(
                        scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]
                    ).inverse
                )
            else:
                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record["translation"]))
                box.rotate(Quaternion(pose_record["rotation"]).inverse)

                #  Move box to sensor coord system.
                box.translate(-np.array(cs_record["translation"]))
                box.rotate(Quaternion(cs_record["rotation"]).inverse)

            # For perspective camera-like sensors, check if the box is visible
            if is_camera_like and sensor_record["channel"] != "CAM_EQUIRECTANGULAR":
                # pyre-fixme[6]: For 2nd argument expected `ndarray[Any, Any]` but
                #  got `Optional[ndarray[Any, dtype[Any]]]`.
                # pyre-fixme[6]: For 3rd argument expected `Tuple[int, int]` but got
                #  `Optional[Tuple[Any, Any]]`.
                if not box_in_image(box, cam_intrinsic, imsize):
                    continue

            box_list.append(box)

        #### 2D Annotations, only for cameras ####
        box_2d_list = None

        if is_camera_like:
            if selected_2d_anntokens is not None:
                boxes_2d = list(map(self.get_box_2d, selected_2d_anntokens))
            else:
                boxes_2d = self.get_boxes_2d(sample_data_token)

            if sensor_record["channel"] == "CAM_EQUIRECTANGULAR":
                # For equirectangular images just return all boxes
                box_2d_list = boxes_2d
            else:
                # For perspective images, project the 2D boxes from the corresponding
                # equirectangular image
                sample_record = self.get("sample", sd_record["sample_token"])

                box_2d_list = []
                if "CAM_EQUIRECTANGULAR" in sample_record["data"]:
                    sd_eq_record = self.get(
                        "sample_data", sample_record["data"]["CAM_EQUIRECTANGULAR"]
                    )
                    cs_eq_record = self.get(
                        "calibrated_sensor", sd_eq_record["calibrated_sensor_token"]
                    )

                    for box_2d in boxes_2d:
                        box_2d_eq = EquiBox2d.from_box_2d(
                            box_2d,
                            Quaternion(cs_eq_record["rotation"]),
                            Quaternion(cs_record["rotation"]),
                            # pyre-fixme[6]: For 4th argument expected `ndarray[Any,
                            #  Any]` but got `Optional[ndarray[Any, dtype[Any]]]`.
                            cam_intrinsic,
                            (sd_eq_record["width"], sd_eq_record["height"]),
                            (sd_record["width"], sd_record["height"]),
                        )

                        if box_2d_eq is not None:
                            box_2d_list.append(box_2d_eq)

        return data_path, box_list, box_2d_list, cam_intrinsic

    def render_pointcloud_in_image(
        self,
        sample_token: str,
        # pyre-fixme[9]: dot_size has type `int`; used as `float`.
        dot_size: int = 0.5,
        downsample: int = 20,
        pointsensor_channel: str = "MVS",
        camera_channel: str = "CAM_FRONT",
        out_path: Optional[str] = None,
        ax: Optional[Axes] = None,
        nsweeps: int = 1,
    ):
        """Scatter-plots a point-cloud on top of equirectangular image.

        Args:
            sample_token: Sample token.
            dot_size: Scatter plot dot size.
            downsample: Downsampling factor.
            pointsensor_channel: Pointcloud channel name, e.g. 'MVS'.
            camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
            out_path: Optional path to save the rendered figure to disk.
            ax: Optional existing matplotlib axes object to draw on.
            nsweeps: Number of sweeps for lidar and radar.
        """
        sample_record = self.get("sample", sample_token)

        # Here we just grab the front camera and the point sensor.
        pointsensor_token = sample_record["data"][pointsensor_channel]
        camera_token = sample_record["data"][camera_channel]

        points, coloring, im = self.map_pointcloud_to_image(
            pointsensor_token,
            camera_token,
            nsweeps=nsweeps,
        )

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(18, 16), dpi=160)

        ax.imshow(im)
        ax.scatter(
            points[0, 0::downsample],
            points[1, 0::downsample],
            c=coloring[0::downsample],
            s=dot_size,
        )
        ax.axis("off")

        if out_path is not None:
            plt.xlim(0, im.size[0])
            plt.ylim(im.size[1], 0)

            with pathmgr.open(out_path, "wb") as fid:
                plt.savefig(fid, bbox_inches="tight", pad_inches=0)

    def map_pointcloud_to_image(
        self,
        pointsensor_token: str,
        camera_token: str,
        nsweeps: int = 1,
        min_dist: float = 1.0,
    ) -> Tuple:
        """Given a point sensor (e.g. lidar / mvs) token and camera sample_data token,
        load point-cloud and map it to an image.

        Args:
            pointsensor_token: Lidar/mvs sample_data token.
            camera_token: Camera sample_data token.
            nsweeps: Number of sweeps for lidar and mvs.
            min_dist: Distance from the camera below which points are discarded.

        Returns:
            pointcloud
            coloring
            image
        """
        # Find all relevant records
        sd_cam_record = self.get("sample_data", camera_token)
        sd_pts_record = self.get("sample_data", pointsensor_token)
        cs_cam_record = self.get(
            "calibrated_sensor", sd_cam_record["calibrated_sensor_token"]
        )
        cs_pts_record = self.get(
            "calibrated_sensor", sd_pts_record["calibrated_sensor_token"]
        )
        sample_pts_record = self.get("sample", sd_pts_record["sample_token"])
        ego_pose_cam = self.get("ego_pose", sd_cam_record["ego_pose_token"])
        ego_pose_pts = self.get("ego_pose", sd_pts_record["ego_pose_token"])

        # Load the point cloud
        pc, times = LidarPointCloud.from_file_multisweep(
            self,
            sample_pts_record,
            sd_pts_record["channel"],
            sd_pts_record["channel"],
            nsweeps,
        )

        # Load the image
        with pathmgr.open(
            path.join(self.dataroot, sd_cam_record["filename"]), "rb"
        ) as fid:
            im = Image.open(fid)
            im.load()

        # Transform the point cloud to the camera frame
        # 1. From point sensor to world
        pc.rotate(Quaternion(cs_pts_record["rotation"]).rotation_matrix)
        pc.translate(np.array(cs_pts_record["translation"]))
        pc.rotate(Quaternion(ego_pose_pts["rotation"]).rotation_matrix)
        pc.translate(np.array(ego_pose_pts["translation"]))

        # 2. From world to camera
        pc.translate(-np.array(ego_pose_cam["translation"]))
        pc.rotate(Quaternion(ego_pose_cam["rotation"]).rotation_matrix.T)
        pc.translate(-np.array(cs_cam_record["translation"]))
        pc.rotate(Quaternion(cs_cam_record["rotation"]).rotation_matrix.T)

        # Project the points to the camera plane / sphere
        if sd_cam_record["channel"] == "CAM_EQUIRECTANGULAR":
            # Equirectangular mode
            depths = np.sqrt((pc.points[:3, :] ** 2).sum(axis=0))

            # Do projection
            points = view_points_eq(pc.points[:3, :], im.size[0], im.size[1])
        else:
            # Projective mode
            depths = pc.points[2, :]

            # Do projection
            points = view_points(
                pc.points[:3, :],
                np.array(cs_cam_record["camera_intrinsic"]),
                normalize=True,
            )

            # Filter out points that are behind the camera, outside the frame or too close
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > min_dist)
            mask = np.logical_and(mask, points[0, :] > 1)
            mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
            mask = np.logical_and(mask, points[1, :] > 1)
            mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)

            points = points[:, mask]
            depths = depths[mask]

        return points, depths, im

    def render_sample_data(  # noqa C901
        self,
        sample_data_token: str,
        axes_limit: float = 40,
        ax: Optional[Axes] = None,
        nsweeps: int = 1,
        out_path: Optional[str] = None,
        use_flat_vehicle_coordinates: bool = True,
        show_3d_boxes: bool = False,
        show_all_visible_3d_boxes: bool = False,
        verbose: bool = False,
    ) -> None:
        """Render sample data onto axis.

        Args:
            sample_data_token: Sample_data token.
            axes_limit: Axes limit for lidar / mvs (measured in meters).
            ax: Axes onto which to render.
            nsweeps: Number of sweeps for lidar / mvs.
            out_path: Optional path to save the rendered figure to disk.
            use_flat_vehicle_coordinates: Instead of the current sensor's coordinate
                frame, use ego frame which is aligned to z-plane in the world.
            show_3d_boxes: When rendering images, the default is to show 2D boxes.
                If this is set to True, show 3D boxes instead.
            show_all_visible_3d_boxes: If true, when rendering 3D boxes we show all
                those that are potentially visible from this sample_data (i.e. those
                that have a 2D annotation in the corresponding 360 image). Otherwise,
                we only show those that have been annotated directly on this sample.
            verbose: Whether to display the image after it is rendered.
        """
        # Get sensor modality.
        sd_record = self.get("sample_data", sample_data_token)

        sensor_modality = sd_record["sensor_modality"]

        if sensor_modality == "lidar" or sensor_modality == "mvs":
            sample_rec = self.get("sample", sd_record["sample_token"])
            chan = sd_record["channel"]
            ref_chan = "MVS"
            ref_sd_token = sample_rec["data"][ref_chan]
            ref_sd_record = self.get("sample_data", ref_sd_token)

            # Get aggregated lidar point cloud in lidar frame.
            pc, times = LidarPointCloud.from_file_multisweep(
                self, sample_rec, chan, ref_chan, nsweeps=nsweeps
            )

            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
            # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
            if use_flat_vehicle_coordinates:
                # Retrieve transformation matrices for reference point cloud.
                cs_record = self.get(
                    "calibrated_sensor", ref_sd_record["calibrated_sensor_token"]
                )
                pose_record = self.get("ego_pose", ref_sd_record["ego_pose_token"])
                ref_to_ego = transform_matrix(
                    translation=cs_record["translation"],
                    rotation=Quaternion(cs_record["rotation"]),
                )

                # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
                ego_yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]
                rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(
                        scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]
                    ).rotation_matrix,
                    Quaternion(pose_record["rotation"]).inverse.rotation_matrix,
                )
                vehicle_flat_from_vehicle = np.eye(4)
                vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
            else:
                viewpoint = np.eye(4)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            # Show point cloud.
            points = view_points(pc.points[:3, :], viewpoint, normalize=False)
            dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            point_scale = 0.2

            ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

            # Show ego vehicle.
            ax.plot(0, 0, "x", color="red")

            # Get boxes in lidar frame.
            _, boxes, _, _ = self.get_sample_data(
                ref_sd_token,
                use_flat_vehicle_coordinates=use_flat_vehicle_coordinates,
                get_all_visible_boxes=show_all_visible_3d_boxes,
            )

            # Show boxes.
            for box in boxes:
                c = np.array(self.get_color(box.name)) / 255.0
                box.render(ax, view=np.eye(4), colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)
        elif sensor_modality == "camera" or sensor_modality == "depth":
            # Load boxes and image.
            data_path, boxes, boxes_2d, camera_intrinsic = self.get_sample_data(
                sample_data_token,
                get_all_visible_boxes=show_all_visible_3d_boxes,
            )
            with pathmgr.open(data_path, "rb") as fid:
                data = Image.open(fid)
                data.load()

            # depthmaps are stored in 16 bit pngs, so it needs to be converted in meters
            if sensor_modality == "depth":
                data = np.array(data, dtype=np.float32) / 256.0
                max_depth_color = 120.0
                depth_map_color = plot_deph_normalized_colormap(data, max_depth_color)
                data = Image.fromarray(depth_map_color.astype(np.uint8))

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 16))

            # Show image.
            ax.imshow(data)

            # Show boxes.
            if show_3d_boxes:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    if sd_record["channel"] == "CAM_EQUIRECTANGULAR":
                        box.render_eq(ax, data.size, colors=(c, c, c))
                    else:
                        box.render(
                            ax, view=camera_intrinsic, normalize=True, colors=(c, c, c)
                        )
            else:
                for box in boxes_2d:
                    c = np.array(self.get_color(box.name)) / 255.0
                    if isinstance(box, Box2d):
                        box.render(ax, sd_record["width"], color=c, linewidth=1)
                    else:
                        box.render(ax, color=c)

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax.axis("off")
        ax.set_title(sd_record["channel"])
        ax.set_aspect("equal")

        if out_path is not None:
            with pathmgr.open(out_path, "wb") as fid:
                plt.savefig(fid, bbox_inches="tight", pad_inches=0, dpi=200)

        if verbose:
            plt.show()

    def render_aerial_view(
        self,
        sample_data_token: str,
        axes_limit: float = 40,
        ax: Optional[Axes] = None,
        nsweeps: int = 1,
        out_path: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Render a view of a point cloud onto an aerial image.

        NOTE: this function requires the GDAL library.

        Args:
            sample_data_token: Sample_data token of the point cloud to render.
            axes_limit: Axes limit for lidar / mvs (measured in meters).
            ax: Axes onto which to render.
            nsweeps: Number of sweeps for lidar / mvs.
            out_path: Optional path to save the rendered figure to disk.
            verbose: Whether to display the image after it is rendered.
        """
        assert gdal is not None, "GDAL is required to use this function"

        # Get sensor modality.
        sd_record = self.get("sample_data", sample_data_token)
        sensor_modality = sd_record["sensor_modality"]

        assert (
            sensor_modality == "lidar" or sensor_modality == "mvs"
        ), "This function is only available for pointclouds"

        # Get the point cloud
        sample_rec = self.get("sample", sd_record["sample_token"])
        pc, _ = LidarPointCloud.from_file_multisweep(
            self, sample_rec, sd_record["channel"], "MVS", nsweeps=nsweeps
        )

        # Create coordinates converter
        converter = TopocentricConverter(
            self.geo["reference"]["lat"],
            self.geo["reference"]["lon"],
            self.geo["reference"]["alt"],
        )

        # Transform to global coordinates
        ref_sd_record = self.get("sample_data", sample_rec["data"]["MVS"])
        cs_record = self.get(
            "calibrated_sensor", ref_sd_record["calibrated_sensor_token"]
        )
        pose_record = self.get("ego_pose", ref_sd_record["ego_pose_token"])

        sens_to_ego = transform_matrix(
            rotation=Quaternion(cs_record["rotation"]),
            translation=np.array(cs_record["translation"]),
        )
        ego_to_world = transform_matrix(
            rotation=Quaternion(pose_record["rotation"]),
            translation=np.array(pose_record["translation"]),
        )
        sens_to_world = np.dot(ego_to_world, sens_to_ego)
        pc.transform(sens_to_world)

        # Get center and view area corners in geo-referenced coordinates
        y_c_lla, x_c_lla, _ = converter.to_lla(
            pose_record["translation"][0],
            pose_record["translation"][1],
            pose_record["translation"][2],
        )
        y0_lla, x0_lla, _ = converter.to_lla(
            pose_record["translation"][0] - axes_limit,
            pose_record["translation"][1] - axes_limit,
            pose_record["translation"][2],
        )
        y1_lla, x1_lla, _ = converter.to_lla(
            pose_record["translation"][0] + axes_limit,
            pose_record["translation"][1] + axes_limit,
            pose_record["translation"][2],
        )

        # Take a crop of the aerial data
        crop = gdal.Warp(
            "",
            path.join(self.dataroot, self.geo["aerial"]["filename"]),
            dstSRS="WGS84",
            format="VRT",
            outputBounds=(x0_lla, y0_lla, x1_lla, y1_lla),
        )
        img = np.stack(
            [crop.GetRasterBand(i + 1).ReadAsArray() for i in range(3)], axis=-1
        )
        img = Image.fromarray(img)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))

        # Draw the aerial data
        ax.imshow(img)

        # Project the point cloud to image coordinates
        y_p_lla, x_p_lla, _ = converter.to_lla(
            pc.points[0, :], pc.points[1, :], pc.points[2, :]
        )

        gt = crop.GetGeoTransform()
        R = np.array([[gt[1], gt[2]], [gt[4], gt[5]]])
        t = np.array([[gt[0]], [gt[3]]])
        points = np.linalg.solve(R, np.vstack([x_p_lla, y_p_lla]) - t)

        # Draw the points
        dists = np.sqrt(
            np.sum(
                (
                    pc.points[:2, :]
                    - np.array(pose_record["translation"][:2]).reshape(2, 1)
                )
                ** 2,
                axis=0,
            )
        )
        colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
        ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

        # Show ego vehicle
        ev_pos = np.linalg.solve(R, np.array([[x_c_lla], [y_c_lla]]) - t)
        ax.plot(ev_pos[0], ev_pos[1], "x", color="red")

        # Limit visible range
        ax.set_xlim(0, img.size[0])
        ax.set_ylim(img.size[1], 0)

        # Produce final output and optionally save
        ax.axis("off")
        ax.set_title(sd_record["channel"])
        ax.set_aspect("equal")

        if out_path is not None:
            with pathmgr.open(out_path, "wb") as fid:
                plt.savefig(fid, bbox_inches="tight", pad_inches=0, dpi=200)

        if verbose:
            plt.show()

    def get_panoptic_mask(
        self, sample_data_token: str
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """Get the panoptic mask for a given image

        Since panoptic masks are originally computed on the equirectangular images,
        this function performs a warp operation in case the requested image is a
        perspective one (e.g. "CAM_FRONT"), which might introduce small artificats
        in the returned mask.

        Args:
            sample_data_token: Sample_data token of the image.

        Returns:
            pano_record: The panoptic record dictionary.
            pano: The panoptic mask, possibly re-projected to the requested image.
        """
        # Get the sample data
        sd_record = self.get("sample_data", sample_data_token)
        assert (
            sd_record["sensor_modality"] == "camera"
        ), "Panoptic masks can only be computed for images"

        # Get the corresponding sample and panoptic records
        sample_record = self.get("sample", sd_record["sample_token"])
        pano_record = self.get("panoptic", sample_record["panoptic_token"])

        # Load the panoptic image
        pano_record = self.get("panoptic", sample_record["panoptic_token"])
        with pathmgr.open(
            path.join(self.dataroot, pano_record["filename"]), "rb"
        ) as fid:
            pano = Image.open(fid)
            pano = np.array(pano, copy=True)

        # Warp the mask to the requested view if needed
        if sd_record["channel"] != "CAM_EQUIRECTANGULAR":
            # Get the corresponding equirectangulare image sample data
            sd_eq_record = self.get(
                "sample_data", sample_record["data"]["CAM_EQUIRECTANGULAR"]
            )

            # Get the sensor records
            cs_eq_record = self.get(
                "calibrated_sensor", sd_eq_record["calibrated_sensor_token"]
            )
            cs_record = self.get(
                "calibrated_sensor", sd_record["calibrated_sensor_token"]
            )

            # 3D transformation CURRENT -> EQ
            t_ego_to_eq = transform_matrix(
                np.array(cs_eq_record["translation"]),
                Quaternion(cs_eq_record["rotation"]),
                inverse=True,
            )
            t_cur_to_ego = transform_matrix(
                np.array(cs_record["translation"]), Quaternion(cs_record["rotation"])
            )
            t_cur_to_eq = np.dot(t_ego_to_eq, t_cur_to_ego)

            # Warp the panoptic mask using scikit image
            dtype = pano.dtype
            pano = warp(
                pano,
                inverse_map_eq,
                map_args={
                    "transform": t_cur_to_eq,
                    "intrinsics": np.array(cs_record["camera_intrinsic"]),
                    "eq_size": (sd_eq_record["width"], sd_eq_record["height"]),
                },
                output_shape=(sd_record["height"], sd_record["width"]),
                order=0,
                preserve_range=True,
            ).astype(dtype)

        return pano_record, pano

    def render_panoptic(
        self, sample_data_token: str, out_path: Optional[str] = None
    ) -> Image.Image:
        """Overlay an image with the corresponding panoptic segmentation

        Since panoptic masks are originally computed on the equirectangular images,
        this function performs a warp operation in case the requested image is a
        perspective one (e.g. "CAM_FRONT"), which might introduce small artificats
        in the returned mask.

        Args:
            sample_data_token: Sample_data token of the image.
            out_path: Path to save the rendered figure to disk.

        Returns:
            The rendered image as a PIL Image
        """
        # Load the image
        with pathmgr.open(
            self.get_sample_data_path(sample_data_token),
            "rb",
        ) as fid:
            img = Image.open(fid)
            img.load()

        # Load the panoptic image
        pano_record, pano = self.get_panoptic_mask(sample_data_token)

        # Make a palette for this image
        palette = []
        for cat_token in pano_record["category_tokens"]:
            if cat_token is not None:
                cat_name = self.get("category", cat_token)["name"]
                palette.append(self.get_color(cat_name))
            else:
                palette.append((0, 0, 0))

        # Remap panoptic mask to colors
        palette = np.array(palette, dtype=np.uint8)
        pano_color = palette[pano, :]
        pano_color = Image.fromarray(pano_color)

        # Highlight the edges between instances
        is_stuff = np.array(
            [it is None for it in pano_record["instance_tokens"]], dtype=bool
        )
        pano[is_stuff[pano]] = 0

        contours = (
            find_boundaries(pano, mode="outer", background=0).astype(np.uint8) * 255
        )
        contours = dilation(contours)

        contours = np.expand_dims(contours, -1).repeat(4, -1)
        contours_img = Image.fromarray(contours, mode="RGBA")

        # Render and save
        out = Image.blend(img, pano_color, 0.5).convert(mode="RGBA")
        out = Image.alpha_composite(out, contours_img)
        out = out.convert(mode="RGB")

        if out_path is not None:
            with pathmgr.open(out_path, "wb") as fid:
                out.save(fid)
        return out
