# Copyright (c) Facebook, Inc. and its affiliates.

# Original copyright notice:
# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2018.

import copy
import os.path as osp
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict, TYPE_CHECKING, Any, Optional

import numpy as np
from matplotlib.axes import (  # @manual=fbsource//third-party/pypi/matplotlib:matplotlib
    Axes,
)
from pyquaternion import Quaternion

from . import pathmgr
from .geometry_utils import view_points, view_points_eq, transform_matrix, split_poly_eq

if TYPE_CHECKING:
    from ..metropolis import Metropolis

EYE3 = np.eye(3)
EYE4 = np.eye(4)


class PointCloud(ABC):
    """Abstract class for manipulating and viewing point clouds.

    Every point cloud (lidar and radar) consists of points where:
    - Dimensions 0, 1, 2 represent x, y, z coordinates.
        These are modified when the point cloud is rotated or translated.
    - All other dimensions are optional. Hence these have to be manually modified if
        the reference frame changes.

    Args:
        points: d-dimensional input point cloud matrix.
    """

    def __init__(self, points: np.ndarray):
        assert (
            points.shape[0] == self.nbr_dims()
        ), f"Error: Pointcloud points must have format: {self.nbr_dims()} x n"
        self.points = points

    @staticmethod
    @abstractmethod
    def nbr_dims() -> int:
        """Returns the number of dimensions.

        Returns: Number of dimensions.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, file_name: str) -> "PointCloud":
        """Loads point cloud from disk.

        Args:
            file_name: Path of the pointcloud file on disk.

        Returns:
            PointCloud instance.
        """
        pass

    @classmethod
    def from_file_multisweep(
        cls,
        metr: "Metropolis",
        sample_rec: Dict[str, Any],
        chan: str,
        ref_chan: str,
        nsweeps: int = 5,
        min_distance: float = 1.0,
    ) -> Tuple["PointCloud", np.ndarray]:
        """Return a point cloud that aggregates multiple sweeps.

        As every sweep is in a different coordinate frame, we need to map the
        coordinates to a single reference frame. As every sweep has a different
        timestamp, we need to account for that in the transformations and timestamps.

        Args:
            metr: A Metropolis instance.
            sample_rec: The current sample.
            chan: The lidar/radar channel from which we track back n sweeps to
                aggregate the point cloud.
            ref_chan: The reference channel of the current sample_rec that the point
                clouds are mapped to.
            nsweeps: Number of sweeps to aggregated.
            min_distance: Distance below which points are discarded.

        Returns:
            all_pc: The aggregated point clouds.
            all_times: The aggregated timestamps.
        """
        # Init.
        points = np.zeros(
            (cls.nbr_dims(), 0),
            dtype=np.float32 if cls == LidarPointCloud else np.float64,
        )
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp.
        ref_sd_token = sample_rec["data"][ref_chan]
        ref_sd_rec = metr.get("sample_data", ref_sd_token)
        ref_pose_rec = metr.get("ego_pose", ref_sd_rec["ego_pose_token"])
        ref_cs_rec = metr.get(
            "calibrated_sensor", ref_sd_rec["calibrated_sensor_token"]
        )
        ref_time = 1e-6 * ref_sd_rec["timestamp"]

        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(
            ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True
        )

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(
            ref_pose_rec["translation"],
            Quaternion(ref_pose_rec["rotation"]),
            inverse=True,
        )

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec["data"][chan]
        current_sd_rec = metr.get("sample_data", sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = cls.from_file(
                osp.join(metr.dataroot, current_sd_rec["filename"])
            )
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = metr.get("ego_pose", current_sd_rec["ego_pose_token"])
            global_from_car = transform_matrix(
                current_pose_rec["translation"],
                Quaternion(current_pose_rec["rotation"]),
                inverse=False,
            )

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = metr.get(
                "calibrated_sensor", current_sd_rec["calibrated_sensor_token"]
            )
            car_from_current = transform_matrix(
                current_cs_rec["translation"],
                Quaternion(current_cs_rec["rotation"]),
                inverse=False,
            )

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(
                np.dot,
                [ref_from_car, car_from_global, global_from_car, car_from_current],
            )
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = (
                ref_time - 1e-6 * current_sd_rec["timestamp"]
            )  # Positive difference.
            times = time_lag * np.ones((1, current_pc.nbr_points()))
            all_times = np.hstack((all_times, times))

            # Merge with key pc.
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

            # Abort if there are no previous sweeps.
            if current_sd_rec["prev"] == "":
                break
            else:
                current_sd_rec = metr.get("sample_data", current_sd_rec["prev"])

        return all_pc, all_times

    def nbr_points(self) -> int:
        """Returns the number of points.

        Returns:
            Number of points.
        """
        return self.points.shape[1]

    def subsample(self, ratio: float) -> None:
        """Sub-samples the pointcloud.

        Args:
            ratio: Fraction to keep.
        """
        selected_ind = np.random.choice(
            np.arange(0, self.nbr_points()), size=int(self.nbr_points() * ratio)
        )
        self.points = self.points[:, selected_ind]

    def remove_close(self, radius: float) -> None:
        """Removes point too close within a certain radius from origin.

        Args:
            radius: Radius below which points are removed.
        """

        x_filt = np.abs(self.points[0, :]) < radius
        y_filt = np.abs(self.points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        self.points = self.points[:, not_close]

    def translate(self, x: np.ndarray) -> None:
        """Applies a translation to the point cloud.

        Args:
            x: Translation in x, y, z.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix: np.ndarray) -> None:
        """Applies a rotation.

        Args:
            rot_matrix: Rotation matrix.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix: np.ndarray) -> None:
        """Applies a homogeneous transform.

        Args:
            transf_matrix: Homogenous transformation matrix.
        """
        self.points[:3, :] = transf_matrix.dot(
            np.vstack((self.points[:3, :], np.ones(self.nbr_points())))
        )[:3, :]

    def render_height(
        self,
        ax: Axes,
        view: np.ndarray = EYE4,
        x_lim: Tuple[float, float] = (-20, 20),
        y_lim: Tuple[float, float] = (-20, 20),
        marker_size: float = 1,
    ) -> None:
        """Very simple method that applies a transformation and then scatter plots
        the points colored by height (z-value).

        Args:
            ax: Axes on which to render the points.
            view: Defines an arbitrary projection (n <= 4).
            x_lim: (min, max). x range for plotting.
            y_lim: (min, max). y range for plotting.
            marker_size: Marker size.
        """
        self._render_helper(2, ax, view, x_lim, y_lim, marker_size)

    def render_intensity(
        self,
        ax: Axes,
        view: np.ndarray = EYE4,
        x_lim: Tuple[float, float] = (-20, 20),
        y_lim: Tuple[float, float] = (-20, 20),
        marker_size: float = 1,
    ) -> None:
        """Very simple method that applies a transformation and then scatter plots
        the points colored by intensity.

        Args:
            ax: Axes on which to render the points.
            view: Defines an arbitrary projection (n <= 4).
            x_lim: (min, max).
            y_lim: (min, max).
            marker_size: Marker size.
        """
        self._render_helper(3, ax, view, x_lim, y_lim, marker_size)

    def _render_helper(
        self,
        color_channel: int,
        ax: Axes,
        view: np.ndarray,
        x_lim: Tuple[float, float],
        y_lim: Tuple[float, float],
        marker_size: float,
    ) -> None:
        """Helper function for rendering.

        Args:
            color_channel: Point channel to use as color.
            ax: Axes on which to render the points.
            view: Defines an arbitrary projection (n <= 4).
            x_lim: (min, max).
            y_lim: (min, max).
            marker_size: Marker size.
        """
        points = view_points(self.points[:3, :], view, normalize=False)
        ax.scatter(
            points[0, :], points[1, :], c=self.points[color_channel, :], s=marker_size
        )
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])


class LidarPointCloud(PointCloud):
    @staticmethod
    def nbr_dims() -> int:
        """Returns the number of dimensions.

        Returns:
            Number of dimensions.
        """
        return 4

    @classmethod
    def from_file(cls, file_name: str) -> "LidarPointCloud":
        """Loads LIDAR data from binary numpy format. Data is stored as (x, y, z,
        intensity, ring index).

        Args:
            file_name: Path of the pointcloud file on disk.

        Returns:
            LidarPointCloud instance (x, y, z, intensity).
        """

        assert file_name.endswith(".bin"), f"Unsupported filetype {file_name}"

        with pathmgr.open(file_name, "rb") as fid:
            scan = np.frombuffer(fid.read(), dtype=np.float32)
        points = scan.reshape((-1, 5))[:, : cls.nbr_dims()]
        return cls(points.T)


class Box:
    """Simple data class representing a 3d box

    Args:
        center: Center of box given as x, y, z.
        size: Size of box in width, length, height.
        orientation: Box orientation.
        name: Box name, optional. Can be used e.g. for denote category name.
        token: Unique string identifier from DB.
    """

    def __init__(
        self,
        center: List[float],
        size: List[float],
        orientation: Quaternion,
        name: str = None,
        token: str = None,
    ):
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.name = name
        self.token = token

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)

        return center and wlh and orientation

    def __repr__(self):
        repr_str = (
            "xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], "
            "rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, "
            "name: {}, token: {}"
        )

        return repr_str.format(
            self.center[0],
            self.center[1],
            self.center[2],
            self.wlh[0],
            self.wlh[1],
            self.wlh[2],
            self.orientation.axis[0],
            self.orientation.axis[1],
            self.orientation.axis[2],
            self.orientation.degrees,
            self.orientation.radians,
            self.name,
            self.token,
        )

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Return a rotation matrix.

        Returns:
            The box's rotation matrix.
        """
        return self.orientation.rotation_matrix

    def translate(self, x: np.ndarray) -> None:
        """Applies a translation.

        Args:
            x: Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        """Rotates box.

        Args:
            quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """Returns the bounding box corners.

        Args:
            wlh_factor: Multiply w, l, h by a factor to scale the box.

        Returns:
            First four corners are the ones facing forward. The last four are the
            ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """Returns the four bottom corners.

        Returns:
            Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]

    def render(
        self,
        axis: Axes,
        view: np.ndarray = EYE3,
        normalize: bool = False,
        colors: Tuple[Any, Any, Any] = ("b", "r", "k"),
        linewidth: float = 2,
    ) -> None:
        """Renders the box in the provided Matplotlib axis.

        Args:
            axis: Axis onto which the box should be drawn.
            view: Define a projection in needed (e.g. for drawing projection in an image).
            normalize: Whether to normalize the remaining coordinate.
            colors: Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
                back and sides.
            linewidth: Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot(
                    [prev[0], corner[0]],
                    [prev[1], corner[1]],
                    color=color,
                    linewidth=linewidth,
                )
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot(
                [corners.T[i][0], corners.T[i + 4][0]],
                [corners.T[i][1], corners.T[i + 4][1]],
                color=colors[2],
                linewidth=linewidth,
            )

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot(
            [center_bottom[0], center_bottom_forward[0]],
            [center_bottom[1], center_bottom_forward[1]],
            color=colors[0],
            linewidth=linewidth,
        )

    def render_eq(
        self,
        axis: Axes,
        img_size: Tuple[int, int],
        colors: Tuple[Any, Any, Any] = ("b", "r", "k"),
        linewidth: float = 2,
        num_samples: int = 20,
    ) -> None:
        """Renders the box in the provided Matplotlib axis, using an equirectangular
        projection.

        Args:
            axis: Axis onto which the box should be drawn.
            img_size: Image size as (width, height).
            colors: Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
                back and sides.
            linewidth: Width in pixel of the box sides.
            num_samples: Number of points to sample on each edge of the bounding box
        """
        t = np.linspace(0, 1, num_samples).reshape(1, -1)
        corners = self.corners()

        def draw_line(p1, p2, color):
            line = p1.reshape(3, -1) + t * (p2 - p1).reshape(3, -1)
            line = view_points_eq(line, img_size[0], img_size[1])

            for line in split_poly_eq(line, img_size[0]):
                axis.plot(line[0, :], line[1, :], color=color, linewidth=linewidth)

        # Draw the sides
        for i in range(4):
            draw_line(corners[:, i], corners[:, i + 4], colors[2])

        # Draw front and back
        for i in range(4):
            draw_line(corners[:, i % 4], corners[:, (i + 1) % 4], colors[0])
        for i in range(4):
            draw_line(corners[:, i % 4 + 4], corners[:, (i + 1) % 4 + 4], colors[1])

    def copy(self) -> "Box":
        """Create a copy of self.

        Returns:
            A copy.
        """
        return copy.deepcopy(self)


class Box2d:
    """Simple data class representing a 2d box

    This representa an axis-aligned box on an equirectangular image, meaning that
    the same region on a side-view perspective image will be a deformed rectangle.

    Args:
        coords: Bounding box coordinates [left, top, right, bottom]. Note that for
            boxes that "wrap around" the equirectangular image, left > right, while
            for all others left < right.
        name: Box name, optional. Can be used e.g. for denote category name.
        token: Unique string identifier from DB.
    """

    def __init__(
        self,
        coords: List[float],
        name: str = None,
        token: str = None,
    ):
        assert not np.any(np.isnan(coords))
        assert len(coords) == 4

        self.coords = coords
        self.name = name
        self.token = token

    def __eq__(self, other) -> bool:
        return np.allclose(self.coords, other.coords)

    def __repr__(self) -> str:
        return "[{:.2f}, {:.2f}, {:.2f}, {:.2f}], name={}, token={}".format(
            self.coords[0],
            self.coords[1],
            self.coords[2],
            self.coords[3],
            self.name,
            self.token,
        )

    def render(
        self, axis: Axes, width: int, color: Any = "r", linewidth: int = 2
    ) -> None:
        """Renders the box in the provided Matplotlib axis.

        Args:
            axis: Axis onto which the box should be drawn.
            width: Width of the equirectangular image.
            colors: Valid Matplotlib color.
            linewidth: Width in pixel of the box sides.
        """
        if self.coords[0] < self.coords[2]:
            segments = [
                ([self.coords[0], self.coords[2]], [self.coords[1], self.coords[1]]),
                ([self.coords[2], self.coords[2]], [self.coords[1], self.coords[3]]),
                ([self.coords[2], self.coords[0]], [self.coords[3], self.coords[3]]),
                ([self.coords[0], self.coords[0]], [self.coords[3], self.coords[1]]),
            ]
        else:
            segments = [
                ([self.coords[0], width], [self.coords[1], self.coords[1]]),
                ([0, self.coords[2]], [self.coords[1], self.coords[1]]),
                ([self.coords[2], self.coords[2]], [self.coords[1], self.coords[3]]),
                ([self.coords[2], 0], [self.coords[3], self.coords[3]]),
                ([width, self.coords[0]], [self.coords[3], self.coords[3]]),
                ([self.coords[0], self.coords[0]], [self.coords[3], self.coords[1]]),
            ]

        for x, y in segments:
            axis.plot(x, y, color=color, linewidth=linewidth)


class EquiBox2d:
    """2D bounding box on an equirectangular image reprojected to a perspective image

    Args:
        points: 2D points defining the box contour.
        name: Box name, optional. Can be used e.g. for denote category name.
        token: Unique string identifier from DB.
    """

    def __init__(
        self,
        points: np.ndarray,
        name: Optional[str] = None,
        token: Optional[str] = None,
    ):
        assert points.shape[0] == 2

        self.points = points
        self.name = name
        self.token = token

    def render(self, axis: Axes, color: Any = "r", linewidth: int = 2) -> None:
        """Renders the box in the provided Matplotlib axis.

        Args:
            axis: Axis onto which the box should be drawn.
            width: Width of the equirectangular image.
            colors: Valid Matplotlib color.
            linewidth: Width in pixel of the box sides.
        """
        axis.plot(self.points[0], self.points[1], color=color, linewidth=linewidth)

    @classmethod
    def from_box_2d(
        cls,
        box: Box2d,
        q_eq: Quaternion,
        q_pr: Quaternion,
        intrinsic: np.ndarray,
        size_eq: Tuple[int, int],
        size_pr: Tuple[int, int],
        num_samples: int = 20,
    ) -> Optional["EquiBox2d"]:
        """Project an equirectangular bounding box to a projective image

        This assumes that both cameras have the same center, being related by a
        simple rotation transformation. The output box is constructed by projecting
        a fixed number of points on each edge of the input box from the
        equirectangular to the perspective image, forming a discrete approximation
        of its curved shape.

        Args:
            box: A bounding box defined in the equirectangular image.
            q_eq: The rotation of the equirectangular image with respect to an
                external frame of reference (e.g the ego vehicle).
            q_pr: The rotation of the projective image with respect to the same
                external frame of referene.
            intrinsic: Intrinsic matrix of the projective image.
            size_eq: The size of the equirectangular image, as width, height
            size_pr: The size of the projective image, as width, height
            num_samples: Number of points to sample on each edge of the bounding box

        Returns:
            The projected box, or None if the given box falls completely outside
            of the projective image.
        """

        def get_points(x0, y0, x1, y1):
            side1 = np.stack(
                [
                    np.linspace(x0, x1, num_samples, dtype=np.float32),
                    np.full((num_samples,), y0, dtype=np.float32),
                ],
                axis=0,
            )
            side2 = np.stack(
                [
                    np.full((num_samples,), x1, dtype=np.float32),
                    np.linspace(y0, y1, num_samples, dtype=np.float32),
                ],
                axis=0,
            )
            side3 = np.stack(
                [
                    np.linspace(x1, x0, num_samples, dtype=np.float32),
                    np.full((num_samples,), y1, dtype=np.float32),
                ],
                axis=0,
            )
            side4 = np.stack(
                [
                    np.full((num_samples,), x0, dtype=np.float32),
                    np.linspace(y1, y0, num_samples, dtype=np.float32),
                ],
                axis=0,
            )
            return np.concatenate([side1, side2, side3, side4], axis=1)

        def project(points):
            # From pixel coordinates to angles
            u = (points[0] / size_eq[0] - 0.5) * 2 * np.pi
            v = (points[1] / size_eq[1] - 0.5) * np.pi

            # From angles to 3D coordinates on the sphere
            p_eq = np.stack(
                [np.cos(v) * np.sin(u), np.sin(v), np.cos(v) * np.cos(u)], axis=0
            )

            # Rotate to target camera
            p_pr = np.dot(q_pr.rotation_matrix.T, np.dot(q_eq.rotation_matrix, p_eq))

            # Filter out points that end up behind the camera
            p_pr = p_pr[:, p_pr[2] > 0]

            # Apply camera intrinsics and project
            p_pr = np.dot(intrinsic, p_pr)
            p_pr = np.stack([p_pr[0] / p_pr[2], p_pr[1] / p_pr[2]], axis=0)

            return p_pr

        # Extract and normalize the input box coordinates
        x0, y0, x1, y1 = box.coords
        if x0 > x1:
            x1 += size_eq[0]

        # "draw" the box with num_samples on each side
        points = get_points(x0, y0, x1, y1)

        # Project to the perspective image
        points = project(points)

        # Filter out points that end up outside of the image
        valid = points[0] >= 0
        valid = np.logical_and(valid, points[1] >= 0)
        valid = np.logical_and(valid, points[0] < size_pr[0])
        valid = np.logical_and(valid, points[1] < size_pr[1])

        if not valid.any():
            return None

        # points = points[:, valid]

        return cls(points, name=box.name, token=box.token)
