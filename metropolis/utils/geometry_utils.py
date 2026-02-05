# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe

# Original copyright notice:
# nuScenes dev-kit.
# Code written by Oscar Beijbom and Alex Lang, 2018.

from enum import IntEnum
from typing import Tuple, TYPE_CHECKING, List

import numpy as np
from pyquaternion import Quaternion

# The following is a trick to solve a cyclic import loop
if TYPE_CHECKING:
    from .data_classes import Box  # noqa F409


class BoxVisibility(IntEnum):
    """Enumerates the various level of box visibility in an image"""

    ALL = 0  # Requires all corners are inside the image.
    ANY = 1  # Requires at least one corner visible in the image.
    NONE = (
        2  # Requires no corners to be inside, i.e. box can be fully outside the image.
    )


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    Args:
        points: Matrix of points, where each point (x, y, z) is along each column.
        view: Defines an arbitrary projection (n <= 4). The projection should be
            such that the corners are projected onto the first 2 axis.
        normalize: Whether to normalize the remaining coordinate (along the third axis).

    Returns:
        Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def view_points_eq(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Project 3D points to equirectangular image

    Args:
        points: 3xN array of points.
        width: Image width.
        height: Image height

    Returns:
        A 2xN array of projected points in pixel coordinates.
    """
    # Polar coordinates
    u = np.arctan2(points[0, :], points[2, :])
    v = np.arctan2(points[1, :], np.sqrt(points[0, :] ** 2 + points[2, :] ** 2))

    # Map to image
    u_pix = (u / (2 * np.pi) + 0.5) * width
    v_pix = (v / np.pi + 0.5) * height

    return np.stack([u_pix, v_pix], axis=0)


def inverse_map_eq(
    points: np.ndarray,
    transform: np.ndarray,
    intrinsics: np.ndarray,
    eq_size: Tuple[int, int],
) -> np.ndarray:
    """Inverse map function to warp from perspective to equirect. using scikit-image

    Args:
        points: Points to inverse map, given as an Nx2 array of (x, y) pixel coordinates
            in the perspective image.
        transform: 3D transformation from the perspective image frame to the equirectangular
            image frame, give as a 4x4 matrix.
        intrinsics: Intrinsic parameters of the perspective camera, given as a 3x3 matrix
        eq_size: Size of the equirectangular image, given as (height, width)

    Returns:
        Inverse mapped points, given as an Nx2 array of (x, y) pixel coordinates in
        the equirectangular image.
    """
    # Pixels to image plane
    points = np.concatenate(
        [points.T, np.ones((1, points.shape[0]), dtype=points.dtype)],
        axis=0,
    )
    points = np.linalg.solve(intrinsics, points)

    # Perspective camera frame to equirectangular camera frame
    points = np.concatenate(
        [points, np.ones((1, points.shape[1]), dtype=points.dtype)], axis=0
    )
    points = np.dot(transform, points)
    points = points[:3, :]

    # Equirectangular projection
    points = view_points_eq(points, eq_size[0], eq_size[1])

    return points.T


def box_in_image(
    box: "Box",
    intrinsic: np.ndarray,
    imsize: Tuple[int, int],
    vis_level: int = BoxVisibility.ANY,
) -> bool:
    """Check if a box is visible inside an image without accounting for occlusions.

    Args:
        box: The box to be checked.
        intrinsic: Intrinsic camera matrix.
        imsize: (width, height).
        vis_level: One of the enumerations of `BoxVisibility`.

    Returns:
        True if visibility condition is satisfied.
    """

    corners_3d = box.corners()
    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
    visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    visible = np.logical_and(visible, corners_3d[2, :] > 1)

    in_front = (
        corners_3d[2, :] > 0.1
    )  # True if a corner is at least 0.1 meter in front of the camera.

    if vis_level == BoxVisibility.ALL:
        return all(visible) and all(in_front)
    elif vis_level == BoxVisibility.ANY:
        return any(visible) and all(in_front)
    elif vis_level == BoxVisibility.NONE:
        return True
    else:
        raise ValueError(f"vis_level: {vis_level} not valid")


DEFAULT_TRANSLATION = np.array([0, 0, 0])
DEFAULT_ROTATION = Quaternion([1, 0, 0, 0])


def transform_matrix(
    translation: np.ndarray = DEFAULT_TRANSLATION,
    rotation: Quaternion = DEFAULT_ROTATION,
    inverse: bool = False,
) -> np.ndarray:
    """Convert pose to transformation matrix.

    Args:
        translation: Translation in x, y, z.
        rotation: Rotation in quaternions (w ri rj rk).
        inverse: Whether to compute inverse transform matrix.

    Returns:
        Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rotation_matrix = getattr(rotation, 'rotation_matrix', None)
        if rotation_matrix is not None:
            rot_inv = rotation_matrix.T
        else:
            raise AttributeError("Quaternion object does not have rotation_matrix attribute")
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        rotation_matrix = getattr(rotation, 'rotation_matrix', None)
        if rotation_matrix is not None:
            tm[:3, :3] = rotation_matrix
        else:
            raise AttributeError("Quaternion object does not have rotation_matrix attribute")
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


def points_in_box(
    box: "Box", points: np.ndarray, wlh_factor: float = 1.0
) -> np.ndarray:
    """Checks whether points are inside the box.

    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579

    Args:
        box: A 3D bounding box.
        points: Points to check.
        wlh_factor: Inflates or deflates the box.

    Returns:
        A boolean array with the check result for each point.
    """
    # pyre-fixme[28]: Unexpected keyword argument `wlh_factor`.
    corners = box.corners(wlh_factor=wlh_factor)

    p1 = corners[:, 0]
    p_x = corners[:, 4]
    p_y = corners[:, 1]
    p_z = corners[:, 3]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape((-1, 1))

    iv = np.dot(i, v)
    jv = np.dot(j, v)
    kv = np.dot(k, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    return mask


def split_poly_eq(poly: np.ndarray, width: int) -> List[np.ndarray]:
    """Split a polygon into multiple segments to render it properly on an eq image

    When projecting lines (e.g. 3D bounding box edges) to an equirectangular image,
    we obtain curves that can cross over from the right to the left edge of the
    image (or vice vers), potentially multiple times. Plotting these naively produces
    large artifacts. To avoid this, this function splits the projection (given as
    a polygon) into multiple segments which, when plotted independently, produce
    the correct visualization.

    Args:
        poly: A polygon, i.e. a sequence of 2D points given as a 2 x N array.
        width: Width of the equirectangular image in pixels.

    Returns:
        A list of polygon segments to be plotted separately.
    """
    segments = []
    curr_segment = []

    for i in range(0, poly.shape[1] - 1):
        x1 = poly[0, i]
        x2 = poly[0, i + 1]

        y1 = poly[1, i]
        y2 = poly[1, i + 1]

        if x2 > x1:
            d_dir = x2 - x1
            d_crs = x1 + width - x2

            if d_crs < d_dir:
                # Crossing is better, we need to split
                y_interp = (width - x2) * (y1 - y2) / (x1 + width - x2) + y2

                # Finish the current segment
                curr_segment.append((x1, y1))
                curr_segment.append((0, y_interp))
                segments.append(curr_segment)

                # Start a new one
                curr_segment = [(width, y_interp)]
            else:
                # No need to split
                curr_segment.append((x1, y1))
        elif x1 > x2:
            d_dir = x1 - x2
            d_crs = x2 + width - x1

            if d_crs < d_dir:
                # Crossing is better, we need to split
                y_interp = (width - x1) * (y2 - y1) / (x2 + width - x1) + y1

                # Finish the current segment
                curr_segment.append((x1, y1))
                curr_segment.append((width, y_interp))
                segments.append(curr_segment)

                # Start a new one
                curr_segment = [(0, y_interp)]
            else:
                # No need to split
                curr_segment.append((x1, y1))
        else:
            # No need to split
            curr_segment.append((x1, y1))

    curr_segment.append((poly[0, -1], poly[1, -1]))
    segments.append(curr_segment)

    # TODO: covert segments to numpy
    return [np.array(segment).T for segment in segments]
