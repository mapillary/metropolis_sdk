import unittest
from typing import cast

import numpy as np
from mapillary.research.metropolis.metropolis.utils import geometry_utils as gu
from mapillary.research.metropolis.metropolis.utils.data_classes import Box
from pyquaternion import Quaternion


def _box(center: list[float], size: list[float] | None = None) -> Box:
    return Box(center, size or [2.0, 4.0, 1.0], Quaternion([1, 0, 0, 0]))


class ViewPointsTest(unittest.TestCase):
    def test_perspective_projection_normalizes_depth(self) -> None:
        intrinsic = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
        points = np.array([[1.0, 0.0], [0.5, 0.0], [2.0, 4.0]])
        projected = gu.view_points(points, intrinsic, normalize=True)
        np.testing.assert_allclose(projected[:2], [[100.0, 50.0], [65.0, 40.0]])
        np.testing.assert_allclose(projected[2], [1.0, 1.0])

    def test_orthographic_projection_with_translation(self) -> None:
        view = np.hstack([np.eye(3), np.array([[1.0], [2.0], [3.0]])])
        points = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        projected = gu.view_points(points, view, normalize=False)
        np.testing.assert_allclose(projected, [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

    def test_view_points_validates_shapes(self) -> None:
        with self.assertRaises(AssertionError):
            gu.view_points(np.zeros((2, 3)), np.eye(3), normalize=False)
        with self.assertRaises(AssertionError):
            gu.view_points(np.zeros((3, 3)), np.eye(5), normalize=False)


class EquirectangularTest(unittest.TestCase):
    def test_view_points_eq_maps_directions_to_pixels(self) -> None:
        points = np.array(
            [
                [0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        pixels = gu.view_points_eq(points, 1000, 500)
        # Forward -> image center, right -> 3/4 width, down -> bottom row,
        # left -> 1/4 width.
        np.testing.assert_allclose(
            pixels, [[500.0, 750.0, 500.0, 250.0], [250.0, 250.0, 500.0, 250.0]]
        )

    def test_inverse_map_eq_identity_transform(self) -> None:
        intrinsic = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        # Principal point looks straight ahead -> center of the eq image.
        points = np.array([[50.0, 50.0], [150.0, 50.0]])
        mapped = gu.inverse_map_eq(points, np.eye(4), intrinsic, (2000, 1000))
        np.testing.assert_allclose(mapped[0], [1000.0, 500.0])
        # 45 degrees to the right -> 1/8 of the width right of center.
        np.testing.assert_allclose(mapped[1], [1250.0, 500.0])


class BoxInImageTest(unittest.TestCase):
    INTRINSIC: np.ndarray = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )

    def test_box_fully_visible(self) -> None:
        box = _box([0.0, 0.0, 10.0], [1.0, 1.0, 1.0])
        for level in gu.BoxVisibility:
            self.assertTrue(gu.box_in_image(box, self.INTRINSIC, (100, 100), level))

    def test_box_partially_visible(self) -> None:
        box = _box([4.5, 0.0, 10.0], [1.0, 1.0, 2.0])
        self.assertTrue(
            gu.box_in_image(box, self.INTRINSIC, (100, 100), gu.BoxVisibility.ANY)
        )
        self.assertFalse(
            gu.box_in_image(box, self.INTRINSIC, (100, 100), gu.BoxVisibility.ALL)
        )

    def test_box_behind_camera(self) -> None:
        box = _box([0.0, 0.0, -10.0])
        self.assertFalse(
            gu.box_in_image(box, self.INTRINSIC, (100, 100), gu.BoxVisibility.ANY)
        )
        self.assertTrue(
            gu.box_in_image(box, self.INTRINSIC, (100, 100), gu.BoxVisibility.NONE)
        )

    def test_box_in_image_invalid_level(self) -> None:
        with self.assertRaisesRegex(ValueError, "vis_level"):
            gu.box_in_image(_box([0.0, 0.0, 10.0]), self.INTRINSIC, (100, 100), 7)


class TransformMatrixTest(unittest.TestCase):
    def test_default_is_identity(self) -> None:
        np.testing.assert_allclose(gu.transform_matrix(), np.eye(4))

    def test_forward_and_inverse_compose_to_identity(self) -> None:
        rotation = Quaternion(axis=[0.0, 0.0, 1.0], angle=np.pi / 2)
        translation = np.array([1.0, 2.0, 3.0])
        forward = gu.transform_matrix(translation, rotation)
        inverse = gu.transform_matrix(translation, rotation, inverse=True)

        np.testing.assert_allclose(forward[:3, 3], translation)
        np.testing.assert_allclose(forward @ [1.0, 0.0, 0.0, 1.0], [1.0, 3.0, 3.0, 1.0])
        np.testing.assert_allclose(forward @ inverse, np.eye(4), atol=1e-12)

    def test_rotation_without_matrix_raises(self) -> None:
        with self.assertRaisesRegex(AttributeError, "rotation_matrix"):
            gu.transform_matrix(rotation=cast(Quaternion, object()))
        with self.assertRaisesRegex(AttributeError, "rotation_matrix"):
            gu.transform_matrix(rotation=cast(Quaternion, object()), inverse=True)


class PointsInBoxTest(unittest.TestCase):
    def test_points_in_box(self) -> None:
        box = _box([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
        points = np.array(
            [
                [1.0, 1.9, 2.1, 1.0],
                [1.0, 0.1, 1.0, 1.0],
                [1.0, 1.9, 1.0, -0.5],
            ]
        )
        np.testing.assert_array_equal(
            gu.points_in_box(box, points), [True, True, False, False]
        )

    def test_points_in_box_wlh_factor(self) -> None:
        box = _box([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
        point = np.array([[1.5], [0.0], [0.0]])
        self.assertFalse(gu.points_in_box(box, point)[0])
        self.assertTrue(gu.points_in_box(box, point, wlh_factor=2.0)[0])


class SplitPolyEqTest(unittest.TestCase):
    def test_no_crossing_keeps_single_segment(self) -> None:
        poly = np.array([[100.0, 200.0, 200.0, 150.0], [10.0, 10.0, 20.0, 30.0]])
        segments = gu.split_poly_eq(poly, 1000)
        self.assertEqual(len(segments), 1)
        np.testing.assert_allclose(segments[0], poly)

    def test_crossing_right_edge_splits(self) -> None:
        # 990 -> 10 is shorter across the right edge.
        poly = np.array([[990.0, 10.0], [0.0, 20.0]])
        segments = gu.split_poly_eq(poly, 1000)
        self.assertEqual(len(segments), 2)
        np.testing.assert_allclose(segments[0], [[990.0, 1000.0], [0.0, 10.0]])
        np.testing.assert_allclose(segments[1], [[0.0, 10.0], [10.0, 20.0]])

    def test_crossing_left_edge_splits(self) -> None:
        # 10 -> 990 is shorter across the left edge.
        poly = np.array([[10.0, 990.0], [0.0, 20.0]])
        segments = gu.split_poly_eq(poly, 1000)
        self.assertEqual(len(segments), 2)
        np.testing.assert_allclose(segments[0], [[10.0, 0.0], [0.0, 10.0]])
        np.testing.assert_allclose(segments[1], [[1000.0, 990.0], [10.0, 20.0]])
