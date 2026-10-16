import os
import tempfile
import unittest
from typing import Any, cast
from unittest import mock

import numpy as np
from mapillary.research.metropolis.metropolis.utils.data_classes import (
    Box,
    Box2d,
    EquiBox2d,
    LidarPointCloud,
)
from pyquaternion import Quaternion


IDENTITY = Quaternion([1, 0, 0, 0])


def _rot_z(angle: float) -> Quaternion:
    return Quaternion(axis=[0.0, 0.0, 1.0], angle=angle)


def _cloud(xyz: list[list[float]]) -> LidarPointCloud:
    points = np.ones((4, len(xyz)))
    points[:3] = np.array(xyz, dtype=float).T
    return LidarPointCloud(points)


class LidarPointCloudTest(unittest.TestCase):
    def test_requires_four_dims(self) -> None:
        with self.assertRaises(AssertionError):
            LidarPointCloud(np.zeros((3, 2)))

    def test_geometry_operations(self) -> None:
        pc = _cloud([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        self.assertEqual(pc.nbr_points(), 2)

        pc.translate(np.array([1.0, 1.0, 1.0]))
        np.testing.assert_allclose(pc.points[:3].T, [[2.0, 1.0, 1.0], [1.0, 3.0, 1.0]])

        pc.rotate(_rot_z(np.pi / 2).rotation_matrix)
        np.testing.assert_allclose(
            pc.points[:3].T, [[-1.0, 2.0, 1.0], [-3.0, 1.0, 1.0]], atol=1e-12
        )

        transform = np.eye(4)
        transform[:3, 3] = [0.0, 0.0, -1.0]
        pc.transform(transform)
        np.testing.assert_allclose(
            pc.points[:3].T, [[-1.0, 2.0, 0.0], [-3.0, 1.0, 0.0]], atol=1e-12
        )
        # Intensity channel is untouched.
        np.testing.assert_allclose(pc.points[3], [1.0, 1.0])

    def test_remove_close(self) -> None:
        pc = _cloud([[0.5, 0.5, 9.0], [0.5, 3.0, 0.0], [3.0, 0.1, 0.0]])
        pc.remove_close(1.0)
        np.testing.assert_allclose(pc.points[:2].T, [[0.5, 3.0], [3.0, 0.1]])

    def test_subsample(self) -> None:
        pc = _cloud([[float(i), 0.0, 0.0] for i in range(10)])
        np.random.seed(0)
        pc.subsample(0.5)
        self.assertEqual(pc.nbr_points(), 5)
        self.assertTrue(set(pc.points[0]).issubset(set(range(10))))

    def test_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cloud.npz")
            np.savez(path, points=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            pc = LidarPointCloud.from_file(path)
        np.testing.assert_allclose(
            pc.points, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0], [1.0, 1.0]]
        )

    def test_from_file_rejects_other_formats(self) -> None:
        with self.assertRaisesRegex(AssertionError, "Unsupported filetype"):
            LidarPointCloud.from_file("cloud.bin")

    def test_render_helpers_scatter_by_channel(self) -> None:
        pc = _cloud([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pc.points[3] = [0.25, 0.75]
        ax = mock.MagicMock()

        pc.render_height(ax, x_lim=(-5, 5), y_lim=(-6, 6), marker_size=3)
        args, kwargs = ax.scatter.call_args
        np.testing.assert_allclose(args[0], [1.0, 4.0])
        np.testing.assert_allclose(args[1], [2.0, 5.0])
        np.testing.assert_allclose(kwargs["c"], [3.0, 6.0])
        self.assertEqual(kwargs["s"], 3)
        ax.set_xlim.assert_called_with(-5, 5)
        ax.set_ylim.assert_called_with(-6, 6)

        pc.render_intensity(ax)
        np.testing.assert_allclose(ax.scatter.call_args.kwargs["c"], [0.25, 0.75])


class MultisweepTest(unittest.TestCase):
    def test_from_file_multisweep_aggregates_into_reference_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(
                os.path.join(tmp, "sweep1.npz"),
                points=np.array([[5.0, 0.0, 0.0], [0.1, 0.1, 0.0]]),
            )
            np.savez(
                os.path.join(tmp, "sweep0.npz"), points=np.array([[5.0, 0.0, 0.0]])
            )
            tables: dict[str, dict[str, dict[str, Any]]] = {
                "sample_data": {
                    "sd1": {
                        "filename": "sweep1.npz",
                        "ego_pose_token": "pose1",
                        "calibrated_sensor_token": "cs",
                        "timestamp": 2_000_000,
                        "previous_sample_data": "sd0",
                    },
                    "sd0": {
                        "filename": "sweep0.npz",
                        "ego_pose_token": "pose0",
                        "calibrated_sensor_token": "cs",
                        "timestamp": 1_500_000,
                        "previous_sample_data": "",
                    },
                },
                "ego_pose": {
                    "pose1": {"translation": [10.0, 0.0, 0.0], "rotation": [1, 0, 0, 0]},
                    "pose0": {"translation": [8.0, 0.0, 0.0], "rotation": [1, 0, 0, 0]},
                },
                "calibrated_sensor": {
                    "cs": {"translation": [0.0, 0.0, 1.0], "rotation": [1, 0, 0, 0]}
                },
            }
            metr = mock.MagicMock()
            metr.dataroot = tmp
            metr.get.side_effect = lambda table, token: tables[table][token]

            pc, times = LidarPointCloud.from_file_multisweep(
                metr, {"data": {"LIDAR": "sd1"}}, "LIDAR", "LIDAR", nsweeps=5
            )

        # The close point is removed; the older sweep is shifted by the 2m of
        # ego motion between the two sweeps.
        np.testing.assert_allclose(
            pc.points[:3].T, [[5.0, 0.0, 0.0], [3.0, 0.0, 0.0]], atol=1e-12
        )
        np.testing.assert_allclose(times, [[0.0, 0.5]])

    def test_from_file_multisweep_respects_nsweeps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(os.path.join(tmp, "s.npz"), points=np.array([[5.0, 0.0, 0.0]]))
            record = {
                "filename": "s.npz",
                "ego_pose_token": "p",
                "calibrated_sensor_token": "c",
                "timestamp": 0,
                "previous_sample_data": "sd",
            }
            pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1, 0, 0, 0]}
            metr = mock.MagicMock()
            metr.dataroot = tmp
            metr.get.side_effect = lambda table, token: (
                record if table == "sample_data" else pose
            )

            pc, times = LidarPointCloud.from_file_multisweep(
                metr, {"data": {"L": "sd"}}, "L", "L", nsweeps=3
            )

        self.assertEqual(pc.nbr_points(), 3)
        self.assertEqual(times.shape, (1, 3))


class BoxTest(unittest.TestCase):
    def test_init_validates_inputs(self) -> None:
        with self.assertRaises(AssertionError):
            Box([0.0, float("nan"), 0.0], [1.0, 1.0, 1.0], IDENTITY)
        with self.assertRaises(AssertionError):
            Box([0.0, 0.0], [1.0, 1.0, 1.0], IDENTITY)
        with self.assertRaises(AssertionError):
            Box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], cast(Quaternion, [1, 0, 0, 0]))

    def test_equality_and_copy(self) -> None:
        box = Box([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], IDENTITY, name="car", token="t")
        copied = box.copy()
        self.assertEqual(box, copied)
        self.assertIsNot(box.center, copied.center)
        copied.translate(np.array([1.0, 0.0, 0.0]))
        self.assertNotEqual(box, copied)
        self.assertIs(box.__eq__("box"), NotImplemented)

    def test_repr(self) -> None:
        box = Box([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], IDENTITY, name="car", token="t")
        self.assertIn("xyz: [1.00, 2.00, 3.00]", repr(box))
        self.assertIn("lwh: [4.00, 5.00, 6.00]", repr(box))
        self.assertIn("name: car, token: t", repr(box))

    def test_corners_layout(self) -> None:
        box = Box([0.0, 0.0, 0.0], [4.0, 2.0, 1.0], IDENTITY)
        corners = box.corners()
        self.assertEqual(corners.shape, (3, 8))
        # First four corners face forward (+y), last four backwards.
        np.testing.assert_allclose(corners[1, :4], [2.0] * 4)
        np.testing.assert_allclose(corners[1, 4:], [-2.0] * 4)
        np.testing.assert_allclose(np.abs(corners[0]), [1.0] * 8)
        np.testing.assert_allclose(np.abs(corners[2]), [0.5] * 8)
        np.testing.assert_allclose(box.corners(2.0), 2.0 * corners)
        np.testing.assert_allclose(box.bottom_corners()[2], [-0.5] * 4)

    def test_rotate_and_translate(self) -> None:
        box = Box([1.0, 0.0, 0.0], [4.0, 2.0, 1.0], IDENTITY)
        box.rotate(_rot_z(np.pi / 2))
        np.testing.assert_allclose(box.center, [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            box.rotation_matrix, _rot_z(np.pi / 2).rotation_matrix
        )
        box.translate(np.array([0.0, 0.0, 2.0]))
        np.testing.assert_allclose(box.center, [0.0, 1.0, 2.0], atol=1e-12)
        # After a 90 degree yaw the length runs along x.
        np.testing.assert_allclose(
            np.ptp(box.corners()[0]), 4.0, atol=1e-12
        )

    def test_render_draws_box_edges(self) -> None:
        box = Box([0.0, 0.0, 10.0], [1.0, 1.0, 1.0], IDENTITY)
        axis = mock.MagicMock()
        box.render(axis, colors=("g", "m", "y"), linewidth=3)
        # 4 sides + 4 front + 4 rear + 1 heading line.
        self.assertEqual(axis.plot.call_count, 13)
        colors = [c.kwargs["color"] for c in axis.plot.call_args_list]
        self.assertEqual(colors.count("y"), 4)
        self.assertEqual(colors.count("m"), 4)
        self.assertEqual(colors.count("g"), 5)
        self.assertTrue(all(c.kwargs["linewidth"] == 3 for c in axis.plot.call_args_list))

    def test_render_eq_draws_all_edges(self) -> None:
        box = Box([0.0, 0.0, 10.0], [1.0, 1.0, 1.0], IDENTITY)
        axis = mock.MagicMock()
        box.render_eq(axis, (1000, 500), num_samples=5)
        # 12 edges, none crosses the image border for a box straight ahead.
        self.assertEqual(axis.plot.call_count, 12)
        for c in axis.plot.call_args_list:
            self.assertEqual(len(c.args[0]), 5)
            self.assertTrue(np.all((c.args[0] > 400) & (c.args[0] < 600)))

    def test_render_eq_splits_edges_crossing_border(self) -> None:
        # A box straight behind the camera straddles the left/right image edge.
        box = Box([0.0, 0.0, -10.0], [4.0, 4.0, 1.0], IDENTITY)
        axis = mock.MagicMock()
        box.render_eq(axis, (1000, 500), num_samples=5)
        self.assertGreater(axis.plot.call_count, 12)


class Box2dTest(unittest.TestCase):
    def test_init_equality_and_repr(self) -> None:
        box = Box2d([1.0, 2.0, 3.0, 4.0], name="sign", token="t")
        self.assertEqual(box, Box2d([1.0, 2.0, 3.0, 4.0]))
        self.assertNotEqual(box, Box2d([1.0, 2.0, 3.0, 5.0]))
        self.assertIs(box.__eq__(1), NotImplemented)
        self.assertEqual(repr(box), "[1.00, 2.00, 3.00, 4.00], name=sign, token=t")
        with self.assertRaises(AssertionError):
            Box2d([1.0, 2.0, 3.0])
        with self.assertRaises(AssertionError):
            Box2d([1.0, float("nan"), 3.0, 4.0])

    def test_render_regular_box(self) -> None:
        axis = mock.MagicMock()
        Box2d([10.0, 20.0, 30.0, 40.0]).render(axis, 100, color="b", linewidth=1)
        segments = [(c.args[0], c.args[1]) for c in axis.plot.call_args_list]
        self.assertEqual(
            segments,
            [
                ([10.0, 30.0], [20.0, 20.0]),
                ([30.0, 30.0], [20.0, 40.0]),
                ([30.0, 10.0], [40.0, 40.0]),
                ([10.0, 10.0], [40.0, 20.0]),
            ],
        )
        axis.plot.assert_called_with([10.0, 10.0], [40.0, 20.0], color="b", linewidth=1)

    def test_render_wrapping_box(self) -> None:
        axis = mock.MagicMock()
        Box2d([90.0, 20.0, 10.0, 40.0]).render(axis, 100)
        segments = [(c.args[0], c.args[1]) for c in axis.plot.call_args_list]
        self.assertEqual(len(segments), 6)
        self.assertIn(([90.0, 100], [20.0, 20.0]), segments)
        self.assertIn(([0, 10.0], [20.0, 20.0]), segments)


class EquiBox2dTest(unittest.TestCase):
    INTRINSIC: np.ndarray = np.array(
        [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]]
    )

    def test_init_and_render(self) -> None:
        points = np.array([[1.0, 2.0], [3.0, 4.0]])
        box = EquiBox2d(points, name="n", token="t")
        axis = mock.MagicMock()
        box.render(axis, color="g", linewidth=4)
        axis.plot.assert_called_once()
        np.testing.assert_allclose(axis.plot.call_args.args[0], [1.0, 2.0])
        self.assertEqual(axis.plot.call_args.kwargs, {"color": "g", "linewidth": 4})
        with self.assertRaises(AssertionError):
            EquiBox2d(np.zeros((3, 2)))

    def test_from_box_2d_projects_center_box(self) -> None:
        # A box around the center of a 1000x500 eq image, seen by a forward
        # perspective camera with the same orientation.
        box = Box2d([480.0, 240.0, 520.0, 260.0], name="sign", token="tok")
        result = EquiBox2d.from_box_2d(
            box, IDENTITY, IDENTITY, self.INTRINSIC, (1000, 500), (200, 200), 5
        )
        assert result is not None
        self.assertEqual((result.name, result.token), ("sign", "tok"))
        self.assertEqual(result.points.shape, (2, 20))
        center = result.points.mean(axis=1)
        np.testing.assert_allclose(center, [100.0, 100.0], atol=1.0)
        self.assertTrue(np.all(np.abs(result.points - 100.0) < 30.0))

    def test_from_box_2d_wrapping_box(self) -> None:
        # Box wraps around the eq image border, i.e. it is behind; a camera
        # rotated by 180 degrees sees it in front.
        box = Box2d([990.0, 240.0, 10.0, 260.0])
        result = EquiBox2d.from_box_2d(
            box,
            IDENTITY,
            Quaternion(axis=[0.0, 1.0, 0.0], angle=np.pi),
            self.INTRINSIC,
            (1000, 500),
            (200, 200),
            5,
        )
        assert result is not None
        np.testing.assert_allclose(result.points.mean(axis=1), [100.0, 100.0], atol=1.0)

    def test_from_box_2d_outside_returns_none(self) -> None:
        box = Box2d([480.0, 240.0, 520.0, 260.0])
        result = EquiBox2d.from_box_2d(
            box,
            IDENTITY,
            Quaternion(axis=[0.0, 1.0, 0.0], angle=np.pi),
            self.INTRINSIC,
            (1000, 500),
            (200, 200),
        )
        self.assertIsNone(result)
