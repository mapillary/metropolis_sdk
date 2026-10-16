import contextlib
import io
import json
import os
import tempfile
import unittest
from typing import Any
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
from mapillary.research.metropolis.metropolis import metropolis as metropolis_module
from mapillary.research.metropolis.metropolis.metropolis import Metropolis
from mapillary.research.metropolis.metropolis.utils.data_classes import (
    Box2d,
    EquiBox2d,
)
from PIL import Image


IDENTITY = [1.0, 0.0, 0.0, 0.0]
INTRINSIC = [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]]
SPLIT = "train"


def _sample_data(
    token: str,
    sensor: str,
    filename: str,
    sample: str = "sample1",
    width: int = 200,
    height: int = 200,
) -> dict[str, Any]:
    return {
        "token": token,
        "sample_token": sample,
        "ego_pose_token": "pose1",
        "calibrated_sensor_token": f"cs_{sensor}",
        "filename": filename,
        "width": width,
        "height": height,
        "timestamp": 1_000_000,
        "previous_sample_data": "",
    }


def _tables() -> dict[str, Any]:
    def sensor(name: str, modality: str, channel: str) -> dict[str, Any]:
        return {"token": f"s_{name}", "modality": modality, "channel": channel}

    def calibrated(name: str, intrinsic: list[list[float]]) -> dict[str, Any]:
        return {
            "token": f"cs_{name}",
            "sensor_token": f"s_{name}",
            "translation": [0.0, 0.0, 0.0],
            "rotation": IDENTITY,
            "camera_intrinsic": intrinsic,
        }

    names = ("eq", "front", "mvs", "depth", "radar")
    return {
        "category": [
            {"token": "c_car", "name": "object--vehicle--car"},
            {"token": "c_road", "name": "construction--flat--road"},
        ],
        "attribute": [],
        "instance": [
            {"token": "inst1", "category_token": "c_car"},
            {"token": "inst2", "category_token": "c_car"},
        ],
        "sensor": [
            sensor("eq", "camera", "CAM_EQUIRECTANGULAR"),
            sensor("front", "camera", "CAM_FRONT"),
            sensor("mvs", "mvs", "MVS"),
            sensor("depth", "depth", "DEPTH_FRONT"),
            sensor("radar", "radar", "RADAR"),
        ],
        "calibrated_sensor": [
            calibrated(n, INTRINSIC if n in ("front", "depth") else []) for n in names
        ],
        # The ego vehicle sits 1m along x in the world frame.
        "ego_pose": [
            {"token": "pose1", "translation": [1.0, 0.0, 0.0], "rotation": IDENTITY}
        ],
        "panoptic": [
            {
                "token": "pano1",
                "sample_token": "sample1",
                "filename": "pano.png",
                "category_tokens": [None, "c_road", "c_car"],
                "instance_tokens": [None, None, "inst1"],
            }
        ],
        "scene": [{"token": "scene1"}],
        "sample": [{"token": "sample1"}, {"token": "sample2"}],
        "sample_data": [
            _sample_data("sd_eq", "eq", "eq.png", width=400),
            _sample_data("sd_front", "front", "front.png"),
            _sample_data("sd_mvs", "mvs", "mvs.npz"),
            _sample_data("sd_depth", "depth", "depth.png"),
            _sample_data("sd_radar", "radar", "radar.npz", sample="sample2"),
        ],
        "sample_annotation": [
            # 10m in front of the cameras (world x=1 is the ego origin).
            {
                "token": "ann1",
                "sample_token": "sample1",
                "instance_token": "inst1",
                "translation": [1.0, 0.0, 10.0],
                "size": [1.0, 1.0, 1.0],
                "rotation": IDENTITY,
            },
            # Annotated on another sample, behind the cameras.
            {
                "token": "ann2",
                "sample_token": "sample2",
                "instance_token": "inst2",
                "translation": [1.0, 0.0, -10.0],
                "size": [1.0, 1.0, 1.0],
                "rotation": IDENTITY,
            },
        ],
        "sample_annotation_2d": [
            # Around the center of the 400x200 equirectangular image.
            {
                "token": "a2d_1",
                "sample_token": "sample1",
                "instance_token": "inst1",
                "bounding_box": [190.0, 90.0, 210.0, 110.0],
            },
            # Wraps around the image border, i.e. behind the cameras.
            {
                "token": "a2d_2",
                "sample_token": "sample1",
                "instance_token": "inst2",
                "bounding_box": [395.0, 90.0, 5.0, 110.0],
            },
        ],
    }


def _write_dataset(root: str) -> None:
    table_root = os.path.join(root, SPLIT)
    os.makedirs(table_root)
    for name, records in _tables().items():
        with open(os.path.join(table_root, f"{name}.json"), "w") as f:
            json.dump(records, f)
    geo = {
        "reference": {"lat": 48.0, "lon": 11.0, "alt": 500.0},
        "aerial": {"filename": "aerial.tif"},
    }
    with open(os.path.join(table_root, "geo.json"), "w") as f:
        json.dump(geo, f)

    Image.new("RGB", (400, 200), (10, 20, 30)).save(os.path.join(root, "eq.png"))
    Image.new("RGB", (200, 200), (40, 50, 60)).save(os.path.join(root, "front.png"))
    depth = np.full((200, 200), 10 * 256, dtype=np.uint16)
    depth[:10] = 0
    Image.fromarray(depth).save(os.path.join(root, "depth.png"))
    pano = np.zeros((200, 400), dtype=np.uint8)
    pano[100:] = 1
    pano[90:110, 190:210] = 2
    Image.fromarray(pano).save(os.path.join(root, "pano.png"))
    # One point ahead, one behind, one too close to the sensor (removed).
    points = np.array([[2.0, 0.0, 10.0], [2.0, 0.0, -10.0], [0.5, 0.5, 0.0]])
    np.savez(os.path.join(root, "mvs.npz"), points=points)
    np.savez(os.path.join(root, "radar.npz"), points=points)


class MetropolisTestBase(unittest.TestCase):
    metr: Metropolis

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root: str = self._tmp.name
        _write_dataset(self.root)
        self.metr = Metropolis(SPLIT, self.root, verbose=False)

    def tearDown(self) -> None:
        plt.close("all")
        self._tmp.cleanup()


class LoadingTest(MetropolisTestBase):
    def test_missing_split_raises(self) -> None:
        with self.assertRaisesRegex(AssertionError, "Database version not found"):
            Metropolis("val", self.root, verbose=False)

    def test_verbose_reports_table_sizes(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            Metropolis(SPLIT, self.root, verbose=True)
        self.assertIn("Loading Metropolis tables for split train", out.getvalue())
        self.assertIn("5 sample_data,", out.getvalue())
        self.assertIn("Done reverse indexing", out.getvalue())

    def test_reverse_index(self) -> None:
        metr = self.metr
        self.assertEqual(metr.table_root, os.path.join(self.root, SPLIT))
        self.assertEqual(metr.geo["reference"]["lat"], 48.0)
        sample = metr.get("sample", "sample1")
        self.assertEqual(
            sample["data"],
            {
                "CAM_EQUIRECTANGULAR": "sd_eq",
                "CAM_FRONT": "sd_front",
                "MVS": "sd_mvs",
                "DEPTH_FRONT": "sd_depth",
            },
        )
        self.assertEqual(sample["anns"], ["ann1"])
        self.assertEqual(sample["anns_2d"], ["a2d_1", "a2d_2"])
        self.assertEqual(sample["panoptic_token"], "pano1")
        self.assertEqual(
            metr.get("sample_annotation", "ann1")["category_name"],
            "object--vehicle--car",
        )
        sd = metr.get("sample_data", "sd_mvs")
        self.assertEqual((sd["sensor_modality"], sd["channel"]), ("mvs", "MVS"))

    def test_get_getind_field2token(self) -> None:
        metr = self.metr
        self.assertEqual(metr.getind("sample_data", "sd_front"), 1)
        self.assertEqual(metr.get("category", "c_road")["name"], "construction--flat--road")
        with self.assertRaisesRegex(AssertionError, "Table foo not found"):
            metr.get("foo", "x")
        self.assertEqual(
            metr.field2token("sample_annotation", "instance_token", "inst2"), ["ann2"]
        )
        self.assertEqual(metr.field2token("sample", "token", "missing"), [])

    def test_boxes_and_colors(self) -> None:
        metr = self.metr
        box = metr.get_box("ann1")
        self.assertEqual((box.name, box.token), ("object--vehicle--car", "ann1"))
        np.testing.assert_allclose(box.center, [1.0, 0.0, 10.0])
        box_2d = metr.get_box_2d("a2d_1")
        self.assertEqual(box_2d.coords, [190.0, 90.0, 210.0, 110.0])

        self.assertEqual([b.token for b in metr.get_boxes("sd_front")], ["ann1"])
        self.assertEqual(
            sorted(b.token or "" for b in metr.get_boxes("sd_front", True)),
            ["ann1", "ann2"],
        )
        self.assertEqual(
            [b.token for b in metr.get_boxes_2d("sd_front")], ["a2d_1", "a2d_2"]
        )
        self.assertEqual(metr.get_color("construction--flat--road"), (128, 64, 128))
        self.assertEqual(metr.get_color("unknown"), (0, 0, 0))
        self.assertEqual(
            metr.get_sample_data_path("sd_eq"), os.path.join(self.root, "eq.png")
        )


class GetSampleDataTest(MetropolisTestBase):
    def test_point_sensor(self) -> None:
        data_path, boxes, boxes_2d, intrinsic = self.metr.get_sample_data("sd_mvs")
        self.assertEqual(data_path, os.path.join(self.root, "mvs.npz"))
        self.assertIsNone(boxes_2d)
        self.assertIsNone(intrinsic)
        # Box moved from world into the sensor frame.
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0].center, [0.0, 0.0, 10.0])

    def test_point_sensor_flat_vehicle_coordinates_and_selection(self) -> None:
        _, boxes, _, _ = self.metr.get_sample_data(
            "sd_mvs", selected_anntokens=["ann2"], use_flat_vehicle_coordinates=True
        )
        self.assertEqual([b.token for b in boxes], ["ann2"])
        np.testing.assert_allclose(boxes[0].center, [0.0, 0.0, -10.0])

    def test_perspective_camera_filters_and_projects(self) -> None:
        _, boxes, boxes_2d, intrinsic = self.metr.get_sample_data(
            "sd_front", get_all_visible_boxes=True
        )
        np.testing.assert_allclose(intrinsic, INTRINSIC)
        # The box behind the camera is dropped.
        self.assertEqual([b.token for b in boxes], ["ann1"])
        # Only the 2D box in front projects into the perspective image.
        assert boxes_2d is not None
        self.assertEqual(len(boxes_2d), 1)
        projected = boxes_2d[0]
        assert isinstance(projected, EquiBox2d)
        self.assertEqual(projected.token, "a2d_1")
        np.testing.assert_allclose(projected.points.mean(axis=1), [100.0, 100.0], atol=2.0)

    def test_equirectangular_camera_returns_raw_2d_boxes(self) -> None:
        _, boxes, boxes_2d, _ = self.metr.get_sample_data(
            "sd_eq", selected_2d_anntokens=["a2d_2"]
        )
        self.assertEqual([b.token for b in boxes], ["ann1"])
        assert boxes_2d is not None
        self.assertEqual(len(boxes_2d), 1)
        self.assertIsInstance(boxes_2d[0], Box2d)

    def test_perspective_camera_without_equirectangular_image(self) -> None:
        sample = self.metr.get("sample", "sample1")
        del sample["data"]["CAM_EQUIRECTANGULAR"]
        _, _, boxes_2d, _ = self.metr.get_sample_data("sd_front")
        self.assertEqual(boxes_2d, [])


class PointCloudProjectionTest(MetropolisTestBase):
    def test_map_pointcloud_to_perspective_image(self) -> None:
        points, depths, image = self.metr.map_pointcloud_to_image("sd_mvs", "sd_front")
        self.assertEqual(image.size, (200, 200))
        # Only the point in front of the camera survives: x/z = 0.2 -> u = 120.
        np.testing.assert_allclose(points[:2].T, [[120.0, 100.0]])
        np.testing.assert_allclose(depths, [10.0])

    def test_map_pointcloud_to_equirectangular_image(self) -> None:
        points, depths, image = self.metr.map_pointcloud_to_image("sd_mvs", "sd_eq")
        self.assertEqual(image.size, (400, 200))
        # Both the front and the back point are kept on the sphere.
        np.testing.assert_allclose(depths, [np.sqrt(104.0)] * 2)
        expected_u = (np.arctan2(2.0, [10.0, -10.0]) / (2 * np.pi) + 0.5) * 400
        np.testing.assert_allclose(points[0], expected_u)
        np.testing.assert_allclose(points[1], [100.0, 100.0])

    def test_render_pointcloud_in_image_saves_figure(self) -> None:
        out_path = os.path.join(self.root, "overlay.png")
        _, ax = plt.subplots()
        self.metr.render_pointcloud_in_image(
            "sample1", downsample=1, ax=ax, out_path=out_path
        )
        self.assertEqual(len(ax.images), 1)
        self.assertEqual(len(ax.collections), 1)
        self.assertTrue(os.path.getsize(out_path) > 0)

    def test_render_pointcloud_in_image_creates_axes(self) -> None:
        self.metr.render_pointcloud_in_image("sample1", camera_channel="CAM_EQUIRECTANGULAR")
        self.assertEqual(len(plt.gcf().axes), 1)


class RenderSampleDataTest(MetropolisTestBase):
    def test_render_point_cloud(self) -> None:
        out_path = os.path.join(self.root, "mvs_render.png")
        _, ax = plt.subplots()
        self.metr.render_sample_data("sd_mvs", axes_limit=20, ax=ax, out_path=out_path)
        self.assertEqual(ax.get_title(), "MVS")
        self.assertEqual(ax.get_xlim(), (-20.0, 20.0))
        # Ego marker plus 13 lines for the box.
        self.assertEqual(len(ax.lines), 14)
        self.assertTrue(os.path.exists(out_path))

    def test_render_point_cloud_sensor_frame(self) -> None:
        self.metr.render_sample_data("sd_mvs", use_flat_vehicle_coordinates=False)
        self.assertEqual(plt.gca().get_title(), "MVS")

    def test_render_perspective_camera_2d_and_3d(self) -> None:
        _, ax = plt.subplots()
        self.metr.render_sample_data("sd_front", ax=ax)
        self.assertEqual(ax.get_title(), "CAM_FRONT")
        self.assertEqual(ax.get_xlim(), (0.0, 200.0))
        self.assertEqual(len(ax.lines), 1)  # one projected EquiBox2d

        _, ax3d = plt.subplots()
        self.metr.render_sample_data("sd_front", ax=ax3d, show_3d_boxes=True)
        self.assertEqual(len(ax3d.lines), 13)

    def test_render_equirectangular_camera(self) -> None:
        _, ax = plt.subplots()
        self.metr.render_sample_data("sd_eq", ax=ax)
        # A regular box (4 segments) and a wrap-around box (6 segments).
        self.assertEqual(len(ax.lines), 10)

        _, ax3d = plt.subplots()
        self.metr.render_sample_data("sd_eq", ax=ax3d, show_3d_boxes=True)
        self.assertGreaterEqual(len(ax3d.lines), 12)

    def test_render_depth_map(self) -> None:
        self.metr.render_sample_data("sd_depth")
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "DEPTH_FRONT")
        image = ax.images[0].get_array()
        self.assertEqual(image.shape, (200, 200, 3))

    def test_render_unknown_modality_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown sensor modality"):
            self.metr.render_sample_data("sd_radar")

    def test_render_with_verbose_shows_plot(self) -> None:
        with mock.patch.object(metropolis_module.plt, "show") as show:
            self.metr.render_sample_data("sd_front", verbose=True)
        show.assert_called_once()


class AerialViewTest(MetropolisTestBase):
    def test_requires_gdal(self) -> None:
        with mock.patch.object(metropolis_module, "gdal", None):
            with self.assertRaisesRegex(AssertionError, "GDAL is required"):
                self.metr.render_aerial_view("sd_mvs")

    def test_rejects_images(self) -> None:
        with mock.patch.object(metropolis_module, "gdal", mock.MagicMock()):
            with self.assertRaisesRegex(AssertionError, "only available for pointclouds"):
                self.metr.render_aerial_view("sd_front")

    def test_render_aerial_view_with_fake_gdal(self) -> None:
        band = mock.MagicMock()
        band.ReadAsArray.return_value = np.zeros((50, 60), dtype=np.uint8)
        crop = mock.MagicMock()
        crop.GetRasterBand.return_value = band
        crop.GetGeoTransform.return_value = (11.0, 1e-5, 0.0, 48.0, 0.0, -1e-5)
        gdal = mock.MagicMock()
        gdal.Warp.return_value = crop
        out_path = os.path.join(self.root, "aerial.png")
        _, ax = plt.subplots()

        with mock.patch.object(metropolis_module, "gdal", gdal):
            self.metr.render_aerial_view("sd_mvs", ax=ax, out_path=out_path)

        warp_args = gdal.Warp.call_args
        self.assertEqual(warp_args.args[1], os.path.join(self.root, "aerial.tif"))
        x0, y0, x1, y1 = warp_args.kwargs["outputBounds"]
        self.assertTrue(x0 < 11.0 < x1)
        self.assertTrue(y0 < 48.0 < y1)
        self.assertEqual([c.args[0] for c in crop.GetRasterBand.call_args_list], [1, 2, 3])
        self.assertEqual(ax.get_title(), "MVS")
        self.assertEqual(ax.get_xlim(), (0.0, 60.0))
        self.assertTrue(os.path.exists(out_path))


class PanopticTest(MetropolisTestBase):
    def test_equirectangular_mask_is_returned_as_is(self) -> None:
        record, mask = self.metr.get_panoptic_mask("sd_eq")
        self.assertEqual(record["token"], "pano1")
        self.assertEqual(mask.shape, (200, 400))
        self.assertEqual(mask[100, 200], 2)
        self.assertEqual(mask[150, 10], 1)

    def test_perspective_mask_is_warped(self) -> None:
        _, mask = self.metr.get_panoptic_mask("sd_front")
        self.assertEqual(mask.shape, (200, 200))
        self.assertEqual(mask.dtype, np.uint8)
        # The car occupies the center of the eq image, i.e. straight ahead.
        self.assertEqual(mask[100, 100], 2)
        self.assertEqual(mask[190, 10], 1)
        self.assertEqual(mask[5, 100], 0)

    def test_panoptic_requires_camera(self) -> None:
        with self.assertRaisesRegex(AssertionError, "only be computed for images"):
            self.metr.get_panoptic_mask("sd_mvs")

    def test_render_panoptic(self) -> None:
        out_path = os.path.join(self.root, "panoptic.png")
        out = self.metr.render_panoptic("sd_eq", out_path=out_path)
        self.assertEqual(out.size, (400, 200))
        self.assertEqual(out.mode, "RGB")
        pixels = np.array(out)
        # Road pixels are a 50/50 blend of the image and the road color.
        np.testing.assert_allclose(pixels[180, 10], [69, 42, 79], atol=1)
        # Instance edges are highlighted in white.
        self.assertEqual(tuple(pixels[89, 200]), (255, 255, 255))
        self.assertTrue(os.path.exists(out_path))
