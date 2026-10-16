import unittest

import numpy as np
from mapillary.research.metropolis.metropolis.utils import geo


class GeoTest(unittest.TestCase):
    def test_ecef_from_lla_known_points(self) -> None:
        np.testing.assert_allclose(
            geo.ecef_from_lla(0.0, 0.0, 0.0), (geo.WGS84_a, 0.0, 0.0), atol=1e-6
        )
        np.testing.assert_allclose(
            geo.ecef_from_lla(90.0, 0.0, 10.0),
            (0.0, 0.0, geo.WGS84_b + 10.0),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            geo.ecef_from_lla(0.0, 90.0, 0.0), (0.0, geo.WGS84_a, 0.0), atol=1e-6
        )

    def test_lla_ecef_roundtrip_scalars_and_arrays(self) -> None:
        x, y, z = geo.ecef_from_lla(10.0, 20.0, 30.0)
        np.testing.assert_allclose(geo.lla_from_ecef(x, y, z), (10.0, 20.0, 30.0))

        lat = np.array([-45.0, 0.0, 60.0])
        lon = np.array([170.0, -20.0, 5.0])
        alt = np.array([0.0, 100.0, -50.0])
        xs, ys, zs = geo.ecef_from_lla(lat, lon, alt)
        lat2, lon2, alt2 = geo.lla_from_ecef(xs, ys, zs)
        np.testing.assert_allclose(lat2, lat)
        np.testing.assert_allclose(lon2, lon)
        np.testing.assert_allclose(alt2, alt, atol=1e-6)

    def test_ecef_from_topocentric_transform_matches_finite_diff(self) -> None:
        analytic = geo.ecef_from_topocentric_transform(30.0, 20.0, 10.0)
        numeric = geo.ecef_from_topocentric_transform_finite_diff(30.0, 20.0, 10.0)
        np.testing.assert_allclose(analytic, numeric, atol=1e-6)
        # Rotation part is orthonormal, translation is the ECEF of the origin.
        np.testing.assert_allclose(
            analytic[:3, :3] @ analytic[:3, :3].T, np.eye(3), atol=1e-12
        )
        np.testing.assert_allclose(
            analytic[:3, 3], geo.ecef_from_lla(30.0, 20.0, 10.0)
        )

    def test_topocentric_axes_point_east_north_up(self) -> None:
        # One meter north / east / up of the reference.
        north = geo.topocentric_from_lla(1e-5, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(north[0], 0.0, places=6)
        self.assertGreater(north[1], 1.0)
        east = geo.topocentric_from_lla(0.0, 1e-5, 0.0, 0.0, 0.0, 0.0)
        self.assertGreater(east[0], 1.0)
        self.assertAlmostEqual(east[1], 0.0, places=6)
        up = geo.topocentric_from_lla(0.0, 0.0, 5.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(up, (0.0, 0.0, 5.0), atol=1e-6)

    def test_topocentric_lla_roundtrip(self) -> None:
        np.testing.assert_allclose(
            geo.topocentric_from_lla(-10.0, 20.0, 100.0, -10.0, 20.0, 100.0),
            (0.0, 0.0, 0.0),
            atol=1e-6,
        )
        x, y, z = geo.topocentric_from_lla(-10.0, 20.0, 100.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(
            geo.lla_from_topocentric(x, y, z, 0.0, 0.0, 0.0),
            (-10.0, 20.0, 100.0),
            atol=1e-6,
        )

    def test_gps_distance(self) -> None:
        distance = geo.gps_distance((42.1, -11.1), (42.2, -11.3))
        self.assertTrue(19000 < distance < 20000)
        self.assertEqual(geo.gps_distance([1.0, 2.0], [1.0, 2.0]), 0.0)
        # One degree of latitude at the equator is about 110.6 km.
        self.assertAlmostEqual(
            geo.gps_distance(np.array([0.0, 0.0]), np.array([1.0, 0.0])) / 1000,
            110.6,
            delta=0.2,
        )

    def test_topocentric_converter(self) -> None:
        converter = geo.TopocentricConverter(47.0, 6.0, 0.0)
        x, y, z = converter.to_topocentric(47.001, 6.002, 10.0)
        np.testing.assert_allclose(
            (x, y, z), geo.topocentric_from_lla(47.001, 6.002, 10.0, 47.0, 6.0, 0.0)
        )
        np.testing.assert_allclose(
            converter.to_lla(x, y, z), (47.001, 6.002, 10.0), atol=1e-6
        )

    def test_topocentric_converter_equality(self) -> None:
        converter = geo.TopocentricConverter(47.0, 6.0, 0.0)
        self.assertEqual(converter, geo.TopocentricConverter(47.0, 6.0, 0.0))
        self.assertNotEqual(converter, geo.TopocentricConverter(47.0, 6.1, 0.0))
        self.assertIs(converter.__eq__("not a converter"), NotImplemented)
