# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, Tuple

import matplotlib.pyplot as plt


def get_colormap() -> Dict[str, Tuple[int, int, int]]:
    """Get the defined colormap.

    Returns:
        A mapping from the class names to the respective RGB values.
    """

    classname_to_color = {  # RGB.
        "construction--barrier--concrete-block": (250, 170, 32),
        "void--car-mount": (32, 32, 32),
        "void--dynamic": (111, 74, 0),
        "void--ego-vehicle": (120, 10, 10),
        "void--ground": (81, 0, 81),
        "void--static": (111, 111, 0),
        "construction--flat--bike-lane": (128, 64, 255),
        "construction--flat--curb-cut": (170, 170, 170),
        "construction--flat--parking": (250, 170, 160),
        "construction--flat--parking-aisle": (250, 170, 37),
        "construction--flat--pedestrian-area": (96, 96, 96),
        "construction--flat--rail-track": (230, 150, 140),
        "construction--flat--road": (128, 64, 128),
        "construction--flat--road-shoulder": (110, 110, 110),
        "construction--flat--service-lane": (110, 110, 110),
        "construction--flat--sidewalk": (244, 35, 232),
        "construction--flat--traffic-island": (128, 196, 128),
        "construction--barrier--curb": (196, 196, 196),
        "construction--barrier--fence": (190, 153, 153),
        "construction--barrier--guard-rail": (180, 165, 180),
        "construction--barrier--other-barrier": (90, 120, 150),
        "construction--barrier--road-median": (250, 170, 33),
        "construction--barrier--road-side": (250, 170, 34),
        "construction--barrier--separator": (128, 128, 128),
        "construction--barrier--wall": (102, 102, 156),
        "construction--structure--bridge": (150, 100, 100),
        "construction--structure--building": (70, 70, 70),
        "construction--structure--garage": (150, 150, 150),
        "construction--structure--tunnel": (150, 120, 90),
        "marking--continuous--dashed": (255, 255, 255),
        "marking--continuous--solid": (255, 255, 255),
        "marking--continuous--zigzag": (250, 170, 29),
        "nature--mountain": (64, 170, 64),
        "nature--sand": (230, 160, 50),
        "nature--sky": (70, 130, 180),
        "nature--snow": (190, 255, 255),
        "nature--terrain": (152, 251, 152),
        "nature--vegetation": (107, 142, 35),
        "nature--water": (0, 170, 30),
        "object--pothole": (70, 100, 150),
        "marking--discrete--hatched--chevron": (250, 170, 12),
        "marking--discrete--hatched--diagonal": (250, 170, 11),
        "marking-only--continuous--dashed": (255, 255, 255),
        "marking-only--discrete--crosswalk-zebra": (255, 255, 255),
        "marking-only--discrete--other-marking": (255, 255, 255),
        "marking-only--discrete--text": (255, 255, 255),
        "construction--flat--crosswalk-plain": (140, 140, 200),
        "human--person": (220, 20, 60),
        "human--person--group": (220, 10, 30),
        "human--rider--bicyclist": (255, 0, 0),
        "human--rider--other-rider": (255, 0, 200),
        "marking--discrete--arrow--left": (250, 170, 26),
        "marking--discrete--arrow--other": (250, 170, 25),
        "marking--discrete--arrow--right": (250, 170, 24),
        "marking--discrete--arrow--straight": (250, 170, 20),
        "marking--discrete--crosswalk-zebra": (255, 255, 255),
        "marking--discrete--other-marking": (255, 255, 255),
        "marking--discrete--text": (250, 170, 15),
        "object--banner": (255, 255, 128),
        "object--bench": (250, 0, 30),
        "object--bike-rack": (100, 140, 180),
        "object--catch-basin": (220, 128, 128),
        "object--cctv-camera": (222, 40, 40),
        "object--fire-hydrant": (100, 170, 30),
        "object--junction-box": (40, 40, 40),
        "object--mailbox": (33, 33, 33),
        "object--manhole": (100, 128, 160),
        "object--sign--advertisement": (250, 171, 30),
        "object--sign--information": (250, 174, 30),
        "object--sign--other": (250, 175, 30),
        "object--sign--store": (250, 176, 30),
        "object--street-light": (210, 170, 100),
        "object--support--pole": (153, 153, 153),
        "object--support--traffic-sign-frame": (128, 128, 128),
        "object--support--utility-pole": (0, 0, 80),
        "object--trash-can": (140, 140, 20),
        "object--traffic-light": (250, 170, 30),
        "object--traffic-sign": (220, 220, 0),
        "object--tunnel-light": (210, 170, 100),
        "object--tunnel-light--group": (210, 120, 50),
        "object--vehicle--bicycle": (119, 11, 32),
        "object--vehicle--bus": (0, 60, 100),
        "object--vehicle--car": (0, 0, 142),
        "object--vehicle--group": (128, 64, 64),
        "object--vehicle--motorcycle": (0, 0, 230),
        "object--vehicle--other-vehicle": (128, 64, 64),
        "object--vehicle--truck": (0, 0, 70),
        "object--vehicle--wheeled-slow": (0, 0, 192),
    }

    return classname_to_color


def plot_deph_normalized_colormap(depth_map, scale_range):
    # depth=0 is the area with no depth data, will be colorized as far away background
    depth_map[depth_map <= 0] = 255
    # the color map wil be spread within 0 - scale_range
    depth_map[depth_map > scale_range] = scale_range
    depth_map = depth_map / scale_range
    cmap = plt.cm.jet_r
    rgba = cmap(depth_map)
    rgb = rgba[:, :, 0:3] * 255.0

    return rgb
