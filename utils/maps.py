import cv2
import json
import numpy as np
from copy import deepcopy
from configparser import ConfigParser
from xml.etree.ElementTree import ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, tostring
from utils import Geometry
from sensors import Camera1D


class Point:
    def __init__(self, x, y, z, idx, tags=None):
        if tags is None:
            tags = {}
        self._x, self._y, self._z = x, y, z
        self._coord = np.array([x, y, z])
        self._id = idx
        self._tags = tags

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def id(self):
        return self._id

    def get_tag(self, tag):
        return self._tags.get(tag)

    def update_tag(self, tags: dict):
        for tag, val in tags.items():
            self._tags[tag] = val


class Line:
    def __init__(self, pt_list, idx, tags=None):
        if tags is None:
            tags = {}
        self._pt_list = deepcopy(pt_list)
        self._id = idx
        self._tags = tags

    @property
    def id(self):
        return self._id

    def get_pts(self):
        return deepcopy(self._pt_list)

    def get_tag(self, tag):
        return self._tags.get(tag)

    def update_tag(self, tags: dict):
        for tag, val in tags.items():
            self._tags[tag] = val


class LoaclMap:
    def __init__(self, path, **kwargs):
        self._max_pt_id, self._max_line_id = 0, 0
        self._Mapdict = {"Point": {}, "Line": {}}
        root = ET().parse(path)
        assert root[0].tag == "bounds", "some wrong in OSM file"

        self.reflat = kwargs["minlat"] if kwargs.get("minlat") is not None else float(root[0].attrib["minlat"])
        self.reflon = kwargs["minlon"] if kwargs.get("minlon") is not None else float(root[0].attrib["minlon"])

        for pt in root.iter(tag="node"):
            node = pt.attrib
            idx = node["id"]
            self._max_pt_id = max(int(idx), self._max_pt_id)
            lat, lon = float(node["lat"]), float(node["lon"])
            x, y, z = Geometry.lla2enu(lat, lon, self.reflat, self.reflon)

            tags = {}
            for tag in pt.iter(tag="tag"):
                tags[tag.attrib["k"]] = tag.attrib["v"]

            self._Mapdict["Point"][idx] = Point(x, y, z, idx, tags=deepcopy(tags))

        for line in root.iter(tag="way"):
            way = line.attrib
            idx = way["id"]
            self._max_line_id = max(int(idx), self._max_line_id)
            node_list = []
            for node in line.iter(tag="nd"):
                node_id = node.attrib["ref"]
                node_list.append(self._Mapdict["Point"][node_id])

            tags = {}
            for tag in line.iter(tag="tag"):
                tags[tag.attrib["k"]] = tag.attrib["v"]

            self._Mapdict["Line"][idx] = Line(node_list, idx, deepcopy(tags))

    def create_line(self, pt_list, tags=None):
        if tags is None:
            tags = {}
        self._max_line_id += 1
        self._Mapdict["Line"][str(self._max_line_id)] = Line(pt_list,
                                                             str(self._max_line_id),
                                                             deepcopy(tags))

        return str(self._max_line_id)

    def create_point(self, x, y, z, tags=None):
        if tags is None:
            tags = {}
        self._max_pt_id += 1
        self._Mapdict["Point"][str(self._max_pt_id)] = Point(x, y, z,
                                                             str(self._max_pt_id),
                                                             deepcopy(tags))

        return str(self._max_pt_id)

    def get_line(self, line_id):
        return self._Mapdict["Point"].get(line_id)

    def get_point(self, point_id):
        return self._Mapdict["Line"].get(point_id)

    def get_lines(self):
        for line_id, line in self._Mapdict["Line"].items():
            yield line_id

    def get_points(self):
        for point_id, point in self._Mapdict["Point"].items():
            yield point_id

    def update_point_tag(self, point_id, tags):
        try:
            self._Mapdict["Point"][point_id].update_tag(deepcopy(tags))

        except IndexError:
            raise IndexError(f"{point_id} does not exist")

    def update_line_tag(self, line_id, tags):
        try:
            self._Mapdict["Line"][line_id].update_tag(deepcopy(tags))

        except IndexError:
            raise IndexError(f"{line_id} does not exist")

    def delete_point(self, point_id):
        self._Mapdict["Point"].pop(point_id)

    def delete_line(self, line_id):
        self._Mapdict["Line"].pop(line_id)

    def dump_to_osm(self, output_dir):
        etree = Element()
