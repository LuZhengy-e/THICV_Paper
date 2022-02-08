import numpy as np
from copy import deepcopy
from xml.etree.ElementTree import ElementTree as ET
from xml.etree.ElementTree import Element, SubElement
from matplotlib import pyplot as plt

from .utils import Geometry
from .sensors import Camera1D


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

    def get_tags(self):
        for k, v in self._tags.items():
            yield k, v

    def distance(self, pt):
        assert isinstance(pt, Point), "Please input correct point"
        return np.sqrt((self.x - pt.x) ** 2 + (self.y - pt.y) ** 2 + (self.z - pt.z) ** 2)

    def distance2D(self, pt):
        assert isinstance(pt, Point), "Please input correct point"
        return np.sqrt((self.x - pt.x) ** 2 + (self.y - pt.y) ** 2)


class Pos(Point):
    def __init__(self, x, y, z, idx, tags=None):
        super(Pos, self).__init__(x, y, z, idx, tags)

    def walk(self, direct, step):
        assert np.linalg.norm(direct) != 0, "Incorrect direction"
        if np.linalg.norm(direct) != 1:
            direct = direct / np.linalg.norm(direct)

        self._x += direct[0] * step
        self._y += direct[1] * step
        self._z += direct[2] * step

    def can_walk(self, direct, step, end_point):
        assert np.linalg.norm(direct) != 0, "Incorrect direction"
        if np.linalg.norm(direct) != 1:
            direct = direct / np.linalg.norm(direct)

        delta_x = end_point.x - self.x
        delta_y = end_point.y - self.y
        delta_z = end_point.z - self.z

        if delta_x != 0:
            t = direct[0] * step / delta_x

        elif delta_y != 0:
            t = direct[1] * step / delta_y

        elif delta_z != 0:
            t = direct[2] * step / delta_z

        else:
            raise ValueError("input is wrong")

        return t < 1

    @classmethod
    def copy_from_point(cls, point, if_get_tag=False):
        pos = cls(point.x, point.y, point.z, point.id)
        if if_get_tag:
            for k, v in point.get_tags():
                pos.update_tag({k: v})

        return pos


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

    def get_tags(self):
        for k, v in self._tags.items():
            yield k, v

    @property
    def begin(self):
        return self._pt_list[0]

    @property
    def end(self):
        return self._pt_list[-1]


class LoaclMap:
    def __init__(self, path, **kwargs):
        self._max_pt_id, self._max_line_id = 0, 0
        self._Mapdict = {"Point": {}, "Line": {}}
        root = ET().parse(path)
        assert root[0].tag == "bounds", "some wrong in OSM file"

        self._bounds = root[0].attrib

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

    def indent(self, elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

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
        return self._Mapdict["Line"].get(line_id)

    def get_point(self, point_id):
        return self._Mapdict["Point"].get(point_id)

    def get_lines(self):
        for line_id, line in deepcopy(self._Mapdict["Line"]).items():
            yield line_id

    def get_points(self):
        for point_id, point in deepcopy(self._Mapdict["Point"]).items():
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

    def dump_to_osm(self, output_dir, **kwargs):
        root_dict = {
            "version": "0.6",
            "generator": "CGImap 0.8.3 (3424048 spike-06.openstreetmap.org)",
            "copyright": "OpenStreetMap and contributors",
            "attribution": "http://www.openstreetmap.org/copyright",
            "license": "http://opendatacommons.org/licenses/odbl/1-0/"
        }
        default_node = dict(
            visible="true",
            version="7",
            changeset="86814011",
            timestamp="2020-06-18T09:19:19Z",
            user="luzhengye",
            uid="5077787",
        )

        root = Element("osm")
        for k, v in root_dict.items():
            root.set(k, v)

        bounds = SubElement(root, "bounds")
        for k, v in self._bounds.items():
            bounds.set(k, v)

        node_dict = self._Mapdict["Point"]
        for idx, pt in node_dict.items():
            node = SubElement(root, "node")
            x, y, z = pt.x, pt.y, pt.z
            lat, lon = Geometry.enu2lla(x, y, z, self.reflat, self.reflon)
            node.set("lat", str(lat))
            node.set("lon", str(lon))
            for k, v in default_node.items():
                node.set(k, v)
            node.set("id", idx)

            for k, v in pt.get_tags():
                tag = SubElement(node, "tag")
                tag.set("k", k)
                tag.set("v", v)

        line_dict = self._Mapdict["Line"]
        for idx, line in line_dict.items():
            way = SubElement(root, "way")
            for k, v in default_node.items():
                way.set(k, v)
            way.set("id", idx)
            for pt in line.get_pts():
                ref_node = SubElement(way, "nd")
                ref_node.set("ref", pt.id)

            for k, v in line.get_tags():
                tag = SubElement(way, "tag")
                tag.set("k", k)
                tag.set("v", v)

        # relation = SubElement(root, "relation")

        self.indent(root)
        tree = ET(root)
        tree.write(output_dir, encoding="UTF-8", xml_declaration=True)

    def dump_to_png(self, output_dir, valid_tag=None, **kwargs):
        """
        :param valid_tag:
        :param kwargs
        {
            line_type_i: {
                color: color,
                width: line_width
            }
        }
        """

        if valid_tag is None:
            valid_tag = []

        def plot(line, tag, **kwargs):
            if tag is None:
                return False

            if kwargs.get(tag) is None:
                color = default_color
                width = default_width

            else:
                color = kwargs[tag]["color"] if kwargs[tag].get("color") is not None else default_color
                width = kwargs[tag]["width"] if kwargs[tag].get("width") is not None else default_width

            x = [pt.x for pt in line.get_pts()]
            y = [pt.y for pt in line.get_pts()]
            plt.plot(x, y, c=color, linewidth=width)

            return True

        ax = plt.gca()
        ax.set_aspect(1)
        default_color = "red"
        default_width = 1
        for line_id, line in self._Mapdict["Line"].items():
            if not plot(line, line.get_tag("sensors"), **kwargs):
                highway = line.get_tag("highway")
                if kwargs.get(highway) is None:
                    continue

                plot(line, line.get_tag("highway"), **kwargs)

        plt.savefig(output_dir)
        plt.show()


if __name__ == '__main__':
    rad = np.pi / 180
    localmap = LoaclMap("/home/luzhengye/dataset/1130-DAIR-V2X/Wangjing.osm")
    sensor = Camera1D.create(15 * rad, 23 * rad, np.pi / 2, 3, 2329.297332, 2329.297332)
    for line_id in localmap.get_lines():
        localmap.update_line_tag(line_id, {"id": line_id})

    localmap.dump_to_osm("test.osm")
