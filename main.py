import cv2
import loguru
import numpy as np
from configparser import ConfigParser
from argparse import ArgumentParser
from shapely.geometry import Point, Polygon
from utils.sensors import Camera1D
from utils.datasets import AirDataset
from utils.maps import LoaclMap, Pos
from deployment.optimize import ILP
from matplotlib import pyplot as plt


def argparse():
    parser = ArgumentParser()
    parser.add_argument("--config-path", default="config/config.cfg")

    args = parser.parse_args()

    return args


def main():
    args = argparse()
    cfg = ConfigParser()
    cfg.read(args.config_path)

    # read sensor info
    HFOV = float(cfg.get("MAP", "HFOV"))
    VFOV = float(cfg.get("MAP", "VFOV"))
    fy = float(cfg.get("MAP", "HFOV"))

    # read map info
    Hmin = float(cfg.get("MAP", "Hmin"))
    Hmax = float(cfg.get("MAP", "Hmax"))
    delta_H = float(cfg.get("MAP", "delta_H"))
    delta_phi = float(cfg.get("MAP", "delta_phi"))
    delta_pole = float(cfg.get("MAP", "delta_pole"))
    delta_pt = float(cfg.get("MAP", "delta_pt"))

    num_H = int((Hmax - Hmin) / delta_H) + 1
    num_theta = int(np.pi / 2 / delta_phi) + 1

    # construct optimize problem
    map_path = cfg.get("MAP", "map_path")
    local_map = LoaclMap(map_path)

    # optimize deployment each road
    plt.figure()
    for line_id in local_map.get_lines():
        pole_index = {}
        vis_index = {}
        line = local_map.get_line(line_id)
        if line.get_tag("highway") is None:
            continue

        pt_list = line.get_pts()
        num_pts = len(pt_list)
        pole_idx = 0
        vis_idx = 0
        for i in range(num_pts - 1):
            cur_pt, next_pt = pt_list[i:i+2]
            pos = Pos.copy_from_point(cur_pt)

            direct = np.array(
                [next_pt.x - cur_pt.x, next_pt.y - cur_pt.y, next_pt.z - cur_pt.z]
            )
            direct /= np.linalg.norm(direct)

            for h in range(num_H):
                for t in range(num_theta):
                    pass

        break


if __name__ == '__main__':
    main()
