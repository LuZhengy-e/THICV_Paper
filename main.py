import cv2
import loguru
import numpy as np
from tqdm import tqdm
from configparser import ConfigParser
from argparse import ArgumentParser
from utils.maps import Point
from utils.sensors import Camera1D
from utils.datasets import AirDataset
from utils.maps import LoaclMap, Pos
from deployment.optimize import ILP
from matplotlib import pyplot as plt

rad = np.pi / 180


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
    HFOV = float(cfg.get("MAP", "HFOV")) * rad
    VFOV = float(cfg.get("MAP", "VFOV")) * rad
    fy = float(cfg.get("MAP", "fy"))

    # read map info
    Hmin = float(cfg.get("MAP", "Hmin"))
    Hmax = float(cfg.get("MAP", "Hmax"))
    delta_H = float(cfg.get("MAP", "delta_H"))
    delta_phi = float(cfg.get("MAP", "delta_phi"))
    delta_pole = float(cfg.get("MAP", "delta_pole"))
    delta_pt = float(cfg.get("MAP", "delta_pt"))

    num_H = int((Hmax - Hmin) / delta_H) + 1
    num_theta = int(np.pi / 2 / (delta_phi * rad)) + 1

    # construct optimize problem
    map_path = cfg.get("MAP", "map_path")
    local_map = LoaclMap(map_path)

    # optimize deployment each road
    plt.figure()
    for line_id in local_map.get_lines():
        pole_index = {}
        point_index = {}
        line = local_map.get_line(line_id)
        if line.get_tag("highway") is None:
            continue

        pt_list = line.get_pts()
        num_pts = len(pt_list)
        pole_idx = 0
        vis_idx = 0
        for i in range(num_pts - 1):
            cur_pt, next_pt = pt_list[i:i + 2]
            pos = Pos.copy_from_point(cur_pt)

            direct = np.array(
                [next_pt.x - cur_pt.x, next_pt.y - cur_pt.y, next_pt.z - cur_pt.z]
            )
            direct /= np.linalg.norm(direct)
            direct2D = direct[0:3] / np.linalg.norm(direct[0:3])

            theta = np.arccos(direct2D[1])
            if direct2D[1] < 0:
                theta = 0 - theta

            while pos.can_walk(direct, delta_pole, next_pt):
                for h in range(num_H):
                    for t in range(num_theta):
                        pole_index[pole_idx] = {
                            "coord": np.array([pos.x, pos.y, pos.z]),
                            "phi": t * delta_phi * rad,
                            "H": Hmin + h * delta_H,
                            "dir": direct.copy(),
                            "theta": theta
                        }
                        pole_index[pole_idx + 1] = {
                            "coord": np.array([pos.x, pos.y, pos.z]),
                            "phi": t * delta_phi * rad,
                            "H": Hmin + h * delta_H,
                            "dir": -direct.copy(),
                            "theta": theta + np.pi
                        }
                        pole_idx += 2
                pos.walk(direct, delta_pole)

            pos = Pos.copy_from_point(cur_pt)
            while pos.can_walk(direct, delta_pt, next_pt):
                point_index[vis_idx] = (pos.x, pos.y, pos.z)
                vis_idx += 1
                pos.walk(direct, delta_pt)

        A = np.zeros((len(point_index), len(pole_index)))
        for vis in tqdm(range(len(point_index))):
            for pole in range(len(pole_index)):
                sensor = Camera1D.create(HFOV, VFOV,
                                         pole_index[pole]["phi"],
                                         pole_index[pole]["H"],
                                         fy)
                point = Point(point_index[vis][0],
                              point_index[vis][1],
                              point_index[vis][2], "0")
                if sensor.is_point_vis(point,
                                       pole_index[pole]["coord"][0:2],
                                       pole_index[pole]["theta"]):
                    A[vis, pole] = 1

        break


if __name__ == '__main__':
    main()
