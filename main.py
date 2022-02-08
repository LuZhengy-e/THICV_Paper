from deployment import build_from_dp
import loguru

import numpy as np
from configparser import ConfigParser
from argparse import ArgumentParser
from tqdm import tqdm
from deployment.utils.maps import LoaclMap
from matplotlib import pyplot as plt

logger = loguru.logger
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

    # construct optimize problem
    map_path = cfg.get("MAP", "map_path")
    local_map = LoaclMap(map_path)

    # optimize deployment each road
    plt.figure()
    for line_id in tqdm(local_map.get_lines()):
        local_map.update_line_tag(line_id, {"id": line_id})
        line = local_map.get_line(line_id)
        if line.get_tag("highway") != "secondary" and line.get_tag("highway") != "tertiary":
            continue

        min_cost = build_from_dp("deployment_one_road",
                                 map=local_map, road_id=line_id,
                                 gap_pole=delta_pole,
                                 cameras_info=[0, 90, 4, 40],
                                 cost_pole=5,
                                 cost_cameras=10)

        logger.info(f"{line_id} has deployed")

    # output
    osm_path = cfg.get("FILE", "osm_path")
    fig_path = cfg.get("FILE", "fig_path")
    picture_info = eval(cfg.get("FILE", "picture_info"))
    local_map.dump_to_osm(osm_path)
    local_map.dump_to_png(fig_path, **picture_info)


if __name__ == '__main__':
    main()
