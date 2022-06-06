from deployment import build_from_dp
import loguru

import numpy as np
from configparser import ConfigParser
from argparse import ArgumentParser
from deployment.utils.maps import LoaclMap

from config import config

logger = loguru.logger
rad = np.pi / 180


def argparse():
    parser = ArgumentParser()
    parser.add_argument("--config-path", default="config/config.cfg")
    parser.add_argument("--method", default="LuZhengye")

    args = parser.parse_args()

    return args


def main():
    args = argparse()
    cfg = ConfigParser()
    cfg.read(args.config_path)

    # construct optimize problem
    map_path = cfg.get("MAP", "map_path")
    local_map = LoaclMap(map_path)

    # optimize deployment each road
    params = config.get(args.method)
    params["map"] = local_map
    method = params.pop("method")
    build_from_dp(method, **params)

    osm_path = cfg.get("FILE", "osm_path")
    fig_path = cfg.get("FILE", "fig_path")
    picture_info = eval(cfg.get("FILE", "picture_info"))
    local_map.dump_to_osm(osm_path)
    local_map.dump_to_png(fig_path, **picture_info)


if __name__ == '__main__':
    main()
