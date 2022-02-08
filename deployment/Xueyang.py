import numpy as np
from shapely.geometry import Polygon
from collections import defaultdict

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment.optimize import DP_OBJECT
from utils.maps import LoaclMap
from utils.maps import Line, Point, Pos
from utils.sensors import Camera1D

MAX_VALUE = 1e6


@DP_OBJECT.register
def deployment_one_road(map: LoaclMap, road_id, gap_pole, cameras_info, cost_pole, cost_cameras):
    road: Line = map.get_line(road_id)
    pt_list = road.get_pts()

    pts = []
    thetas = []

    for i in range(len(pt_list) - 1):
        cur_pt, next_pt = pt_list[i:i + 2]
        pos = Pos.copy_from_point(cur_pt)

        direct = np.array(
            [next_pt.x - cur_pt.x, next_pt.y - cur_pt.y, next_pt.z - cur_pt.z]
        )
        direct /= np.linalg.norm(direct)
        direct2D = direct[0:3] / np.linalg.norm(direct[0:3])

        theta = np.arccos(direct2D[0])
        if direct2D[1] < 0:
            theta = 0 - theta

        pts.append(Point(pos.x, pos.y, pos.z, idx="0"))
        thetas.append(theta)
        while pos.can_walk(direct, gap_pole, next_pt):
            pos.walk(direct, gap_pole)
            pts.append(Point(pos.x, pos.y, pos.z, idx="0"))
            thetas.append(theta)

    sensor = Camera1D(cameras_info[0], cameras_info[1], cameras_info[2], cameras_info[3])

    # dp
    mid_result = defaultdict(dict)
    dp = [[MAX_VALUE] * 2 for _ in range(len(pts))]
    dp[0][0] = 0
    dp[0][1] = cost_pole
    for i in range(1, len(pts)):
        for j in range(i):
            if not sensor.is_point_vis(pts[i], np.array([pts[j].x, pts[j].y]), thetas[j]):
                continue
            if dp[i][0] > dp[j][1] + cost_cameras:
                dp[i][0] = dp[j][1] + cost_cameras
                mid_result[i][0] = (j, 1, 0)

        dp[i][1] = dp[i][0] + cost_pole
        mid_result[i][1] = mid_result[i][0]

        for j in range(i):
            if not sensor.is_point_vis(pts[j], np.array([pts[i].x, pts[i].y]), np.pi + thetas[i]):
                continue
            tmp_cost = min(dp[j][0], dp[j][1]) + cost_cameras + cost_pole
            if tmp_cost < dp[i][1]:
                dp[i][1] = tmp_cost
                mid_result[i][1] = (j, 0, 1) if dp[j][0] < dp[j][1] else (j, 1, 1)

    cur_idx = 0 if dp[-1][0] <= dp[-1][1] else 1
    cur_pos = len(pts) - 1

    while cur_pos > 0:
        last_pos, last_idx, if_install = mid_result[cur_pos][cur_idx]
        if cur_idx == 0 or not if_install:
            sensor.deployment(map,
                              np.array([pts[last_pos].x, pts[last_pos].y]),
                              thetas[last_pos],
                              pts[last_pos].z)

        else:
            sensor.deployment(map,
                              np.array([pts[cur_pos].x, pts[cur_pos].y]),
                              np.pi + thetas[cur_pos],
                              pts[cur_pos].z)

        cur_pos = last_pos
        cur_idx = last_idx

    return min(dp[-1])
