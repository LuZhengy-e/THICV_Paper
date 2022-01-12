import cv2
import numpy as np
from configparser import ConfigParser
from maps import Point, Line, LoaclMap


class Camera1D:
    def __init__(self, min_l, max_l, min_h, max_h):
        self.min_l = min_l
        self.max_l = max_l
        self.min_h = min_h
        self.max_h = max_h

    def deployment(self, local_map: LoaclMap, coord: np.array, theta: np.array, ele):
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ]
        )
        t = coord.copy()
        if len(coord.shape) == 1:
            t = np.expand_dims(t, axis=1)

        corner_pts = np.array(
            [
                [0, self.min_h / 2],
                [0, -self.min_h / 2],
                [self.max_l, -self.max_h / 2],
                [self.max_l, self.max_h / 2]
            ]
        )
        corner_pts = corner_pts.T
        local_coord = np.dot(R, corner_pts) + t

        pt_list = []
        for pt in local_coord.T:
            pt_id = local_map.create_point(pt[0], pt[1], ele)
            pt_list.append(local_map.get_point(pt_id))

        pt_list.append(pt_list[0])
        line_id = local_map.create_line(pt_list, tags={"sensors": "camera"})
