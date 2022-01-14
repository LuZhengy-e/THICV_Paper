import cv2
import numpy as np
from configparser import ConfigParser

MAX_VALUE = 1e6


class Camera1D:
    def __init__(self, min_l, max_l, min_h, max_h):
        self.min_l = min_l
        self.max_l = max_l
        self.min_h = min_h
        self.max_h = max_h

    def deployment(self, local_map, coord: np.array, theta: np.array, ele):
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
                [self.min_l, self.min_h / 2],
                [self.min_l, -self.min_h / 2],
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

    @classmethod
    def create(cls, HFOV, VFOV, phi, H, fy, delta_v=1):
        # geometry
        assert 0 <= phi <= np.pi / 2, "pitch angle is wrong"
        Dmin = H * np.tan(phi - VFOV / 2)
        L_HE = H / np.cos(phi - VFOV / 2)
        Hmin = 2 * L_HE * np.cos(VFOV / 2) * np.tan(HFOV / 2)

        if phi + VFOV / 2 >= np.pi / 2:
            Dmax, L_HF = MAX_VALUE, MAX_VALUE
        else:
            Dmax = min(H * np.tan(phi + VFOV / 2), MAX_VALUE)
            L_HF = min(H / np.cos(phi + VFOV / 2), MAX_VALUE)
        Hmax = min(2 * L_HF * np.cos(VFOV / 2) * np.tan(HFOV / 2), MAX_VALUE)

        # localization
        D_thresh = 1 - 2 * H / np.tan(phi) + \
                   np.sqrt(
                       1 + (4 * H * fy) / (delta_v * np.sin(phi) ** 2)
                   )

        if D_thresh <= max(Dmin, 0):
            return None

        Dmaxmin = min(D_thresh, Dmax)
        if Dmaxmin < Dmax:
            Hmax = Hmax - (Dmax - Dmaxmin) / (Dmax - Dmin) * (Hmax - Hmin)

        print(Hmax)
        return cls(Dmin, Dmaxmin, Hmin, Hmax)
