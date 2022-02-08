import cv2
import numpy as np
from configparser import ConfigParser
from shapely.geometry import Point, Polygon
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


MAX_VALUE = 1e6


class Camera1D:
    def __init__(self, min_l, max_l, min_h, max_h):
        self.min_l = min_l
        self.max_l = max_l
        self.min_h = min_h
        self.max_h = max_h

    def deployment(self, local_map, coord: np.array, theta, ele):
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
    def create(cls, HFOV, VFOV, phi, H, fx, fy, delta_u=1, delta_v=1, cx=1., cy=1.):
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
        if phi == 0:
            D_thresh = MAX_VALUE
        else:
            D_thresh_lon = 0.5 * (-cy - 2 * H / np.tan(phi) +
                              np.sqrt(
                                  cy ** 2 + (4 * cy * H * fy) / (delta_v * np.sin(phi) ** 2)
                              ))
            D_thresh_lat = (cx * fx / delta_u - H * np.cos(phi)) / np.sin(phi)
            D_thresh = min(D_thresh_lat, D_thresh_lon)

        if D_thresh <= max(Dmin, 0):
            print("cur sensor is", phi * 180 / np.pi, H)
            return None

        Dmaxmin = min(D_thresh, Dmax)
        if Dmaxmin < Dmax:
            Hmax = Hmax - (Dmax - Dmaxmin) / (Dmax - Dmin) * (Hmax - Hmin)

        return cls(Dmin, Dmaxmin, Hmin, Hmax)

    def is_point_vis(self, point, coord, theta):
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
                [self.max_l, self.max_h / 2],
                [self.min_l, self.min_h / 2]
            ]
        )
        corner_pts = corner_pts.T
        local_coord = np.dot(R, corner_pts) + t

        pt = Point(point.x, point.y)
        poly = Polygon(local_coord.T.tolist())

        return pt.within(poly)


if __name__ == '__main__':
    rad = np.pi / 180
    HFOV = 30 * rad
    VFOV = 23 * rad
    fx, fy = 2329.297332, 2329.297332

    phis = np.arange(0, np.pi / 2, 5 * rad)
    Hs = np.arange(3, 7, 0.5)

    phis, Hs = np.meshgrid(phis, Hs)

    Max_Ds = np.zeros(shape=phis.shape)
    Min_Ds = np.zeros(shape=phis.shape)
    for i in range(phis.shape[0]):
        for j in range(phis.shape[1]):
            sensor = Camera1D.create(HFOV, VFOV, phis[i, j], Hs[i, j], fx, fy, cx=0.1, cy=0.7)

            Max_Ds[i, j] = sensor.max_l
            Min_Ds[i, j] = sensor.min_l

            # print(sensor.min_l, sensor.max_l, sensor.min_h, sensor.max_h)

    fig = plt.figure(figsize=(12, 8))
    ax1 = Axes3D(fig)
    ax1.plot_surface(phis / rad, Hs, Max_Ds, cmap='rainbow')
    ax1.set_xlabel(r"$\phi$")
    ax1.set_ylabel('H/m')
    ax1.set_zlabel('Max field/m')
    plt.savefig("result/max_l.png")

    fig = plt.figure(figsize=(12, 8))
    ax2 = Axes3D(fig)
    ax2.plot_surface(phis / rad, Hs, Min_Ds, cmap='rainbow')
    ax2.set_xlabel(r"$\phi$")
    ax2.set_ylabel('H/m')
    ax2.set_zlabel('Min field/m')

    plt.savefig("result/min_l.png")
    plt.show()
