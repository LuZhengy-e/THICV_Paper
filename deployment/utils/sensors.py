import cv2
import numpy as np
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

    def deployment(self, local_map, coord: np.array, theta, ele, **kwargs):
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
        if kwargs is not None:
            local_map.update_line_tag(line_id, tags=kwargs)

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

    def get_lon_error(self, t, phi, H, fy, delta_v=1):
        a = delta_v / fy * (H * np.cos(phi) + t * np.sin(phi))
        b = H / (H * np.cos(phi) + t * np.sin(phi)) + delta_v / fy * np.sin(phi)

        return abs(a / b)

    def get_lat_error(self, t, phi, H, fx, delta_u=1):
        a = delta_u / fx * (H * np.cos(phi) + t * np.sin(phi))

        return abs(a)


class CameraGroups:
    def __init__(self, min_l, max_l):
        self.groups = []
        self.min_l = min_l
        self.max_l = max_l

    def create_groups(self, HFOV, VFOV, max_phi, max_H, fx, fy, delta_u=1, delta_v=1, cx=1., cy=1.):
        cameras_info = []
        min_l, max_l = self.min_l, self.max_l
        for i in range(0, 95, 5):
            phi = i * rad
            if phi > max_phi:
                break

            for h in range(3, max_H):
                sensor = Camera1D.create(HFOV, VFOV, phi, h, fx, fy, delta_u, delta_v, cx, cy)
                cameras_info.append(
                    {
                        "H": h,
                        "phi": phi,
                        "min_l": sensor.min_l,
                        "max_l": sensor.max_l
                    }
                )

        min_l = MAX_VALUE
        while min_l > self.min_l:
            cur_cameras = sorted(cameras_info,
                                 key=lambda x: abs(x["max_l"] - max_l))
            cur_camera = cur_cameras.pop(0)
            max_l = cur_camera["min_l"]
            min_l = max_l

            self.groups.append(
                {
                    "H": cur_camera["H"],
                    "phi": cur_camera["phi"]
                }
            )

        assert len(self.groups) > 0, "Can't create groups"

    def plot_project(self, **kwargs):
        fig = plt.figure()
        if len(self.groups) == 0:
            raise NotImplementedError("Please create first")

        for info in self.groups:
            sensor = Camera1D.create(H=info["H"], phi=info["phi"], **kwargs)
            print(sensor.min_h)

            plt.plot(
                [sensor.min_l, sensor.min_l, sensor.max_l, sensor.max_l, sensor.min_l],
                [sensor.min_h / 2, -sensor.min_h / 2, -sensor.max_h / 2, sensor.max_h / 2, sensor.min_h / 2]
            )

        plt.xlabel("lon / m")
        plt.ylabel("lat / m")
        plt.show()


class Camera2D:
    def __init__(self, min_l, max_l, min_h, max_h):
        self.min_l = min_l
        self.max_l = max_l
        self.min_h = min_h
        self.max_h = max_h

    def deployment(self, local_map, coord: np.array, theta, ele, **kwargs):
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
        if kwargs is not None:
            local_map.update_line_tag(line_id, tags=kwargs)

    def calculate_prob(self, road_pos, coord, theta, phi, H, k):
        direct = np.array([np.cos(theta), np.sin(theta)])
        vec = np.array([road_pos.x - coord.x, road_pos.y - coord.y])

        project = np.dot(vec, direct)
        assert self.min_l < project < self.max_l, "This point not in camera"
        z = H * np.cos(phi) + project * np.sin(phi)
        prob = 2 / np.pi * np.arctan(k / z)

        return prob

    @classmethod
    def create(cls, HFOV, VFOV, phi, H, fx, k, delta_x, prob_thresh=0.25):
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

        # perception
        if phi == 0:
            D_thresh = MAX_VALUE

        else:
            D_thresh = ((k * fx * delta_x) / np.tan(prob_thresh * np.pi / 2) - H * np.cos(phi)) / np.sin(phi)

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
    HFOV = 25 * rad
    VFOV = 23 * rad
    fx, fy = 2183.375019, 2329.297332
    delta_x = 2.5

    # phis = np.arange(0, np.pi / 2, 5 * rad)
    # Hs = np.arange(3, 7, 0.5)
    #
    # phis, Hs = np.meshgrid(phis, Hs)
    #
    # Max_Ds = np.zeros(shape=phis.shape)
    # Min_Ds = np.zeros(shape=phis.shape)
    # for i in range(phis.shape[0]):
    #     for j in range(phis.shape[1]):
    #         sensor = Camera1D.create(HFOV, VFOV, phis[i, j], Hs[i, j], fx, fy, delta_u=2, delta_v=2, cx=np.sqrt(0.7), cy=1)
    #
    #         Max_Ds[i, j] = sensor.max_l
    #         Min_Ds[i, j] = sensor.min_l
    #
    #         print(sensor.min_l, sensor.max_l, sensor.min_h, sensor.max_h)
    #
    # fig = plt.figure(figsize=(12, 8))
    # ax1 = Axes3D(fig)
    # ax1.plot_surface(phis / rad, Hs, Max_Ds, cmap='rainbow')
    # ax1.set_xlabel(r"$\phi$")
    # ax1.set_ylabel('H/m')
    # ax1.set_zlabel('Max field/m')
    # plt.savefig("result/max_l.png")
    #
    # fig = plt.figure(figsize=(12, 8))
    # ax2 = Axes3D(fig)
    # ax2.plot_surface(phis / rad, Hs, Min_Ds, cmap='rainbow')
    # ax2.set_xlabel(r"$\phi$")
    # ax2.set_ylabel('H/m')
    # ax2.set_zlabel('Min field/m')
    #
    # plt.savefig("result/min_l.png")
    # plt.show()

    # phi = 11.5 * rad
    # H = 6
    # delta_u = 1
    # delta_v = 1
    # sensor = Camera1D.create(HFOV, VFOV, phi, H, fx, fy, cx=0.1, cy=0.7, delta_u=delta_u, delta_v=delta_v)
    #
    # arange = np.linspace(sensor.min_l, sensor.max_l, 50)
    # arange = arange.tolist()
    #
    # lat, lon = [], []
    # for t in arange:
    #     lat.append(sensor.get_lat_error(t, phi, H, fx, delta_u))
    #     lon.append(sensor.get_lon_error(t, phi, H, fy, -delta_v))
    #
    # fig = plt.figure(figsize=(20, 8))
    # plt.subplot(121)
    # plt.plot(arange, lat)
    #
    # plt.subplot(/home/luzhengye/Code/LeetCode122)
    # plt.plot(arange, lon)
    #
    # plt.show()

    groups = CameraGroups(0, 100)
    groups.create_groups(HFOV, VFOV, np.pi / 2, 8, fx, fy, cx=0.1, cy=0.7)
    groups.plot_project(HFOV=HFOV, VFOV=VFOV, fx=fx, fy=fy, cx=0.1, cy=0.7)

    fig = plt.figure(figsize=(20, 8))
    for camera in groups.groups:
        sensor = Camera1D.create(HFOV, VFOV, camera["phi"], camera["H"], fx, fy, cx=0.8, cy=1)

        arange = np.linspace(sensor.min_l, sensor.max_l, 50)
        arange = arange.tolist()

        lat, lon = [], []
        for t in arange:
            lat.append(sensor.get_lat_error(t, camera["phi"], camera["H"], fx, 1))
            lon.append(sensor.get_lon_error(t, camera["phi"], camera["H"], fy, -1))

        plt.subplot(121)
        plt.xlabel("distance / m")
        plt.ylabel("lat error / m")
        plt.plot(arange, lat)

        plt.subplot(122)
        plt.xlabel("distance / m")
        plt.ylabel("lon error / m")
        plt.plot(arange, lon)

    plt.show()
