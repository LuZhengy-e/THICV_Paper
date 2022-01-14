import cv2
import math
import numpy as np

MAX_ITERATION = 1e6


class Point3D:
    @staticmethod
    def RANSAC(pts, threshold, iterations, shape="plane"):
        assert iterations > 0, "Please input reasonable iterations"
        if shape == "plane":
            return Point3D._RANSAC_Plane(pts, threshold, iterations)

        else:
            return None

    @staticmethod
    def _RANSAC_Plane(pts, threshold, iterations):
        pts_4D = np.concatenate((pts, np.ones(shape=(pts.shape[0], 1))), axis=1)

        it = 0
        max_num = -1
        best_norm = None
        results = None
        while it < iterations:
            it += 1
            sample_list = list(range(pts_4D.shape[0]))
            sample_idx = np.random.choice(sample_list, 3)
            sample_pts = pts_4D[sample_idx, :]

            sample_pts_3D = sample_pts[:, 0:3]
            norm = np.cross(sample_pts_3D[1] - sample_pts_3D[0], sample_pts_3D[2] - sample_pts_3D[0])
            norm /= np.linalg.norm(norm)
            d = -np.dot(sample_pts_3D, norm)
            norm = np.concatenate((norm, d[[0]]))

            dist = abs(np.dot(pts_4D, norm))

            inner_pts = pts_4D[np.where(dist < threshold)]
            num = len(inner_pts)
            if num > max_num:
                max_num = num
                best_norm = norm.copy()
                results = inner_pts.copy()

        return best_norm, results[:, 0:3]


class Geometry:
    @staticmethod
    def inv_project(pixel: np.array,  # pixel coord
                    norm: np.array,  # normal vector of plane
                    K: np.array):
        if len(pixel) == 2:
            pixel = np.concatenate((pixel, [1]))

        n, d = norm[0:3], -norm[3]
        inv_depth = d / np.dot(n.T, np.dot(np.linalg.inv(K), pixel))
        Pc = inv_depth * np.dot(np.linalg.inv(K), pixel)

        return Pc.copy()

    @staticmethod
    def project_pts(pts: np.array,
                    K: np.array):
        if len(pts.shape) == 1:
            pts = np.expand_dims(pts, axis=1)

        norm_pts = pts / pts[2, :]
        u = np.dot(K, norm_pts)[0:2, :]

        return u.copy()

    @staticmethod
    def lla2ecef(lat, lon):
        lat, lon = np.pi * lat / 180.0, np.pi * lon / 180.0
        f = 1 / 298.257223565
        R = 6378137.0
        N = R / np.sqrt(1 - f * (2 - f) * np.sin(lat) ** 2)
        X = N * np.cos(lat) * np.cos(lon)
        Y = N * np.cos(lat) * np.sin(lon)
        Z = N * (1 - f) ** 2 * np.sin(lat)

        return X, Y, Z

    @staticmethod
    def ecef2enu(X, Y, Z, X0, Y0, Z0, lat0, lon0):
        lat0, lon0 = np.pi * lat0 / 180.0, np.pi * lon0 / 180.0
        pos = np.array([X, Y, Z])
        ref = np.array([X0, Y0, Z0])
        delta = pos - ref
        S = np.array([[-np.sin(lon0), np.cos(lon0), 0],
                      [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
                      [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)]])
        enu = np.dot(S, delta)
        return enu[0], enu[1], enu[2]

    @staticmethod
    def lla2enu(lat, lon, lat0, lon0):
        X, Y, Z = Geometry.lla2ecef(lat, lon)
        X0, Y0, Z0 = Geometry.lla2ecef(lat0, lon0)

        return Geometry.ecef2enu(X, Y, Z, X0, Y0, Z0, lat0, lon0)

    @staticmethod
    def enu2ecef(x, y, z, lat0, lon0):
        X0, Y0, Z0 = Geometry.lla2ecef(lat0, lon0)
        lat0, lon0 = np.pi * lat0 / 180.0, np.pi * lon0 / 180.0

        S = np.array([[-np.sin(lon0), np.cos(lon0), 0],
                      [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
                      [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)]])
        enu = np.array([x, y, z])

        ecef = np.dot(S.T, enu) + np.array([X0, Y0, Z0])

        return ecef[0], ecef[1], ecef[2]

    @staticmethod
    def ecef2lla(x, y, z):
        f = 1 / 298.257223565
        R = 6378137.0
        e = f * (2 - f)

        lon = np.arctan(y / x) * 180 / np.pi
        if lon < 0:
            lon = 180 + lon

        # calculate lat
        lat = 0
        N = R / np.sqrt(1 - f * (2 - f) * np.sin(lat) ** 2)
        for _ in range(10):
            lat = np.arcsin(z / (N * (1 - f) ** 2))
            N = R / np.sqrt(1 - f * (2 - f) * np.sin(lat) ** 2)

        return lat * 180 / np.pi, lon

    @staticmethod
    def enu2lla(x, y, z, lat0, lon0):
        X, Y, Z = Geometry.enu2ecef(x, y, z, lat0, lon0)
        return Geometry.ecef2lla(X, Y, Z)


if __name__ == '__main__':
    lat0, lon0 = 39.9953100, 116.3152300
    lat, lon = 40.0023170, 116.3244420
    print(lat, lon)

    x, y, z = Geometry.lla2ecef(lat, lon)
    x0, y0, z0 = Geometry.lla2ecef(lat0, lon0)
    print(x, y, z)

    e, n, u = Geometry.ecef2enu(x, y, z, x0, y0, z0, lat0, lon0)
    print(e, n, u)

    lat_, lon_ = Geometry.enu2lla(e, n, u, lat0, lon0)
    print(lat_, lon_)
