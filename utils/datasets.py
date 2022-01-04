import os
import cv2
import json
import open3d as o3d
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from configparser import ConfigParser


class AirDataset:
    def __init__(self, cfg: ConfigParser):
        self.cfg = cfg
        self.root_path = cfg.get("DATASET", "root")
        self.data_info = os.path.join(self.root_path, cfg.get("DATASET", "data_info"))

    def _vis_image(self, image_path, label_path):
        img = cv2.imread(image_path)
        if img is None:
            raise IOError("Image path does not exists")

        with open(label_path, "r") as f:
            label = json.load(f)

        objects = self.cfg.get("DATASET", "objects").split(",")
        vis_image = img.copy()
        for object in label:
            if object["type"] not in objects:
                continue

            bbox = object["2d_box"]
            leftTop = (int(float(bbox["xmin"])), int(float(bbox["ymin"])))
            rightBottom = (int(float(bbox["xmax"])), int(float(bbox["ymax"])))

            vis_image = cv2.rectangle(vis_image, leftTop, rightBottom, (0, 255, 0))
        return vis_image

    def _vis_points(self, lidar_path, label_path):
        pcd = o3d.io.read_point_cloud(lidar_path)

        return pcd

    def _project_pts(self, image_path, lidar_path, calib_path, lidar_calib):
        with open(calib_path, "r") as f:
            inner = json.load(f)

        if inner["calibration_result_flag"] != "Success!":
            return None

        K = np.array(inner["K"])
        K = np.expand_dims(K, axis=0).reshape((3, 3))

        with open(lidar_calib, "r") as f:
            outer = json.load(f)

        Rcl = np.array(outer["R_kitti2cam"])
        tcl = np.array(outer["T_kitti2cam"])

        pcd = o3d.io.read_point_cloud(lidar_path)
        # o3d.visualization.draw_geometries([pcd])
        pts = np.array(pcd.points)

        trans_pts = np.dot(Rcl, pts.T) + tcl
        norm_pts = trans_pts / trans_pts[2, :]

        u = np.dot(K, norm_pts)[0:2, :].T

        image = cv2.imread(image_path)
        if image is None:
            raise IOError("Image path does not exist")

        h, w = image.shape[0:2]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        colors, new_pts = [], []
        for coord, pt in zip(u, pts):
            if coord[0] < 0 or coord[0] >= w or coord[1] < 0 or coord[1] >= h:
                continue

            colors.append((image[int(coord[1]), int(coord[0])] / 255).tolist())
            new_pts.append(pt.tolist())

        colors = o3d.utility.Vector3dVector(colors)
        points = o3d.utility.Vector3dVector(new_pts)

        pcd.points = points
        pcd.colors = colors

        # checkout
        # o3d.visualization.draw_geometries([pcd])

        return pcd

    def _statistic_position(self, image_path, lidar_path, calib_path, lidar_calib, label_path):
        with open(calib_path, "r") as f:
            inner = json.load(f)

        if inner["calibration_result_flag"] != "Success!":
            return None

        K = np.array(inner["K"])
        K = np.expand_dims(K, axis=0).reshape((3, 3))

        with open(lidar_calib, "r") as f:
            outer = json.load(f)

        Rcl = np.array(outer["R_kitti2cam"])
        tcl = np.array(outer["T_kitti2cam"])

        pcd = o3d.io.read_point_cloud(lidar_path)

    def __call__(self, **kwargs):
        with open(self.data_info, "r") as f:
            data_info = json.load(f)

        for info in data_info:
            process = {}

            image_path = os.path.join(self.root_path, info["image_path"])
            pcd_path = os.path.join(self.root_path, info["point_cloud_path"])
            calib_cw = os.path.join(self.root_path, info["calib_camera_intrisinc_path"])
            calib_cl = os.path.join(self.root_path, info["calib_virtuallidar_to_camera_path"])
            camera_label = os.path.join(self.root_path, info["label_camera_std_path"])
            lidar_label = os.path.join(self.root_path, info["label_lidar_std_path"])

            process["pcd_name"] = info["point_cloud_path"].split("/")[-1]

            if kwargs.get("vis_image") is True:
                process["image"] = self._vis_image(image_path, camera_label)

            if kwargs.get("vis_points") is True:
                process["points"] = self._vis_points(pcd_path, lidar_label)

            if kwargs.get("vis_project") is True:
                process["color_pts"] = self._project_pts(image_path,
                                                         pcd_path,
                                                         calib_cw,
                                                         calib_cl)

            yield process


if __name__ == '__main__':
    cfg = ConfigParser()
    cfg.read("config/config.cfg")

    data_handler = AirDataset(cfg)
    for image_info in data_handler(vis_image=False,
                                   vis_points=True,
                                   vis_project=True):
        if image_info.get("points") is not None:
            pass

        if image_info.get("color_pts") is not None:
            output_dir = cfg.get("DATASET", "lidar_output")
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            print(image_info["pcd_name"])
            o3d.io.write_point_cloud(os.path.join(output_dir, image_info["pcd_name"]),
                                     image_info["color_pts"])

        if image_info.get("image") is not None:
            cv2.imshow("image", image_info["image"])
            cv2.waitKey(0)
