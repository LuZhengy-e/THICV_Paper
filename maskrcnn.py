import cv2
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.optimize import curve_fit
from configparser import ConfigParser
from collections import Counter

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util

from deployment.utils.datasets import AirDataset
from maskrcnn_benchmark.config import cfg


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image


class COCODemo(object):
    # COCO categories for pretty print
    CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(
            self,
            cfg,
            confidence_threshold=0.7,
            show_mask_heatmaps=False,
            masks_per_dim=2,
            min_image_size=224,
            weight_loading=None
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        if weight_loading:
            print('Loading weight from {}.'.format(weight_loading))
            _ = checkpointer._load_model(torch.load(weight_loading))

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        if self.cfg.MODEL.KEYPOINT_ON:
            result = self.overlay_keypoints(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)

        return result

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None].astype(np.uint8)
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite

    def overlay_keypoints(self, image, predictions):
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        scores = keypoints.get_field("logits")
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
        for region in kps:
            image = vis_keypoints(image, region.transpose((1, 0)))
        return image

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image

    def evaluate_bbox(self, image, gt_bboxes, xs, ys, iou_thresh=0.85):
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        labels = top_predictions.get_field("labels")
        bboxes = top_predictions.bbox

        num_pred, num_gt = len(bboxes), len(gt_bboxes)
        if num_gt == 0:
            return 0, 0

        num_car = 0
        used = [False] * num_gt
        matched = 0
        offset = [0, 0]
        for i in range(num_pred):
            if labels[i] != 3:
                continue

            num_car += 1

            x_min, y_min, x_max, y_max = bboxes[i]
            max_iou = -1
            max_j = -1
            min_off = None
            for j in range(num_gt):
                if used[j] is True:
                    continue

                gt_bbox = gt_bboxes[j]
                gt_x_min, gt_y_min, gt_x_max, gt_y_max = \
                    float(gt_bbox["xmin"]), float(gt_bbox["ymin"]), float(gt_bbox["xmax"]), float(gt_bbox["ymax"])

                left_top_x, left_top_y = max(gt_x_min, x_min), max(gt_y_min, y_min)
                right_bottom_x, right_bottom_y = min(gt_x_max, x_max), min(gt_y_max, y_max)

                intersection = max(0, right_bottom_x - left_top_x) * max(0, right_bottom_y - left_top_y)
                union = (gt_x_max - gt_x_min) * (gt_y_max - gt_y_min) + \
                        (x_max - x_min) * (y_max - y_min) - intersection

                iou = intersection / union
                if iou < iou_thresh:
                    continue

                if iou > max_iou:
                    max_iou = iou
                    max_j = j
                    min_off = (abs((gt_x_max + gt_x_min) / 2 - (x_max + x_min) / 2),
                               abs(gt_y_min - y_min))

            if max_j >= 0:
                matched += 1
                used[max_j] = True
                offset[0] += min_off[0]
                offset[1] += min_off[1]

        precision = matched / (num_car + 1e-6)
        recall = matched / num_gt

        x_off = offset[0] / (num_car + 1e-6)
        y_off = offset[1] / (num_car + 1e-6)
        xs.append(x_off)
        ys.append(y_off)

        plt.scatter(x_off, y_off, c="black", s=3)

        return precision, recall

    def evaluate_recall(self, image, gt_bboxes, gt_dist, recalls: list, dist_thresh, scale=1., iou_thresh=0.75):
        assert len(gt_bboxes) == len(gt_dist)
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        labels = top_predictions.get_field("labels")
        bboxes = top_predictions.bbox

        num_pred, num_gt = len(bboxes), len(gt_bboxes)
        if num_gt == 0:
            return

        used = [False] * num_gt
        matched = 0

        for i in range(num_pred):
            if labels[i] != 3:
                continue

            x_min, y_min, x_max, y_max = bboxes[i]
            max_iou = -1
            max_j = -1

            for j in range(num_gt):
                if used[j]:
                    continue

                gt_bbox = gt_bboxes[j]
                gt_x_min, gt_y_min, gt_x_max, gt_y_max = \
                    float(gt_bbox["xmin"]) // scale, float(gt_bbox["ymin"]) // scale, \
                    float(gt_bbox["xmax"]) // scale, float(gt_bbox["ymax"]) // scale

                left_top_x, left_top_y = max(gt_x_min, x_min), max(gt_y_min, y_min)
                right_bottom_x, right_bottom_y = min(gt_x_max, x_max), min(gt_y_max, y_max)

                intersection = max(0, right_bottom_x - left_top_x) * max(0, right_bottom_y - left_top_y)
                union = (gt_x_max - gt_x_min) * (gt_y_max - gt_y_min) + \
                        (x_max - x_min) * (y_max - y_min) - intersection

                iou = intersection / union
                if iou < iou_thresh:
                    continue

                if iou > max_iou:
                    max_iou = iou
                    max_j = j

            if max_j >= 0:
                used[max_j] = True
                if abs(gt_dist[max_j][0]) < dist_thresh and 25 < gt_dist[max_j][2] < 125:
                    matched += 1
                    gt_bbox = gt_bboxes[max_j]

                    recalls.append((float(gt_bbox["xmax"]) - float(gt_bbox["xmin"])) / scale)


if __name__ == '__main__':
    config_file = "config/e2e_faster_rcnn_R_101_FPN_1x.yaml"
    # config_file = "config/e2e_faster_rcnn_R_50_C4_1x.yaml"

    dist_thresh = 7.5

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    coco_demo = COCODemo(
        cfg,
        min_image_size=1080,
        confidence_threshold=0.75,
    )

    config = ConfigParser()
    config.read("config/config.cfg")

    dataset = AirDataset(config)

    recalls, gts = [], []
    xs, ys = [], []
    num = 0
    for data in tqdm(dataset(vis_image=True, bbox=True)):
        num += 1
        if num > 0 * 1000:
            break
        gt = data["bbox"]
        img = gt["img"].copy()
        # for scale in range(1, 2):
        #     img = cv2.pyrUp(img)
        #     coco_demo.evaluate_recall(img, gt["bbox"], gt["dist"], recalls, dist_thresh, scale=0.5 ** scale)
        #     gts.extend([(float(gt_bbox["xmax"]) - float(gt_bbox["xmin"])) / (0.5 ** scale)
        #                 for gt_bbox, gt_dist in zip(gt["bbox"], gt["dist"]) if abs(gt_dist[0]) < dist_thresh
        #                 and 25 < gt_dist[2] < 125])
        #
        # img = gt["img"].copy()
        # for scale in range(4):
        #     if scale != 0:
        #         img = cv2.pyrDown(img)
        #     coco_demo.evaluate_recall(img, gt["bbox"], gt["dist"], recalls, dist_thresh, scale=2 ** scale)
        #     gts.extend([(float(gt_bbox["xmax"]) - float(gt_bbox["xmin"])) / (2 ** scale)
        #                 for gt_bbox, gt_dist in zip(gt["bbox"], gt["dist"]) if abs(gt_dist[0]) < dist_thresh
        #                 and 25 < gt_dist[2] < 125])

        precision, recall = coco_demo.evaluate_bbox(img, gt["bbox"], xs, ys)
        #     prediction = coco_demo.run_on_opencv_image(img)
            # cv2.imshow("pred", prediction)
        # cv2.imshow("gt", data["image"])
        #
            # cv2.waitKey(0)

        # print(precision, recall)

    # plt.xlabel("u offset")
    # plt.ylabel("v offset")
    # plt.figure()
    # x, y = Counter(xs), Counter(ys)
    # xs, ys = sorted(x.keys()), sorted(y.keys())
    # time_x, time_y = [x[k] for k in xs], [y[k] for k in ys]
    # acc_x, acc_y = np.cumsum(time_x), np.cumsum(time_y)
    # plt.plot(xs, (acc_x / acc_x[-1]).tolist(), label="x offset")
    # plt.plot(ys, (acc_y / acc_x[-1]).tolist(), label="y offset")
    # plt.xlabel("offset")
    # plt.ylabel("CDF")
    # plt.legend()
    # plt.show()
    # plt.savefig("result/offset.png")
    # gts = sorted(gts)
    # recalls = sorted(recalls)
    #
    with open("result/faster_r_101.json", "r") as f:
        # json.dump({"gt": gts, "pred": recalls}, f)
        res = json.load(f)

    gts, recalls = res["gt"], res["pred"]

    p1 = plt.hist(gts, bins=32, range=[0, 128], label="groundtruth")
    p2 = plt.hist(recalls, bins=32, range=[0, 128], label="predict")
    plt.xlabel("width")
    plt.ylabel("number")
    plt.legend()

    plt.savefig("result/hist_c4.pdf")

    section = np.linspace(0, 128, 32).tolist()
    cuts = pd.cut(np.array(gts), section)
    counts = pd.value_counts(cuts)

    cuts_ = pd.cut(np.array(recalls), section)
    counts_ = pd.value_counts(cuts_)

    x, r = [], []
    keys = list(counts.keys())
    keys = sorted(keys, key=lambda x: x.mid)
    for k in keys:
        x.append(k.mid)
        r.append(counts_[k] / (counts[k] + 1e-6))

    r_ = np.array(r)

    K = curve_fit(lambda _x, _k: _k * _x, x, np.tan(r_ * np.pi / 2))[0]
    fit = [2 / np.pi * np.arctan(i * K) for i in range(128)]
    print(K)

    plt.figure()
    p1 = plt.plot(x, r, label="origin")
    p2 = plt.plot(list(range(128)), fit, label="fit")
    plt.xlabel("width")
    plt.ylabel("recall")
    plt.legend()

    plt.savefig("result/curve_c4.pdf")

    plt.show()

    # draw heatmap
