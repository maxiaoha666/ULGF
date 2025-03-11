import os
import json
from glob import glob
from pathlib import Path
import cv2


def yolo_to_coco(yolo_dir, image_dir, categories):
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": cat} for i, cat in enumerate(categories)],
    }

    image_id = 0
    annotation_id = 0

    # 遍历YOLO标签文件
    yolo_files = glob(os.path.join(yolo_dir, "*.txt"))
    for yolo_file in yolo_files:
        # 获取图片文件名
        image_file = os.path.join(image_dir, Path(yolo_file).stem + ".jpg")
        image = cv2.imread(image_file)
        h, w, _ = image.shape

        coco_dict["images"].append({
            "id": image_id,
            "file_name": Path(image_file).name,
            "width": w,
            "height": h
        })

        # 读取YOLO标签
        with open(yolo_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                center_x, center_y, bbox_width, bbox_height = map(float, parts[1:])

                # 转换为COCO格式的bbox
                x_min = round((center_x - bbox_width / 2) * w)
                y_min = round((center_y - bbox_height / 2) * h)
                bbox_width = round(bbox_width * w)
                bbox_height = round(bbox_height * h)

                coco_dict["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1

    # 输出为JSON文件
    with open("/mnt/data0/mly/RUOD/ruod_expand/ruod_annotations/train_new/instances_train.json", 'w') as f:
        json.dump(coco_dict, f, indent=4)


# 示例调用
yolo_dir = '/mnt/data0/mly/RUOD/ruod_expand/labelsraw'  # YOLO标签文件夹
image_dir = '/mnt/data0/mly/RUOD/ruod_expand/images-raw/train'  # 图片文件夹
categories = ["holothurian","echinus","scallop","starfish","fish","corals","diver","cuttlefish","turtle","jellyfish"]  # 类别列表
yolo_to_coco(yolo_dir, image_dir, categories)
