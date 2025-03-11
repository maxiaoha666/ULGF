colors = [
    (255, 0, 0),   # 红色
    (0, 255, 0),   # 绿色
    (0, 0, 255),   # 蓝色
    (255, 255, 0), # 黄色
    (255, 0, 255), # 紫色
    (0, 255, 255), # 青色
    (128, 128, 0), # 橄榄绿
    (128, 0, 128), # 深紫色
    (0, 128, 128), # 深青色
    (255, 165, 0)  # 橙色
]
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# 创建一个空白的白色图像
image_height = 500  # 图像高度
image_width = 500   # 图像宽度
image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255  # 白色背景

# 显示空白图像
plt.imshow(image)
plt.axis('off')

def read_yolo_labels(label_file):
    """
    读取YOLO标签文件，返回类别编号和边界框信息
    """
    boxes = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            class_id = int(data[0])  # 类别编号
            x_center = float(data[1])  # 归一化的中心点 x 坐标
            y_center = float(data[2])  # 归一化的中心点 y 坐标
            width = float(data[3])  # 归一化的宽度
            height = float(data[4])  # 归一化的高度
            boxes.append((class_id, x_center, y_center, width, height))
    return boxes

def visualize_labels_on_blank_image(label_file, image_width, image_height):
    """
    在空白图像上绘制YOLO标签框
    """
    # 读取标签文件
    boxes = read_yolo_labels(label_file)

    # 创建空白白色图像
    image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255

    # 获取图像的宽高
    for (class_id, x_center, y_center, width, height) in boxes:
        # 反归一化坐标
        x_min = int((x_center - width / 2) * image_width)
        y_min = int((y_center - height / 2) * image_height)
        x_max = int((x_center + width / 2) * image_width)
        y_max = int((y_center + height / 2) * image_height)

        # 获取颜色
        color = colors[class_id % len(colors)]  # 使用类别编号来选择颜色

        # 绘制矩形框
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=-1)

    # 显示结果
    plt.imshow(image)
    plt.axis('off')
    plt.show()
def get_all_files_in_directory(directory):
    # 使用Path对象遍历文件夹并获取所有文件的绝对路径
    directory_path = Path(directory)
    file_paths = [str(file) for file in directory_path.rglob('*') if file.is_file()]
    return file_paths
# 调用函数可视化标签
file_paths = get_all_files_in_directory('/mnt/data0/mly/RUOD/ruod_expand/labels/绘图标签')

# 打印所有文件的绝对路径
for path in file_paths:
    label_file = path # 这里替换为你的标签文件路径
    visualize_labels_on_blank_image(label_file, 500, 500)  # 500x500的图像
    print(path)
    print(label_file)