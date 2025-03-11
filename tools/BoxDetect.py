import os
import numpy as np
import cv2
import json

def draw_yolo_box(img, class_id, x_center, y_center, w, h, color=(0, 0, 255), thickness=2):
    """ 使用YOLO格式的边界框参数在图片上绘制边界框 """
    x = int((x_center - w / 2) * img.shape[1])
    y = int((y_center - h / 2) * img.shape[0])
    w = int(w * img.shape[1])
    h = int(h * img.shape[0])
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)


def load_and_annotate_image(img_path, label_path, extra_labels):
    """ 加载图片，读取YOLO格式的标签，进行标注，并返回标注后的图片 """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")
        return None

        # 读取YOLO格式的标签文件
    with open(label_path, 'r') as f:
        for line in f.readlines():
            class_id, x_center, y_center, w, h = map(float, line.strip().split())
            draw_yolo_box(img, class_id, x_center, y_center, w, h)

            # 使用额外的标签进行标注（这里简单地在图片上打印文本）
    for label in extra_labels:
        class_id, x_center, y_center, w, h = label
        # 假设我们只在图片的一个固定位置打印所有额外标签
        # 实际应用中，你可能需要为每个标签指定不同的位置
        draw_yolo_box(img, class_id, x_center, y_center, w, h, color=(255, 0, 0))

    return img


def read_boxes_from_txt(file_path):
    """
    读取txt文件中的边界框信息。
    假设每行格式为：类别ID x_center y_center width height
    其中各个值由空格分隔。

    :param file_path: txt文件的路径
    :return: 边界框列表，每个边界框是一个包含(类别ID, x_center, y_center, width, height)的元组
    """
    boxes = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 去除行尾的换行符，并按空格分割字符串
                line_parts = line.strip().split()
                # 假设前五个元素分别是类别ID、x_center、y_center、width、height
                if len(line_parts) >= 5:
                    class_id = int(line_parts[0])  # 假设类别ID是整数
                    x_center = float(line_parts[1])
                    y_center = float(line_parts[2])
                    width = float(line_parts[3])
                    height = float(line_parts[4])
                    # 将解析出的数据添加到边界框列表中
                    boxes.append((class_id, x_center, y_center, width, height))
        return boxes
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return []
    except ValueError:
        print("Error: Unable to parse data from the file.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def calculate_iou(box1, box2):
    cls_1, x_center1, y_center1, width1, height1 = box1
    cls_2, x_center2, y_center2, width2, height2 = box2

    b1_x1 = x_center1 - width1 / 2
    b1_y1 = y_center1 - height1 / 2
    b1_x2 = x_center1 + width1 / 2
    b1_y2 = y_center1 + height1 / 2
    b2_x1 = x_center2 - width2 / 2
    b2_y1 = y_center2 - height2 / 2
    b2_x2 = x_center2 + width2 / 2
    b2_y2 = y_center2 + height2 / 2

    xx1 = np.maximum(b1_x1, b2_x1)
    yy1 = np.maximum(b1_y1, b2_y1)
    xx2 = np.minimum(b1_x2, b2_x2)
    yy2 = np.minimum(b1_y2, b2_y2)

    w = np.maximum(0.0, yy2 - yy1)
    h = np.maximum(0.0, xx2 - xx1)

    inter = w * h
    IoU = inter / ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter)
    return IoU

def process_files(pred_folder, label_folder, iou_value):
    pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.txt')]
    label_files = {f.split('.')[0]: os.path.join(label_folder, f) for f in os.listdir(label_folder) if
                   f.endswith('.txt')}

    results = {}
    cnt1 = 0
    cnt2 = 0
    for pred_file in pred_files:
        pred_boxes = read_boxes_from_txt(os.path.join(pred_folder, pred_file))
        pred_file_name = pred_file.split('.')[0]

        if pred_file_name in label_files:
            label_boxes = read_boxes_from_txt(label_files[pred_file_name])

            for pred_box in pred_boxes:
                cnt1 = cnt1 + 1
                flag = False
                for label_box in label_boxes:
                    iou = calculate_iou(pred_box, label_box)
                    if iou > iou_value:
                        flag = True
                        # 种类相同且IOU阈值较大
                        if pred_box[0] == label_box[0]:  # 类别相同
                            pass
                        # 种类不同但是IOU阈值比较大，也不考虑，反正在扩散模型训练的时候也不会有什么帮助
                        else:
                            pass
                if not flag:
                    cnt2 = cnt2 + 1
                    if pred_file_name in results:
                        results[pred_file_name].append(pred_box)
                    else:
                        results[pred_file_name] = [pred_box,]
    print(cnt1, cnt2)
    return results

def merge_dicts(dict1, dict2):
    # 使用get方法获取value，如果key不存在则返回空list
    # 然后使用+操作符来合并两个list
    merged = {k: dict1.get(k, []) + dict2.get(k, []) for k in set(dict1) | set(dict2)}
    return merged

# pred_folder1 = '/home/mly/AMSP_UOD/runs/detect/exp4/labels'             # 对应AMSP_UOD检测器
pred_folder2 = '/home/mly/ultralytics_main/runs/detect/predict3/labels'  # 对应yolov8n检测器
label_folder = '/mnt/data0/mly/RUOD/Generate_image_dataset/labels'
# results1 = process_files(pred_folder1, label_folder, 0.5)
results2 = process_files(pred_folder2, label_folder, 0.5)


'''
# 按照标签文件绘制图像
merged_dict = merge_dicts(results1, results2)
# 查询在result标签存在的图像名称，进行真实labels的绘制，在此基础上进行result标签绘制
print(merged_dict)
with open('detect_result.json', 'w') as f:
    json.dump(merged_dict, f, indent=4)
'''

# 图像绘制

img_path_0 = '/mnt/data0/mly/RUOD/Generate_image_dataset/images/'
for img_path, extra_labels in results2.items():

    label_path = os.path.splitext(img_path)[0] + '.txt'
    img_path = img_path_0 + img_path + '.jpg'
    label_path = label_folder + '/' + label_path

    # 加载图片，标注，并保存
    annotated_img = load_and_annotate_image(img_path, label_path, extra_labels)
    if annotated_img is not None:
        # 保存标注后的图片，例如保存在名为 'annotated_images' 的文件夹中
        output_dir = '/mnt/data0/mly/RUOD/展示扩充数据集在yolov8n检测器下漏检情况'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, annotated_img)
        print(f"Annotated image saved: {output_path}")
