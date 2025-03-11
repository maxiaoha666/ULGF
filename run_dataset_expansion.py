import cv2
import os
import random
import numpy as np

from PIL import Image
from utils.generation_utils import load_checkpoint, bbox_encode, draw_layout
from argparse import ArgumentParser
from accelerate.utils import set_seed
import matplotlib.pyplot as plt


# 定义函数将类别名转换为类别索引
def class_to_index(class_name):
    return classes.index(class_name)


def obj_to_yolo_label(obj):
    class_name = obj[0]
    xmin, ymin, xmax, ymax = obj[1:]

    # 计算物体中心点的相对位置
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    # 计算物体的相对宽度和高度
    width = xmax - xmin
    height = ymax - ymin

    # 将相对位置和尺寸归一化为0~1之间
    x_center_normalized = x_center
    y_center_normalized = y_center
    width_normalized = width
    height_normalized = height

    # 获取类别索引
    class_index = class_to_index(class_name)

    # 格式化为YOLO标签文本
    yolo_label = f"{class_index} {x_center_normalized} {y_center_normalized} {width_normalized} {height_normalized}"

    return yolo_label


# 定义函数将YOLO标签写入txt文件
def write_yolo_labels_to_file(yolo_labels, file_path):
    with open(file_path, 'w') as f:
        for label in yolo_labels:
            f.write(label + '\n')


# 省略小与0.2%的box
def is_area_too_small(tmp_precent):
    xmin = tmp_precent[1]
    ymin = tmp_precent[2]
    xmax = tmp_precent[3]
    ymax = tmp_precent[4]

    area = (xmax - xmin) * (ymax - ymin)

    if area < 0.0001:
        return False
    else:
        return True


# 50%的概率翻转标签
def mirror_horizontal(tmp):
    random_number = random.random()
    if random_number <= 0.5:
        name = tmp[0]
        xmin = tmp[1]
        ymin = tmp[2]
        xmax = tmp[3]
        ymax = tmp[4]

        # 计算镜像后的坐标
        new_xmin = 1 - xmax
        new_xmax = 1 - xmin
        new_ymin = ymin
        new_ymax = ymax

        # 构建镜像后的列表
        mirrored_tmp = [name, new_xmin, new_ymin, new_xmax, new_ymax]
    else:
        mirrored_tmp = tmp
    return mirrored_tmp


def txtShow(img, txt, classes, img_dic, label_dic, save=True):
    image = cv2.imread(img)
    # height, width = image.shape[:2]  # 获取原始图像的高和宽
    height, width = 256, 256

    # 读取yolo格式标注的txt信息
    with open(txt, 'r') as f:
        labels = f.read().splitlines()
    # ['0 0.403646 0.485491 0.103423 0.110863', '1 0.658482 0.425595 0.09375 0.099702', '2 0.482515 0.603795 0.061756 0.045387', '3 0.594122 0.610863 0.063244 0.052083', '4 0.496652 0.387649 0.064732 0.049107']

    ob = []  # 存放目标信息
    ob_precent = []  # 存放目标信息的百分比形式
    for i in labels:
        cl, x_centre, y_centre, w, h = i.split(' ')

        # 需要将数据类型转换成数字型
        cl, x_centre, y_centre, w, h = int(cl), float(x_centre), float(y_centre), float(w), float(h)
        name = classes[cl]  # 根据classes文件获取真实目标
        xmin = int(x_centre * width - w * width / 2)  # 坐标转换
        ymin = int(y_centre * height - h * height / 2)
        xmax = int(x_centre * width + w * width / 2)
        ymax = int(y_centre * height + h * height / 2)

        tmp = [name, xmin, ymin, xmax, ymax]  # 单个检测框
        tmp_precent = [
            name,
            min(xmin / width, 1),
            min(ymin / height, 1),
            min(xmax / width, 1),
            min(ymax / height, 1)
        ]
        # tmp_precent = mirror_horizontal(tmp_precent)

        if is_area_too_small(tmp_precent):
            ob.append(tmp)
            ob_precent.append(tmp_precent)
        else:
            print('The box area is too small:', tmp_precent)

        if len(ob_precent) >= 12:
            print('When this line of text appears, it indicates that: 1. The number of annotation boxes is greater than the preset 2. The token converted from the annotation boxes is too long')
            break

    layout = {
        "camera": "front",
        "bbox": ob_precent
    }

    # 保存增强后的图像yolo类型的标签
    save_label = label_dic + '/Geo' + os.path.basename(txt)
    yolo_labels = [obj_to_yolo_label(obj) for obj in ob_precent]
    write_yolo_labels_to_file(yolo_labels, save_label)

    return layout


'''
    # plot
    for name, x1, y1, x2, y2 in ob:
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)  # 绘制矩形框
        cv2.putText(image, name, (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, thickness=1, color=(0, 0, 255))

        # save
    file_name = os.path.basename(img)
    file_name = "./results/" +'Geo_'+file_name
    if save:
        cv2.imwrite(file_name, image)
'''


def run_layout_to_image(layout, save_label, pipe, generation_config, denoise_img):
    ########################
    # Build pipeline
    #########################
    pipe = pipe
    generation_config = generation_config

    # Sometimes the nsfw checker is confused by the Pokémon images, you can disable
    # it at your own risk here
    disable_safety = True
    if disable_safety:
        def null_safety(images, **kwargs):
            return images, False

        pipe.safety_checker = null_safety

    # camera sanity check
    assert not generation_config['dataset'] == 'nuimages' or (
            "camera" in layout and layout['camera'] in ['front', 'front left', 'front right', 'back', 'back left',
                                                        'back right'])
    bboxes = layout['bbox'].copy()
    layout["bbox"] = bbox_encode(layout['bbox'], generation_config)
    prompt = generation_config['prompt_template'].format(**layout)
    print(prompt)

    ########################
    # Generation
    ########################
    # generation params
    width = generation_config["width"]
    height = generation_config["height"]
    scale = generation_config["cfg_scale"]
    n_samples = generation_config["nsamples"]
    num_inference_steps = generation_config["num_inference_steps"]

    # run generation
    images = pipe.__call__(n_samples * [prompt], guidance_scale=scale, num_inference_steps=num_inference_steps,
                  height=int(height), width=int(width),).images


    ########################
    # Save results
    #########################
    root = generation_config["output_dir"]
    os.makedirs(root, exist_ok=True)
    for idx, image in enumerate(images):
        image = np.asarray(image)
        image = Image.fromarray(image, mode='RGB')
        save_label = root + save_label
        print(save_label)
        image.save(save_label)


def delete_geo_files(folder_path):
    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件名是否以"Geo"开头
            if file.startswith("Geo"):
                # 构建文件的完整路径
                file_path = os.path.join(root, file)
                # 删除文件
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except OSError as e:
                    print(f"Error: {e.strerror} : {file_path}")


if __name__ == '__main__':

    classes = ["holothurian", "echinus", "scallop", "starfish", "fish", "corals", "diver", "cuttlefish", "turtle",
               "jellyfish"]
    img_dic = '/mnt/data0/mly/RUOD/ruod_expand/images/train'  # 传入图片
    label_dic = '/mnt/data0/mly/RUOD/ruod_expand/labels/train'
    seed_dic = '/home/mly/GeoDiffusion/configs/random_seed/ruod_seeds.txt'


    '''
    classes = ["holothurian", "echinus", "scallop", "starfish"]
    img_dic = '/mnt/data0/mly/URPC2020/datasets/images/train'  # 传入图片
    label_dic = '/mnt/data0/mly/URPC2020/datasets/labels/train'
    seed_dic = '/home/mly/GeoDiffusion/configs/random_seed/ruod_seeds.txt'
    '''
    '''
    classes = ["holothurian", "echinus", "scallop", "starfish", "fish", "corals", "diver", "cuttlefish", "turtle",
               "jellyfish"]
    img_dic = '/mnt/data0/mly/RUOD/ruod_expand/labels/绘图图像'  # 传入图片
    label_dic = '/mnt/data0/mly/RUOD/ruod_expand/labels/绘图标签'
    seed_dic = '/home/mly/GeoDiffusion/configs/random_seed/ruod_seeds.txt'
    '''
    parser = ArgumentParser(description='Layout-to-image generation script')
    parser.add_argument('--ckpt_path', type=str,
                        default='/mnt/data0/mly/GeoModel/work_dir/geodiffusion_ruod_256_bz16_LossMsakRandomTrue/checkpoint/final')
    parser.add_argument('--nsamples', type=int, default=1)
    parser.add_argument('--cfg_scale', type=float, default=None)
    parser.add_argument('--num_inference_steps', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=img_dic)
    args = parser.parse_args()

    read_seeds = []
    with open(seed_dic, 'r') as f:
        for line in f:
            read_seeds.append(int(line.strip()))
    cnt_num = 0
    # Delete generated images
    ''''''
    delete_geo_files(img_dic)
    delete_geo_files(label_dic)

    # 加载管线
    pipe, generation_config = load_checkpoint(args.ckpt_path)
    print('The model has been loaded and the path is', args.ckpt_path)
    pipe = pipe.to("cuda")
    args = {arg: getattr(args, arg) for arg in vars(args) if getattr(args, arg) is not None}
    generation_config.update(args)

    for root, dirs, files in os.walk(img_dic):
        for file in files:
            # 获取文件的完整路径（包括根目录和文件名）
            img_path = os.path.join(root, file)
            label_path = os.path.join(label_dic, file)
            label_path = label_path[:-3] + 'txt'
            if os.path.basename(img_path)[0] == 'G':
                print('This is the generated image:', img_path)
                print(
                    '----------------------------------------------------------------------------------------------------------------------------------')
                continue
            else:
                if os.path.exists(root + '/Geo' + os.path.basename(img_path)):
                    cnt_num = cnt_num + 1
                    print('There are already generated images:', root + '/Geo' + os.path.basename(img_path))
                    print(
                        '----------------------------------------------------------------------------------------------------------------------------------')
                    continue
                else:
                    if os.path.basename(img_path) == '008431.jpg':
                        print('The image label has an issue and will not be enhanced')
                        print(
                            '----------------------------------------------------------------------------------------------------------------------------------')
                        continue

                    set_seed(read_seeds[cnt_num])
                    cnt_num = cnt_num + 1
                    print('This image has not undergone any generation operation', os.path.basename(img_path), 'The random seed assigned to it is:',
                          read_seeds[cnt_num - 1])

                    layout = txtShow(img=img_path, txt=label_path, classes=classes, img_dic=img_dic,
                                     label_dic=label_dic, save=True)

                    save_label = '/Geo' + os.path.basename(img_path)
                    denoise_img = img_dic + '/' + os.path.basename(img_path)
                    run_layout_to_image(layout, save_label, pipe, generation_config, denoise_img)
                    print(
                        '----------------------------------------------------------------------------------------------------------------------------------')
