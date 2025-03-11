import torch_fidelity

import os
import shutil
'''
# 设定源文件夹路径
source_folder = 'F:/datasets/RUOD/先验去噪，超参数为0.85，图片'

# 目标文件夹路径（假设它们位于源文件夹的同级目录下）
target_folder_A = os.path.join(os.path.dirname(source_folder), 'FID0.85Image')
target_folder_B = os.path.join(os.path.dirname(source_folder), 'FIDNoneImage')

# 确保目标文件夹存在
os.makedirs(target_folder_A, exist_ok=True)
os.makedirs(target_folder_B, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 构建完整的文件路径
    file_path = os.path.join(source_folder, filename)

    # 检查是否为文件（排除目录）
    if os.path.isfile(file_path):
        # 检查文件名长度
        if len(filename) > 12:
            # 如果文件名长度大于12，则复制到A文件夹
            shutil.copy(file_path, os.path.join(target_folder_A, filename))
        else:
            # 否则，复制到B文件夹
            shutil.copy(file_path, os.path.join(target_folder_B, filename))

print("文件分类完成。")
'''
'''
from PIL import Image
import os


def resize_images_in_folder(folder_path, target_size=(256, 256), output_folder='/mnt/data0/mly/uwcnnganuwnr/UWCNN256/gt_type5'):
    """
    遍历指定文件夹中的所有图像文件，将它们调整大小到目标尺寸，并保存到新文件夹中。

    :param folder_path: 原始图像的文件夹路径
    :param target_size: 目标尺寸，格式为(宽度, 高度)
    :param output_folder: 保存调整大小后图像的文件夹名称
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 忽略非图像文件
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            continue

            # 构造原始图像和输出图像的路径
        img_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder, filename)

        # 使用Pillow打开图像
        with Image.open(img_path) as img:
            # 调整图像大小
            if img.mode == 'RGBA':
                # 将RGBA模式转换为RGB模式
                img = img.convert('RGB')

            resized_img = img.resize(target_size, Image.ANTIALIAS)

            # 保存调整大小后的图像
            resized_img.save(output_path)

        # 使用示例


folder_path = '/mnt/data0/mly/uwcnnganuwnr/UWCNN/gt_typeIA'  # 替换为你的图像文件夹路径
output_folder ='/mnt/data0/mly/uwcnnganuwnr/UWCNN256/gt_typeIA'
resize_images_in_folder(folder_path=folder_path,output_folder=output_folder)

'''

metrics_dict = torch_fidelity.calculate_metrics(
    input1='/mnt/data0/mly/uwcnnganuwnr/ULGF/Raw256Image',
    input2='/mnt/data0/mly/RUOD/Generate_image_dataset/ULFG',
    fid=True
)
print(metrics_dict)






