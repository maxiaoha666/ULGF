import os
import shutil

# 设置源文件夹和目标文件夹路径
source_dir = '/mnt/data0/mly/URPC2020/datasets/labels/train'  # 替换为你的源文件夹路径
destination_dir = '/mnt/data0/mly/URPC2020/gendatasets/priorrandom/labels'  # 替换为你的目标文件夹路径

# 确保目标文件夹存在
os.makedirs(destination_dir, exist_ok=True)

# 遍历源文件夹中的文件
for filename in os.listdir(source_dir):
    # 检查文件名是否以 'geo' 开头
    if filename.startswith('Geo'):
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)

        # 如果是文件，执行移动操作
        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination_file)
            print(f'Moved: {filename}')
