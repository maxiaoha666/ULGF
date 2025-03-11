import torch
import os
import matplotlib.pyplot as plt
import numpy as np

# 指定存储图像的文件夹
save_folder = '/mnt/data0/mly/RUOD/每张图像Loss掩码权重文件归一化可视化/'

# 确保保存文件夹存在
os.makedirs(save_folder, exist_ok=True)

# 指定加载 .pth 文件的文件夹
pth_folder = '/mnt/data0/mly/RUOD/每张图像Loss掩码权重文件归一化/'

# 遍历文件夹中的所有 .pth 文件
for filename in os.listdir(pth_folder):
    if filename.endswith('.pth'):
        # 构建完整的文件路径
        pth_file = os.path.join(pth_folder, filename)

        # 加载 .pth 文件
        tensor_dict = torch.load(pth_file)

        # 假设 tensor_dict 中每个项的值都是一个 Tensor
        for tensor_name, tensor in tensor_dict.items():
            # 进行图像绘制
            plt.figure(figsize=(6, 6))

            # 将 tensor 转换为 numpy 数组并绘制
            tensor_np = tensor.numpy()

            # 创建图像并显示数据条
            im = plt.imshow(tensor_np, cmap='viridis')  # 可以根据需要更改颜色映射 (cmap)
            plt.colorbar(im)  # 显示颜色条

            # 设置图像标题为文件名 + tensor_name
            plt.title(f'{filename} - {tensor_name}')

            # 构造保存路径
            save_path = os.path.join(save_folder, filename[:6] + '.png')

            # 保存图像
            plt.savefig(save_path)
            plt.close()  # 关闭当前图像

            print(f"已保存图像: {save_path}")
