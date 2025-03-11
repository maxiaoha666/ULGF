import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
class CustomDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None, device=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        checkpoint = torch.load(self.image_paths[idx], map_location=device)  # 读取图片
        input_tensor = None
        label_tensor = None
        for key in checkpoint.keys():
            input_tensor = checkpoint[key]
        checkpoint = torch.load(self.label_paths[idx], map_location=device)  # 读取图片
        for key in checkpoint.keys():
            label_tensor = checkpoint[key]

        if self.transform:
            image = self.transform(input_tensor)  # 应用变换

        return input_tensor, label_tensor


class MaskConvNet(nn.Module):
    def __init__(self):
        super(MaskConvNet, self).__init__()

        # 第一层卷积，输入16个通道，输出16个通道
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        # 第二层卷积，输入16个通道，输出16个通道
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        # 第三层卷积，输入16个通道，输出16个通道
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)
        # BatchNormalization层，对每个通道进行归一化
        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, x):
        # 卷积 + ReLU + 卷积 + ReLU + BatchNormalization
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.batch_norm(x)

        # 归一化后输出
        return x

if __name__ == '__main__':
    device = torch.device('cuda:0')

    image_dir = '/mnt/data0/mly/RUOD/LossInput'
    label_dir = '/mnt/data0/mly/RUOD/LossLabel'

    # 获取图像和标签文件路径
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
    label_paths = [os.path.join(label_dir, filename) for filename in os.listdir(label_dir)]

    # 确保图像与标签一一对应
    assert len(image_paths) == len(label_paths)

    # 设置随机种子，确保每次分割结果相同
    random.seed(0)
    # 打乱数据
    data = list(zip(image_paths, label_paths))  # 将图像路径和标签路径配对
    random.shuffle(data)  # 随机打乱数据

    # 分割数据，80% 用于训练，20% 用于验证
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    # 创建训练集和验证集的 Dataset 对象
    train_dataset = CustomDataset(
        image_paths=[item[0] for item in train_data],
        label_paths=[item[1] for item in train_data],
        transform=transforms.Compose([
        ]),
        device=device
    )

    val_dataset = CustomDataset(
        image_paths=[item[0] for item in val_data],
        label_paths=[item[1] for item in val_data],
        transform=transforms.Compose([
        ]),
        device=device
    )

    # 创建 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=0)

    '''
    # 测试数据加载
    for inputs, labels in train_loader:
        print("Input batch shape:", inputs.shape)
        print("Label batch shape:", labels.shape)
        break  # 只打印第一个 batch
    '''
    # 加载训练模型
    model = MaskConvNet()
    model.to(device)
    # 设计损失函数
    criterion = nn.MSELoss()
    # 设置优化器（2层）0.00001 0.0961  0.001 0.0889    0.01 0.2883（震荡）
    # 设置优化器（3层）CCCB 0.00001 0.2883（震荡） 0.001 0.0804  0.01 0.0750
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 每训练10个epoch，学习率减小为原来的一半
    # scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    # 训练循环
    best_loss = 1
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0

        for inputs, labels in train_loader:
            # 转成想要的输入格式 (16, 32, 32)  ----》 (16, 1, 32, 32)
            inputs = inputs.unsqueeze(1)
            labels = labels.unsqueeze(1)
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新权重
            optimizer.step()
            # 累积损失
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        # 每个epoch结束后，进行评估（可以使用验证集）
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        with torch.no_grad():  # 不计算梯度，节省内存
            for inputs, labels in val_loader:
                inputs = inputs.unsqueeze(1)
                labels = labels.unsqueeze(1)

                # 前向传播
                outputs = model(inputs)

                # 计算损失
                loss = criterion(outputs, labels)

                # 累积损失
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')

        # 如果验证集上的损失比之前更低，则保存模型
        if val_loss < best_loss:
            best_loss = val_loss
            # 保存模型
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved model with validation loss: {val_loss:.4f}")

    print("训练完成！")



