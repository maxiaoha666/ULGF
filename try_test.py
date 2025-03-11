import random
import numpy as np
import matplotlib.pyplot as plt
# 假设目标和掩码如下
targets = [(2, 2, 6, 6), (4, 4, 8, 8), (5, 5, 9, 9)]  # 格式： (x1, y1, x2, y2)
# targets = [(0, 12, 25, 28), (26, 13, 31, 17), (0, 25, 5, 29), (21, 25, 27, 28), (24, 23, 24, 24)]
def deal_with_mask(targets):
    # 存储点对应的目标
    point_to_targets = {}

    # 遍历每个目标，判断坐标点是否在目标范围内
    for target_id, (x1, y1, x2, y2) in enumerate(targets):
        for px in range(x1, x2 + 1):
            for py in range(y1, y2 + 1):
                if (px, py) not in point_to_targets:
                    point_to_targets[(px, py)] = []
                point_to_targets[(px, py)].append(target_id)

    # 归类处理：将所有具有相同value（目标ID列表）的key进行合并
    grouped_by_value = {}
    for point, target_list in point_to_targets.items():
        # 将目标ID列表转换为元组，以便进行hashable比较
        target_tuple = tuple(target_list)
        if target_tuple not in grouped_by_value:
            grouped_by_value[target_tuple] = []
        grouped_by_value[target_tuple].append(point)

    # 对具有相同目标ID列表的点，随机选择一个目标ID
    final_point_to_target = {}


    # 自定义排序：优先处理1位元组，再处理3位元组，最后处理2位元组
    def custom_sort_key(item):
        target_list = item[0]  # 获取元组
        length = len(target_list)
        # 通过长度优先级排序：长度为1的排在最前，3位的排其后，2位的最后
        if length == 1:
            return (0, target_list)  # 长度为1的元组优先
        elif length == 3:
            return (1, target_list)  # 长度为3的元组排在第二
        else:
            return (2, target_list)  # 长度为2的元组排最后
    grouped_by_value = dict(sorted(grouped_by_value.items(), key=custom_sort_key))
    grouped_by_value_result = {key: 0 for key in grouped_by_value}

    for num, (target_list, points) in enumerate(grouped_by_value.items()):
        # random_target_id = 99 默认检测框没这么多
        print(target_list)
        random_target_id = 99
        if len(target_list) == 1:
            # 长度为1的key，直接选
            grouped_by_value_result[target_list] = target_list[0]
        if len(target_list) == 3:
            grouped_by_value_result[target_list] = random.choice(target_list)
        if len(target_list) == 2:
            target_list_2 = True
            for i_target_list in grouped_by_value_result.keys():
                if len(i_target_list) == 3 and (set(target_list).issubset(i_target_list)) and (grouped_by_value_result[i_target_list] in target_list):
                    grouped_by_value_result[target_list] = grouped_by_value_result[i_target_list]
                    target_list_2 = False
                else:
                    if len(i_target_list) == 3 and (set(target_list).issubset(i_target_list)):
                        grouped_by_value_result[target_list] = random.choice(target_list)
                        target_list_2 = False
            if target_list_2:
                grouped_by_value_result[target_list] = random.choice(target_list)
        for point in points:
            final_point_to_target[point] = grouped_by_value_result[target_list]


    # 打印结果
    print("归类后的点和目标ID：")
    for point, target_id in final_point_to_target.items():
        print(f"点 {point} 被归类为目标 {target_id}")
    return final_point_to_target

# 可视化最终结果
# 设定一个32x32的空白图像
img = np.zeros((32, 32), dtype=int)

# 填充每个点的位置为其目标ID
for point, target_id in deal_with_mask(targets).items():
    px, py = point
    img[py, px] = target_id + 1  # +1 是为了避免目标ID为0，确保有颜色区分

# 使用matplotlib显示图像
plt.imshow(img, cmap='tab20', interpolation='nearest')  # 使用tab20颜色映射
plt.colorbar()  # 显示颜色条
plt.title("Final Point to Target Visualization")
plt.show()

