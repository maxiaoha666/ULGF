import json
import numpy as np



def check_lists_for_floats(data_dict):
    errors = []  # 用于存储错误的列表

    for key, value in data_dict.items():
        if isinstance(value, list):  # 检查值是否为列表
            for item in value:
                if not isinstance(item, float):  # 检查列表中的每个元素是否为浮点型

                    errors.append(f"Key '{key}' contains a non-float value: {item} (type: {type(item)})")
                    break  # 如果找到非浮点型元素，则跳出内层循环
            if not errors:  # 如果当前列表检查没有错误，但之前可能有错误
                # 你可以在这里添加逻辑来处理完全通过检查的列表，但在这个示例中我们省略了
                pass

    if errors:
        return errors
    else:
        return "All lists contain only float values."


def flatten_dict_lists(input_dict):
    # 创建一个新的字典来存储结果
    result_dict = {}

    # 遍历输入字典的每个key和value
    for key, nested_list in input_dict.items():
        # 初始化一个空列表来存储扁平化后的元素
        flat_list = []

        # 遍历嵌套列表中的每个子列表
        for sublist in nested_list:
            # 使用列表推导式将子列表中的元素添加到flat_list中
            flat_list.extend(sublist)

            # 将扁平化后的列表存储在结果字典中
        result_dict[key] = flat_list

        # 返回结果字典
    return result_dict

with open('data.json', 'r') as f:
    data = json.load(f)

data_flatten = flatten_dict_lists(data)

prior_distribution = {}

for key, value in data_flatten.items():
    my_array = np.array(value)
    mean = np.mean(my_array)
    std_dev = np.std(my_array)
    variance = np.var(my_array)
    print(f'class:{key}     Mean: {mean}, Standard Deviation: {std_dev}, Variance: {variance}')
    prior_distribution[key] = [mean, std_dev, variance]
with open('/home/mly/UWLGM/prior_probability_distribution/prior_distribution.json', 'w') as f:
    json.dump(prior_distribution, f)
# print(check_lists_for_floats(data_flatten))

