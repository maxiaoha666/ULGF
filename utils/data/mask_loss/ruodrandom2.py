import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from mmdet.datasets.builder import DATASETS
from utils.data.nuimage import NuImageDataset

# 修改ruodrandom中的错位问题
@DATASETS.register_module()
class RUODDataset(NuImageDataset):
    CLASSES = ["holothurian","echinus","scallop","starfish","fish","corals"
                ,"diver","cuttlefish","turtle","jellyfish"]

    def __init__(self, prompt_version='v1', num_bucket_per_side=None,
                 foreground_loss_mode=None, foreground_loss_weight=1.0, foreground_loss_norm=False, feat_size=64,
                 uncond_prob=0.,
                 **kwargs):
        super().__init__(**kwargs)
        self.prompt_version = prompt_version
        self.no_sections = num_bucket_per_side
        print('Using prompt version: {}, num_bucket_per_side: {}'.format(prompt_version, num_bucket_per_side))

        self.FEAT_SIZE = [each // 8 for each in kwargs['pipeline'][2].img_scale][::-1]
        print('Using feature size: {}'.format(self.FEAT_SIZE))

        self.foreground_loss_mode = foreground_loss_mode
        self.foreground_loss_weight = foreground_loss_weight
        self.foreground_loss_norm = foreground_loss_norm
        print('Using foreground_loss_mode: {}, foreground_loss_weight: {}, foreground_loss_norm: {}'.format(
            foreground_loss_mode, foreground_loss_weight, foreground_loss_norm))

        self.uncond_prob = uncond_prob
        print('Using unconditional generation probability: {}'.format(uncond_prob))

        self.class2text = {
            'holothurian': 'holothurian',
            'echinus': 'echinus',
            'scallop': 'scallop',
            'starfish': 'starfish',
            'fish': 'fish',
            'corals': 'corals',
            'diver': 'diver',
            'cuttlefish': 'cuttlefish',
            'turtle': 'turtle',
            'jellyfish': 'jellyfish',
        }

    def __getitem__(self, idx):
        ##################################
        # Data item: {pixel_values: tensor of (3, H, W),  text: string}
        ##################################
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            # 找出文件对应的标签
            ori_filename = data['img_metas'].data['ori_filename']

            bboxes = data['gt_bboxes'].data
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            labels = [self.CLASSES[each].split('.')[-1] for each in data['gt_labels'].data]
            # camera = ' '.join(data['img_metas'].data['ori_filename'].split('/')[-2].split('_')[1:])

            if self.prompt_version == 'v1':
                pad_shape = data['img_metas'].data['pad_shape']
                img_shape = torch.tensor([pad_shape[1], pad_shape[0], pad_shape[1], pad_shape[0]])
                bboxes /= img_shape

                # random shuffle bbox annotations
                index = list(range(len(labels)))
                index = index[:22]  # 9+3*22+2=77

                objs = []
                bbox_mask = torch.zeros(self.FEAT_SIZE).float()  # [H, W]
                # 存储目标的位置信息列表，不随机：
                location_list = []
                weight_dict = {}
                num_dui = 0
                for num, each in enumerate(index):
                    label = labels[each]
                    bbox = bboxes[each]

                    # generate bbox mask
                    FEAT_SIZE = torch.tensor(
                        [self.FEAT_SIZE[1], self.FEAT_SIZE[0], self.FEAT_SIZE[1], self.FEAT_SIZE[0]])
                    coord = torch.round(bbox * FEAT_SIZE).int().tolist()
                    # 为什么加下面两行：因为deal_with_mask会将最后一个数视为框的范围
                    coord[2] = coord[2] - 1
                    coord[3] = coord[3] - 1
                    if coord[2] < coord[0] or coord[3] < coord[1]:
                        num_dui = num_dui + 1
                        continue
                    location_list.append(tuple(coord))

                    weight = self.foreground_loss_weight * 1 / torch.pow(
                            torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)), 0.2)
                    weight_dict[num-num_dui] = weight
                    bbox = self.token_pair_from_bbox(bbox.tolist())
                    objs.append(' '.join([self.class2text[label], bbox]))

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
                    '''
                    for i in grouped_by_value.items():
                        if len(i[0]) == 4:
                            print(ori_filename+'存在四遮挡问题')
                            break
                    '''
                    grouped_by_value = dict(sorted(grouped_by_value.items(), key=custom_sort_key))
                    grouped_by_value_result = {key: 0 for key in grouped_by_value}

                    for num, (target_list, points) in enumerate(grouped_by_value.items()):
                        # random_target_id = 99 默认检测框没这么多
                        if len(target_list) == 1:
                            # 长度为1的key，直接选
                            grouped_by_value_result[target_list] = target_list[0]
                        if len(target_list) == 3:
                            grouped_by_value_result[target_list] = random.choice(target_list)
                        if len(target_list) == 2:
                            target_list_2 = True
                            for i_target_list in grouped_by_value_result.keys():
                                if len(i_target_list) == 3 and (set(target_list).issubset(i_target_list)) and (
                                        grouped_by_value_result[i_target_list] in target_list):
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

                    return final_point_to_target

                final_point_to_target = deal_with_mask(targets=location_list)
                for point, target_id in final_point_to_target.items():
                    px, py = point
                    bbox_mask[py, px] = target_id + 1  # +1 是为了避免目标ID为0，确保有颜色区分
                for key, value in weight_dict.items():
                    bbox_mask[bbox_mask == (key+1)] = value
                if torch.isnan(bbox_mask).any():
                    print(1)

                '''可视化
                if ori_filename =='011661.jpg':
                    plt.imshow(bbox_mask, cmap='tab20', interpolation='nearest')  # 使用tab20颜色映射
                    plt.colorbar()  # 显示颜色条
                    plt.title("Final Point to Target Visualization")
                    plt.show()
                '''


                # bbox_mask[bbox_mask == 0] = 1 if self.foreground_loss_mode == 'constant' else 1 / torch.pow(torch.sum(bbox_mask == 0), self.foreground_loss_weight)
                bbox_mask[bbox_mask == 0] = 1 * 1 / torch.pow(torch.tensor(self.FEAT_SIZE[0] * self.FEAT_SIZE[1]),
                                                              0.2) if self.foreground_loss_mode == 'constant' else 1 / torch.pow(
                    torch.tensor(self.FEAT_SIZE[0] * self.FEAT_SIZE[1]), self.foreground_loss_weight)
                bbox_mask = bbox_mask / torch.sum(bbox_mask) * self.FEAT_SIZE[0] * self.FEAT_SIZE[
                    1] if self.foreground_loss_norm else bbox_mask
                if torch.isnan(bbox_mask).any():
                    print(1)


                if self.uncond_prob > 0:
                    text = 'A underwater scene image of ' + 'front' + ' camera with ' + ' '.join(
                        objs) if random.random() > self.uncond_prob else ""
                else:
                    text = 'A underwater scene image of ' + 'front' + ' camera with ' + ' '.join(objs)

            else:
                raise NotImplementedError("Prompt version {} is not supported!".format(self.prompt_version))

            example = {}
            example["pixel_values"] = data['img'].data
            example["text"] = text
            example['ori_filename'] = ori_filename
            if self.foreground_loss_mode is not None:
                example["bbox_mask"] = bbox_mask

            return example

    # code borrowed from https://github.com/CompVis/taming-transformers
    def tokenize_coordinates(self, x: float, y: float) -> int:
        """
        Express 2d coordinates with one number.
        Example: assume self.no_tokens = 16, then no_sections = 4:
        0  0  0  0
        0  0  #  0
        0  0  0  0
        0  0  0  x
        Then the # position corresponds to token 6, the x position to token 15.
        @param x: float in [0, 1]
        @param y: float in [0, 1]
        @return: discrete tokenized coordinate
        """
        x_discrete = int(round(x * (self.no_sections[0] - 1)))
        y_discrete = int(round(y * (self.no_sections[1] - 1)))
        return "<l{}>".format(y_discrete * self.no_sections[0] + x_discrete)

    def token_pair_from_bbox(self, bbox):
        return self.tokenize_coordinates(bbox[0], bbox[1]) + ' ' + self.tokenize_coordinates(bbox[2], bbox[3])

