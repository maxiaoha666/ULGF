import random
import json

import torch

from mmdet.datasets.builder import DATASETS
from utils.data.nuimage import NuImageDataset


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

            with open('/home/mly/UWLGM/tools/detect_result.json', 'r', encoding='utf-8') as file:
                detect_box = json.load(file)  # 将JSON文件内容转换为Python字典

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
                random.shuffle(index)
                index = index[:22]  # 9+3*22+2=77

                # generate bbox mask and text prompt
                # constant: background -> 0, foreground -> self.foreground_loss_weight
                # area:     background -> 0, foreground -> 1 / area ^ self.foreground_loss_weight (for area, smaller weight, larger variance with respect to areas)
                objs = []
                bbox_mask = torch.zeros(self.FEAT_SIZE).float()  # [H, W]
                # class_dict储存：一张图像中，每一个类别，所有的box_mask的并集
                
                class_dict = {}
                for each in index:
                    # 对每一个box单独创建一个空白张量
                    bbox_mask_01 = torch.zeros(self.FEAT_SIZE).float()
                    label = labels[each]
                    bbox = bboxes[each]

                    # generate bbox mask
                    FEAT_SIZE = torch.tensor(
                        [self.FEAT_SIZE[1], self.FEAT_SIZE[0], self.FEAT_SIZE[1], self.FEAT_SIZE[0]])
                    coord = torch.round(bbox * FEAT_SIZE).int().tolist()
                    # bbox_mask[coord[1]: coord[3], coord[0]: coord[2]] = self.foreground_loss_weight if self.foreground_loss_mode == 'constant' else \
                    #                                                     1 / torch.pow(torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)), self.foreground_loss_weight)
                    if label in ['truck', 'bicycle', 'motorcycle']:
                        bbox_mask[coord[1]: coord[3],
                        coord[0]: coord[2]] = self.foreground_loss_weight * 2 * 1 / torch.pow(
                            torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)),
                            0.2) if self.foreground_loss_mode == 'constant' else \
                            1 / torch.pow(torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)),
                                          self.foreground_loss_weight)
                    else:
                        if self.foreground_loss_mode == 'constant':
                            bbox_mask[coord[1]: coord[3], coord[0]: coord[2]] = self.foreground_loss_weight * 1 / torch.pow(
                            torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)), 0.2)
                        else:
                            bbox_mask[coord[1]: coord[3], coord[0]: coord[2]] = 1 / torch.pow(torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)),
                                          self.foreground_loss_weight)

                        # 这个掩码用来标注box在潜在空间的位置
                        bbox_mask_01[coord[1]: coord[3], coord[0]: coord[2]] = 1.0
                        if label in class_dict:
                            class_dict[label][coord[1]: coord[3], coord[0]: coord[2]] = 1.0
                        else:
                            class_dict.update({label: bbox_mask_01})


                    # generate text prompt
                    bbox = self.token_pair_from_bbox(bbox.tolist())
                    objs.append(' '.join([self.class2text[label], bbox]))

                # 最后将背景掩码放入，使用的是排除box的做法
                bbox_mask_01 = torch.zeros(self.FEAT_SIZE).float()
                for i in range(bbox_mask_01.size(0)):  # 遍历行
                    for j in range(bbox_mask_01.size(1)):  # 遍历列
                        # 检查所有张量在(i, j)位置的值
                        all_zeros = True
                        for tensor in class_dict.values():
                            if tensor[i, j] != 0.0:
                                all_zeros = False
                                break

                        # 如果所有张量在(i, j)位置的值都是0，则将结果张量的对应位置设置为1
                        if all_zeros:
                            bbox_mask_01[i, j] = 1.0
                class_dict.update({'background': bbox_mask_01})

                # 对background掩码再进行细分，首先读取背景信息中额外的物体信息，这是在detection的基础上做的
                if ori_filename[:-4] not in detect_box:
                    class_dict.update({'bg_de': False})

                else:
                    '''
                    # 添加语义调整信息
                    for each_box in detect_box[ori_filename[:-4]]:
                        if len(objs) < 21:
                            cls, cx, cy, w, h = each_box
                            x1 = cx - w / 2
                            y1 = cy - h / 2
                            x2 = cx + w / 2
                            y2 = cy + h / 2
                            detect_label = self.CLASSES[cls].split('.')[-1]
                            bbox = torch.tensor([x1, y1, x2, y2])

                            # 将detect检测出的背景目标同时给予前景的权重
                            FEAT_SIZE = torch.tensor([self.FEAT_SIZE[1], self.FEAT_SIZE[0], self.FEAT_SIZE[1], self.FEAT_SIZE[0]])
                            coord = torch.round(bbox * FEAT_SIZE).int().tolist()
                            bbox_mask[coord[1]: coord[3], coord[0]: coord[2]] = self.foreground_loss_weight * 1 / torch.pow(
                                torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)),
                                0.2) if self.foreground_loss_mode == 'constant' else \
                                1 / torch.pow(torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)),
                                self.foreground_loss_weight)

                            detect_bbox = self.token_pair_from_bbox(bbox.tolist())

                            objs.append(' '.join([self.class2text[detect_label], detect_bbox]))
                        else:
                            break
                        '''

                    # 进行掩码部分调整
                    bbox_mask_02 = torch.zeros(self.FEAT_SIZE).float()
                    for each_box in detect_box[ori_filename[:-4]]:
                        # 对每一个box单独创建一个空白张量
                        cls, cx, cy, w, h = each_box
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        coord = torch.round(torch.tensor([x1, y1, x2, y2]) * FEAT_SIZE).int().tolist()
                        bbox_mask_02[coord[1]: coord[3], coord[0]: coord[2]] = 1.0
                    # 经过上述操作bbox_mask_02存储了背景信息中所有的检测信息，接下来要保证这些信息均在background里面
                    bgmask = class_dict['background'].bool()
                    demask = bbox_mask_02.bool()
                    demask = torch.logical_and(bgmask, demask)

                    both_true = torch.logical_and(bgmask, demask)
                    one_true = torch.logical_xor(bgmask, demask)
                    bg_de = torch.zeros_like(bgmask, dtype=torch.float32)  # 创建一个与A形状相同、但数据类型为float的tensor
                    bg_de[both_true] = 1.0
                    bg_de[one_true] = 2.0
                    class_dict.update({'bg_de': bg_de})


                # bbox_mask[bbox_mask == 0] = 1 if self.foreground_loss_mode == 'constant' else 1 / torch.pow(torch.sum(bbox_mask == 0), self.foreground_loss_weight)
                bbox_mask[bbox_mask == 0] = 1 * 1 / torch.pow(torch.tensor(self.FEAT_SIZE[0] * self.FEAT_SIZE[1]),
                                                              0.2) if self.foreground_loss_mode == 'constant' else 1 / torch.pow(
                    torch.tensor(self.FEAT_SIZE[0] * self.FEAT_SIZE[1]), self.foreground_loss_weight)

                # 引入背景目标检测策略

                if type(class_dict['bg_de']) != bool:
                    bbox_mask = torch.where(class_dict['bg_de'] == 1.0, torch.tensor(0.0, dtype=bbox_mask.dtype).expand_as(bbox_mask), bbox_mask)

                # 归一化
                bbox_mask = bbox_mask / torch.sum(bbox_mask) * self.FEAT_SIZE[0] * self.FEAT_SIZE[
                    1] if self.foreground_loss_norm else bbox_mask

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
            example['cls_mask'] = class_dict
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