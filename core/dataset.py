import os
import glob
import cv2
import torch
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from .dataset_util import set_data, parse_xml, xml_parse_get_max_id, xywh2xyxy, xyxy2xywh


class Random_Augmentation():
    def __init__(self, random=0.2, img_size=512):
        self.img_size = img_size
        sometimes = lambda aug: iaa.Sometimes(random, aug)
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.ChannelShuffle(p=0.3),
            sometimes(iaa.AddElementwise([-10, 10])),
            sometimes(iaa.JpegCompression(compression=(90, 91))),
            sometimes(iaa.Rotate((-20, 20))),
            sometimes(iaa.MedianBlur(k=(3,7))),
            sometimes(iaa.TranslateY(px=(-25, 25))),
            sometimes(iaa.TranslateX(px=(-25, 25))),
        ])

    def __call__(self, img, boxes):
        boxes[..., :4] = xywh2xyxy(boxes[..., :4])
        bbs = []
        for i in range(len(boxes)):
            box = boxes[i][:4]
            bbs.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
        
        images_aug, bbs = self.seq(images=[img], bounding_boxes=bbs)
        boxes = np.empty(0,)
        for bb in bbs:
            box = np.concatenate([bb[0], bb[1]], 0)
            boxes = np.append(boxes, box)
        boxes = np.reshape(boxes, (-1, 4))
        boxes = np.clip(boxes, 0, self.img_size - 1)
        boxes[..., :4] = xyxy2xywh(boxes[..., :4])
        return images_aug[0], boxes


class MOTDatset(torch.utils.data.Dataset):
    def __init__(self, data_path, class_file, input_image_size=512, grid_resolution=128):
        super(MOTDatset, self).__init__()
        self.data_list = glob.glob(os.path.join(data_path, '*/JPEGImages/*'))
        self._id_counting()
        # self.data_list = sorted(self.data_list)
        np.random.shuffle(self.data_list)
        self.input_image_size = input_image_size
        self.grid_resolution = grid_resolution
        class_list = open(class_file, 'r').readlines()
        self.class_dic = dict([(name.strip(), i) for i, name in enumerate(class_list)])
        self.class_dic_rev = dict([(i, name.strip()) for i, name in enumerate(class_list)])
        self.random_augmentation = Random_Augmentation(random=0.2, img_size=input_image_size)

    def __len__(self):
        return len(self.data_list)

    def _id_counting(self,):
        self.total_id_nums = 0
        for sample_img_path in self.data_list:
            sample_xml_path = sample_img_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')
            max_id = xml_parse_get_max_id(sample_xml_path)
            self.total_id_nums = max(max_id, self.total_id_nums)

    def random_aug(self, img, data):
        img, data[..., :4] = self.random_augmentation(img, data[..., :4])
        return img, data

    def __getitem__(self, idx):
        sample_img_path = self.data_list[idx]
        sample_xml_path = sample_img_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')
        img_ori = cv2.imread(sample_img_path)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        y_size, x_size, _ = img_ori.shape
        img = cv2.resize(img_ori, (self.input_image_size, self.input_image_size))
        data = parse_xml(sample_xml_path, self.class_dic, self.input_image_size, x_size, y_size, box_format='xywh')
        img, data = self.random_aug(img, data)
        filters, others_info, id_info = set_data(data, self.class_dic_rev, self.total_id_nums, self.input_image_size, self.grid_resolution)
        img = torch.from_numpy(img.transpose(2, 0, 1) / 255.).float()
        filters = torch.from_numpy(filters).float()
        others_info = torch.from_numpy(others_info).float()
        return img, filters, others_info, id_info # [width, height, offset_x, offset_y]


    @staticmethod
    def collate_fn(batch):
        """should make each label have a same number of objs"""
        images, filters , others_info, id_info = zip(*batch)
        images = torch.stack(images, dim=0)
        filters = torch.stack(filters, dim=0)
        others_info = torch.stack(others_info, dim=0)
        id_info = torch.stack(id_info, dim=0)
        return images, filters, others_info, id_info