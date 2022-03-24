import cv2
import numpy as np
import xml.etree.ElementTree as ET
import numpy as np


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = np.zeros_like(x)

    y[:, 0] = (x[:, 0] + x[:, 2]) / 2.0
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2.0
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def parse_txt():
    pass


def parse_xml(xml_path, class_dic,  target_size, x_size, y_size, box_format='xyxy'):
    """
    load xml and make array
    box_format xyxy or xywh(xy = center)
    """
    root = ET.parse(xml_path).getroot()
    objs = root.findall('object')
    labels = []
    for obj in objs:
        parsed_label = _get_label_info(obj, class_dic,  target_size, x_size, y_size)
        # if parsed_label[-1] == 2:
        #     continue
        labels.append(parsed_label)
    label = np.array(labels)
    if box_format == 'xywh':
        box_ = label[..., :4]
        class_ = label[..., 4:]
        box_ = xyxy2xywh(box_)
        label = np.concatenate([box_, class_], -1)
    return label


def _get_label_info(_object, class_dic, target_size, x_size, y_size):
    x_scaler = target_size / x_size
    y_scaler = target_size / y_size
    bbox = _object.find('bndbox')
    class_name = _object.find('name').text
    class_name = class_name.strip()
    class_idx = class_dic[class_name]
    x1 = int(bbox.find('xmin').text) * x_scaler
    y1 = int(bbox.find('ymin').text) * y_scaler
    x2 = int(bbox.find('xmax').text) * x_scaler
    y2 = int(bbox.find('ymax').text) * y_scaler
    label = [x1, y1, x2, y2, class_idx]
    return np.array(label, np.int32)



def getGKernel(shape, sigma, offset_x, offset_y):
    s = (shape - 1) / 2
    t = (shape - 1) / 2
    y, x = np.ogrid[-s:s + 1, -t:t + 1]
    x = x - offset_x
    y = y - offset_y
    gaus_kernel = np.exp(-(x**2 + y**2) / ( 2 * sigma**2))# + 0.5
    return gaus_kernel#np.clip(gaus_kernel, 0, 1)


def gaussian_filter(filter, width, height, offset_x=0, offset_y=0):
    kernel_size = max(max(width, height) // 10, 3)
    sigma = max(kernel_size / 10, 0.5)
    kernel = getGKernel(kernel_size, sigma, offset_x, offset_y)
    # kernel = (1 - np.max(kernel)) + kernel
    filter = cv2.filter2D(filter, -1, kernel)
    filter[np.where(filter <= 0.1)] = 0
    filter += 0.3
    filter[np.where(filter == 0.3)] = 0

    # filter[np.where(filter >= 0.5)] = 1
    return np.clip(filter, 0, 1)

def set_data(input_points, class_dic_rev, input_image_size=512, feature_map_size=128):
    class_num = len(class_dic_rev.keys())
    filters = np.zeros((feature_map_size, feature_map_size, class_num))
    filters_temp = np.ones((feature_map_size, feature_map_size, 1))
    
    others_info = np.zeros((feature_map_size, feature_map_size, 4))
    use_point_checker = np.zeros((feature_map_size, feature_map_size)).astype(np.bool)
    scaler = feature_map_size / input_image_size
    for data in input_points:
        filter = np.zeros((feature_map_size, feature_map_size))
        point = data[..., :4]
        class_idx = int(data[..., 4])
        width, height = point[..., 2:] / input_image_size
        point = point[..., :2] * scaler
        point_s = point.astype(int)
        offset_x, offset_y = point - point_s
        filter[point_s[1], point_s[0]] = 1
        filter = gaussian_filter(filter, width * input_image_size, height * input_image_size, offset_x, offset_y)
        filter[point_s[1], point_s[0]] = 1
        now_info = np.ones_like(others_info) * [width, height, offset_x, offset_y]
        now_checker = filter.astype(np.bool)
        others_info = np.where(np.tile(np.expand_dims(filters.max(-1) < filter, -1), (1, 1, 4)), now_info, others_info)
        filters[..., class_idx] = np.where(filters.max(-1) > filter, filters[..., class_idx], filter)
        
        use_point_checker += now_checker

    overlap_area = (filters != 0).sum(-1) > 1
    overlap_value = filters * np.expand_dims(overlap_area, -1)
    max_idx = overlap_value.argmax(-1)
    overlap_area_idx = np.where(overlap_area != 0)
    temp = np.zeros_like(filters[overlap_area_idx])
    max_temp_idx = np.arange(len(max_idx[overlap_area_idx])), max_idx[overlap_area_idx]
    temp[max_temp_idx] = overlap_value[overlap_area_idx][max_temp_idx]
    filters[overlap_area_idx] = temp
    
    # miner_idx = np.where(filters <= 0.5)
    # filters[miner_idx] = 0
    filters_temp = filters_temp - np.expand_dims(filters.sum(-1), -1)
    filters = np.concatenate([filters_temp, filters], -1)
    filters = np.transpose(filters, (-1, 0, 1))
    others_info = np.transpose(others_info, (-1, 0, 1))
    return filters, others_info
