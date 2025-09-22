# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
from collections import OrderedDict

import json
import os

if __name__ == "__main__":

    train_dir = r'E:\huangjiang\project\nnUNet\dataset\nnUNet_raw\Task314_fiber-bundle\imagesTr'
    test_dir =r'E:\huangjiang\project\nnUNet\dataset\nnUNet_raw\Task314_fiber-bundle\imagesTs'
    output_folder = r'E:\huangjiang\project\nnUNet\dataset\nnUNet_raw\Task314_fiber-bundle'

    json_dict = OrderedDict()
    json_dict['name'] = "Fiber Bundle"
    json_dict['description'] = "Fiber Bunder"
    json_dict['tensorImageSize'] = "2D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MR"
    }

    json_dict['labels'] = {
        "0": "background",
        "1": "fiber bundle"
    }

    # 处理训练目录的文件名
    file_name_list = os.listdir(train_dir)
    file_id_list = []
    for name in file_name_list:
        if name.endswith('.tif'):  # 确保只处理.tif格式的文件
            file_id_list.append(name.split('.tif')[0])

    # 处理测试目录的文件名
    file_name_list = os.listdir(test_dir)
    file_name_test_ids = []
    for name in file_name_list:
        if name.endswith('.tif'):  # 确保只处理.tif格式的文件
            file_name_test_ids.append(name.split('.tif')[0])

    json_dict['numTraining'] = len(file_id_list)
    json_dict['training'] = [{'image': "./imagesTr/%s.tif" % i, "label": "./labelsTr/%s.tif" % i} for i in file_id_list]  # 更新为.tif格式
    json_dict['numTest'] = len(file_name_test_ids)
    json_dict['test'] = [{'test': "./imagesTs/%s.tif" % i} for i in file_name_test_ids]  # 更新为.tif格式

    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)
