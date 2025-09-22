#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
from collections import OrderedDict

import json
import os

if __name__ == "__main__":

    train_dir = r'F:\hj\project\nnUNet\dataset\nnUNet_raw\Task728_blood vessel\imagesTr'
    test_dir = r'F:\hj\project\nnUNet\dataset\nnUNet_raw\Task728_blood vessel\imagesTs'
    output_folder = r'F:\hj\project\nnUNet\dataset\nnUNet_raw\Task728_blood vessel'

    json_dict = OrderedDict()
    json_dict['name'] = "blood vessel"
    json_dict['description'] = "blood vessel"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MR"
        # "0": "MR"
    }

    json_dict['labels'] = {
        "0": "background",
        "1": "blood vessel"
    }

    file_name_list = os.listdir(train_dir)
    file_id_list = []
    for name in file_name_list:
        file_id_list.append(name.split('.nii.gz')[0])

    file_name_list = os.listdir(test_dir)
    file_name_test_ids = []
    for name in file_name_list:
        file_name_test_ids.append(name.split('.tif')[0])

    json_dict['numTraining'] = len(file_id_list)
    json_dict['training'] = [{'image': "./imagesTr/%s.tif" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             file_id_list]
    json_dict['numTest'] = 0
    json_dict['test'] = [{'test': "./imagesTs/%s.tif" % i} for i in file_name_test_ids]

    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)