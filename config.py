#!/usr/bin/env python

from pathlib import Path
import torch




url = dict(
    project = r"./",
    param_path = r"./derivatives/sub-mouse1/neuron_features/",
    ROI_mask = r"./anatomical_region_ground_truth/"

    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Available device: {device}")



selected_features = ['slice_numero', 'area_mm2', 'min_radius', 'angle', 'isotropy', 'circularity',
                     'hsv_mean_hue', 'hsv_mean_saturation', 'hsv_mean_value', 'hsv_stand_dev_hue',
                     'hsv_stand_dev_saturation', 'hsv_stand_dev_value']



num_classes = 19

train = dict(
    mice_list=['mouse_Gp1-S1', 'mouse_Gp4-S1','mouse_Gp4-S2','mouse_Gp4-S3','mouse_Gp4-S4','mouse_Gp5-S3','mouse_Gp5-S4','mouse_Gp5-S5','mouse_Gp5-S6'],
    class_number = 19,
    n_epoch = 360,
    lr = 0.08,
    n_test=5,
    warmup_steps=8,
    step_size=20
)

cv = dict(
    mice_list=['mouse_Gp1-S1', 'mouse_Gp4-S1','mouse_Gp4-S2','mouse_Gp4-S3','mouse_Gp4-S4','mouse_Gp5-S3','mouse_Gp5-S4','mouse_Gp5-S5','mouse_Gp5-S6'],
    class_number = 8,
    n_epoch = 81,
    lr = 0.04,
    n_test=20,
    warmup_steps=10,
    step_size=20
)

color_list = ['#069AF3', '#069AF3', '#13EAC9', 'tab:orange', '#f6688e', 'y', 'tab:red',
                               '#89a0b0', '#a4a2fe', '#228b22', '#addffd', '#c7faf2', '#ffe1ab', '#fcc5d3', '#ffffb1', '#ff7e7e', '#d3dbe1',
                               '#d5d4ff', '#bdefbd', '#542E54']

