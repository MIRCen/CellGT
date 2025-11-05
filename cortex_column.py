import logging
import os
import time
from pathlib import Path
from warmup_scheduler_pytorch import WarmUpScheduler
import sys
import torch
import argparse
from torch_geometric.loader import DataLoader
from matplotlib.colors import ListedColormap

import config as cfg
import numpy as np

from data_management.dataloaders import create_dataloader_mice
from utils import compute_classification_metrics
from models.CellGT import  CellGT
from models.GCN import  GCN
from models.GraphUnet import  GraphUNet
from visualization import visualize_one_graph_from_data
os.environ['TORCH'] = torch.__version__
logger = logging.getLogger(__name__)





if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Evaluate a trained model on a specific test mouse dataset.")
    parser.add_argument(
        '-d', '--data_path',
        type=str,
        default=r"/home/Pbiopicsel/projects/TI2201_mouse-NeuN_G-Liot/30_dataset",
        help='Path to the dataset directory'
    )
    parser.add_argument(
        '-n', '--type_network',
        type=str,
        choices=['GCN', 'Cell-GT', 'GUNet'],
        default='Cell-GT',
        help='Type of model to test'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default=r'//home/Pbiopicsel/dataset/Mouse_NeuN_BIDS/data_BIDS/derivatives/classification_model/CellGT_GP1-S1_epoch_105.pth',
        help='Path to the trained model checkpoint'
    )
    parser.add_argument(
        '-t', '--test',
        type=str,
        default='sub-mouse1',
        help='Mouse sample to test'
    )
    args = parser.parse_args()

    
    # ==== Paths and Config ====
    data_results = Path(args.data_path)/ "results_test"
    data_results.mkdir(parents=True, exist_ok=True)
    name_results = f"Test_{args.type_network}_classification_{args.test}_{str(time.time())[-8:]}"
    save_folder_name = data_results / name_results
    save_folder_name.mkdir(parents=True, exist_ok=True)
    selected_features = cfg.selected_features
    num_classes = cfg.num_classes


    import time
    test_loader = create_dataloader_mice([args.test], Path(args.data_path), cfg=cfg,
                                         shuffle=False, knn_nb=10)
                                         
                                         
    data=test_loader.dataset[4]
    #visualize_one_graph_from_data(test_loader.dataset[3], [test_loader.dataset[3].y])

    node_mask = data.y == 10
    data_sub=data.subgraph(torch.tensor(node_mask))
    #visualize_one_graph_from_data(data_sub, [data_sub.y])
    import math
    
    coords=data_sub.pos
    angles = torch.atan2(data_sub.pos[:, 1], data_sub.pos[:, 0])
    print(angles) 
    num_bins = 50
    bin_edges = torch.linspace(- torch.pi/2, torch.pi/2, num_bins + 1)
    bin_indices = torch.bucketize(angles, bin_edges, right=False) - 1
    bin_indices = bin_indices.clamp(0, num_bins - 1)
    partitions = [coords[bin_indices == i] for i in range(num_bins)]
    import matplotlib.pyplot as plt
    
    n_cols = math.ceil(math.sqrt(num_bins))
    n_rows = math.ceil(num_bins / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    axes = axes.flatten()  # make it iterable

    cmap = plt.get_cmap('hsv', num_bins)

    for i in range(num_bins):
     ax = axes[i]
     part = partitions[i]
     if len(part) > 0:
        ax.scatter(part[:, 0], part[:, 1], s=10, color=cmap(i))
     ax.set_title(f"Bin {i}")
     ax.set_aspect('equal')
     ax.axis('off')

    # Hide unused subplots (if any)
    for j in range(num_bins, len(axes)):
     axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
    
    
    
    print(data_sub.pos)





























