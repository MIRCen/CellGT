import os
import time
import PIL.Image as Image
import numpy as np
import torch
import torch_geometric
from torch_geometric.loader import DataLoader
import config as cfg
from data_management.data_preparation import read_aims_features
from data_management.data_preparation import get_list_ROI
from graph_processing.concave_hull import compute_distances_points
from utils import rearrange_labels_ROI_image

def create_dataloader_mice(mice_list, data_path, cfg=cfg, shuffle=False, batch_size=1, knn_nb=10
                      ):
    """
    Creates a PyTorch DataLoader for a list of mice datasets.

    Args:
        mice_list (list of str): List of mouse sample names.
        data_path (Path): Path to the root dataset directory.
        selected_features (list of str): Features to include from each param file.
        shuffle (bool): Whether to shuffle the dataset.
        batch_size (int): Batch size for the DataLoader.
        knn_nb (int): Number of nearest neighbors for KNN graph construction.

    Returns:
        DataLoader: PyTorch DataLoader containing graph-structured mouse data.
    """
    data_list=[]
    for mouse in mice_list:

        param_path = data_path /"derivatives"/ mouse / "neuron_features"
        roi_path = data_path /"derivatives"/ mouse / "ROI_ground_truth"
        data_list.extend(
            create_datalist(param_path, data_ROI_path=roi_path, selected_features=cfg.selected_features, knn_nb=knn_nb)
        )

    loader = DataLoader(data_list, num_workers=42, batch_size=batch_size, shuffle=shuffle)
    return loader


def create_datalist(data_params_path, data_ROI_path=None, selected_features=None, slice_numeros=range(15),
                      knn_nb=10):

    """
    Creates a list of torch_geometric `Data` objects from parameter and ROI image data.

    Args:
        data_params_path (str or Path): Path to the parameter files (.txt).
        data_ROI_path (str or Path, optional): Path to ROI mask images (.tif).
        selected_features (list of str): Features to extract from the parameter files.
        slice_numeros (iterable): Slice indices for contextual embedding.
        knn_nb (int): Number of neighbors for the KNN graph.

    Returns:
        list of torch_geometric.data.Data: Graph-structured data per slice.
    """

    # Load feature files
    params_urls = sorted([os.path.join(data_params_path, f) for f in os.listdir(data_params_path) if
                          (os.path.isfile(os.path.join(data_params_path, f)) and os.path.splitext(f)[1]=='.txt') and f!='Readme.txt'])
    list_features = read_aims_features(params_urls, selected_features=selected_features)

    # Load ROI images (if any)
    try :
        ROI_urls = sorted([os.path.join(data_ROI_path, f) for f in os.listdir(data_ROI_path) if
                           (os.path.isfile(os.path.join(data_ROI_path, f)) and os.path.splitext(f)[1]=='.tif')   ])
        if len(list_features) != len(ROI_urls):
            raise ValueError("the number of ROIs and features don't correspond")
        ROI_images = [np.array(Image.open(f)) for f in ROI_urls]
    except Exception as e:
        data_ROI_path = None
        print(f"Type d'erreur : {e}, No ROI available.")

    data_list = []
    for i, features in enumerate(list_features):
        data = torch_geometric.data.Data(num_workers=48)

        #add labels (if any)
        if data_ROI_path is not None:
            data.ROI_image = rearrange_labels_ROI_image(ROI_images[i])
            raw_x, raw_y = np.asarray(features[:, 0], dtype=int), np.asarray(features[:, 1], dtype=int)
            ratio = 1 / 0.03334 if any(x in str(data_ROI_path) for x in ["Gp1-S1", "Gp2-S4"]) else 50.0418
            labels = get_list_ROI(raw_x, raw_y, data.ROI_image.T, ratio=ratio)
            data.y = torch.tensor(labels, dtype=torch.long)

        # Build node features
        features = torch.tensor(features, dtype=torch.float)
        data.pos = features[:, :2]
        data.x = features[:, 2:]
        x, y = features[:, 0], features[:, 1]

        if 'mc_x' in selected_features:
            data.x = torch.cat([data.x, x.unsqueeze(1)], axis=1)
        if 'mc_y' in selected_features:
            data.x = torch.cat([data.x, y.unsqueeze(1)], axis=1)


        data.x = (data.x - torch.mean(data.x, dim=0)) / torch.sqrt(torch.var(data.x, dim=0)) #data normalization
        if "slice_numero" in selected_features:
            data.x = torch.cat((torch.ones((data.x.size(0), 1)) * slice_numeros[i]*0.2-torch.ones(data.x.size(0), 1), data.x), 1)

        # creation of graph edges
        KNNGraph_tranform = torch_geometric.transforms.KNNGraph(k=knn_nb, loop=False, force_undirected=True)
        data = KNNGraph_tranform.__call__(data)
        mean = (torch.max(data.pos, dim=0)[0] + torch.min(data.pos, dim=0)[0]) / 2
        data.pos_parameters = (mean, (torch.max(data.pos, dim=0)[0] - torch.min(data.pos, dim=0)[0]))
        data.pos = (data.pos - mean) / (torch.max(data.pos, dim=0)[0] - torch.min(data.pos, dim=0)[0])
        data.train_mask = torch.ones(data.x.size(0), dtype=torch.bool)
        data_list.append(data)
    return data_list


