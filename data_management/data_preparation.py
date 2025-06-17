import warnings

import networkx as nx
import numpy as np
import torch
import pandas as pd
import config as cfg
from utils import read_image

warnings.filterwarnings("ignore")



def get_list_ROI(list_centroids_X, list_centroids_Y, ROI_img, ratio= 50.0418):
    """
    project point coordinatinates to ROI_img to get point labels

    :param list_centroids_X:
    :param list_centroids_Y:
    :param ROI_img:
    :param image_size:
    :return: list of labels
    """
    ROI_image_size = ROI_img.shape
    list_cell_ROI = []

    for centroid_X, centroid_Y in zip(list_centroids_X, list_centroids_Y):
        list_cell_ROI.append(ROI_img[min(int(round(centroid_X /ratio)),ROI_image_size[0]-1), min(int(round(centroid_Y  /ratio)),ROI_image_size[1]-1)])
    return np.asarray(list_cell_ROI)

def get_graph_from_range(list_centroids_X, list_centroids_Y, angle_edges=False):
    G = nx.Graph()
    G.add_nodes_from(range(len(list_centroids_X)))
    coordinates = np.column_stack((list_centroids_Y, list_centroids_X))
    return G, coordinates



def read_aims_features(image_urls, selected_features):
    """
    :param image_urls:
    :param selected_features:
    :return:
    """
    pd.set_option('display.max_columns', None)
    exclude = {"slice_numero", "mc_x", "mc_y"}
    filtered_features = [f for f in selected_features if f not in exclude]
    selected_features_csv = ["mc_x", "mc_y"] + filtered_features
    list_features=[]
    for url in image_urls:
        try:
            data = pd.read_csv(url, sep =';')
            data = data[selected_features_csv]
        except KeyError as e:
            raise KeyError(f"Missing expected feature(s) in {url}: {e}")
        list_features.append(data.to_numpy(dtype=float))
    return list_features


