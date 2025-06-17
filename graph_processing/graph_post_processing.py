import torch
import torch_geometric

from torch_geometric.utils import to_scipy_sparse_matrix, k_hop_subgraph
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_scipy_sparse_matrix, k_hop_subgraph
import numpy as np
import cv2

def multi_polygon_to_image_2(mask_label, test, width=600, height=850, ratio=50.0418, label=1):
    mask_image = np.zeros((width, height), dtype=np.uint8)
    m, e = test.pos_parameters
    for label, mask in enumerate(mask_label):
        if not mask.is_empty:
            for polygon in mask.geoms:
                coordinates = [(int((x * e[0] + m[0]) / ratio), int((y * e[1] + m[1]) / ratio)) for x, y in
                               list(polygon.exterior.coords)]
                cv2.fillPoly(mask_image, [np.array(coordinates)], label)
    return mask_image  # mask_image

def multi_polygon_to_image(mask_label,test, width=600, height=900, ratio=50.0418, label=1):
    mask_image = np.zeros((width, height), dtype=np.uint8)
    m, e = test.pos_parameters
    for label, mask in enumerate(mask_label):
        if not mask.is_empty:
            for polygon in mask.geoms:
                coordinates = [(int((x * e[0] + m[0]) / ratio), int((y * e[1] + m[1]) / ratio)) for x, y in
                               list(polygon.exterior.coords)]
                cv2.fillPoly(mask_image, [np.array(coordinates)], label)
    return mask_image  # mask_image



def post_processing_graph(data, cluster_ids_x, nb_labels=19, num_components_selected=2, nb_min_cluster=0):
    new_cluster_ids_x = cluster_ids_x
    index_distance_map = np.where(data.features_name == "distance_map")[0]
    distance_map_cells = data.x[:, index_distance_map]
    data_knn = torch_geometric.data.Data(num_workers=48)
    data_knn.x=data.x
    data_knn.pos = data.pos
    data_knn.edge_attr =None
    data_knn.edge_index = None
    KNNGraph_tranform = torch_geometric.transforms.KNNGraph(k=3, loop=False, force_undirected=True)
    data_knn = KNNGraph_tranform.__call__(data)


    for m in range(0,nb_labels):
        node_index_label, sub_label_graph = subgraph_label(data_knn, cluster_ids_x, m)
        adj = to_scipy_sparse_matrix(sub_label_graph.edge_index, num_nodes=sub_label_graph.num_nodes)
        num_components, component = sp.csgraph.connected_components(adj, directed=False)
        _, count = np.unique(component, return_counts=True)
        idx_in_considered_clusters = count.argsort()[:-num_components_selected]
        idx_in_considered_clusters = [idx for idx in idx_in_considered_clusters if count[idx] > 0]
        subset_bool = np.in1d(component, idx_in_considered_clusters)
        subset, = np.where(subset_bool)
        update_new_cluster_ids_x(data_knn, new_cluster_ids_x, cluster_ids_x, node_index_label[subset], m,
                                 distance_map_cells)
    return new_cluster_ids_x


def subgraph_label(data, cluster_ids_x, m):
    filtered_node_index_label = torch.where(cluster_ids_x == m, True, False)
    keep_idx = torch.nonzero(filtered_node_index_label)
    sub_label_graph = data.subgraph(keep_idx[:,0])
    return keep_idx, sub_label_graph


def update_new_cluster_ids_x(data, new_cluster_ids_x, cluster_ids_x, subset, m, distance_map_cells, depth=10,
                             minimum_distance=-1.2):
    for cell in subset:
        #if distance_map_cells[cell] > minimum_distance:
            #Computes the induced subgraph of edge_index around all nodes in node_idx reachable within  hops.
            neigh_subset, edge_index, mapping, edge_mask = k_hop_subgraph(cell.item(), depth, data.edge_index,
                                                                          relabel_nodes=False)
            unique_labels, count = np.unique(cluster_ids_x[neigh_subset], return_counts=True)
            if not (len(unique_labels) == 1 and unique_labels[0] == m):
                i, = np.where(unique_labels == m)
                unique_labels = np.delete(unique_labels, i)
                count = np.delete(count, i)
                max_l = np.argmax(count)
                new_cluster_ids_x[cell] = unique_labels[max_l]
