import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import config as cfg
import time
from data_management.data_preparation import get_graph_from_range
import torch



def visualize_one_graph_from_data(data, pred, save_folder_name = None, image_number=0, image_name=None):
    coordinates_graph = data.pos
    color = pred.numpy()
    graph, coordinates_graph = get_graph_from_range(
        coordinates_graph[:, 0].numpy(), coordinates_graph[:, 1].numpy())

    graph.add_edges_from([(x.item(),y.item()) for x,y in zip(data.edge_index[0],data.edge_index[1])])

    plt.figure(figsize=(15, 10))
    plt.xticks([])
    plt.yticks([])

    color = [cfg.color_list[c] for c in color]
    node_size = np.full(len(color), 1)
    nx.draw_networkx(graph, pos=[[c[1], c[0]] for c in coordinates_graph], with_labels=False,
                         node_color=color, node_size=node_size)
    plt.gca().invert_yaxis()
    plt.savefig(save_folder_name.joinpath(str(image_name) +str(image_number)+".png"))
