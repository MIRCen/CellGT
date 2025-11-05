import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import config as cfg
import time
from data_management.data_preparation import get_graph_from_range
import torch

def colored(c):
                 #Cortex & HP &         TH &         HT&    ST&     AMG&    GP&         MD&         PO
    #color_list=[ 'b','#069AF3','#13EAC9', 'tab:orange', '#f6688e','y','tab:red','#89a0b0', '#a4a2fe','#228b22', '#addffd','#c7faf2','#ffe1ab','#fcc5d3','#ffffb1','#ff7e7e','#d3dbe1','#d5d4ff','#bdefbd','#542E54']
    color_list = ['#069AF3', '#069AF3', '#13EAC9', 'tab:orange', '#f6688e', 'y', 'tab:red',
                               '#89a0b0', '#a4a2fe', '#228b22', '#addffd', '#c7faf2', '#ffe1ab', '#fcc5d3', '#ffffb1', '#ff7e7e', '#d3dbe1',
                               '#d5d4ff', '#bdefbd', '#542E54']
    #color_list=["green","red","white","tomato", "lightgreen",]
    return color_list[c]

def visualize_graph(G, coordinates, color):
    plt.figure(figsize=(15, 10))
    color = [colored(c) for c in color]
    plt.xticks([])
    plt.yticks([])
    node_size=np.full(len(color),14)
    nx.draw_networkx(G, pos=[[c[1],c[0]] for c in coordinates], with_labels=False,
                     node_color=color, cmap="gist_ncar", node_size= node_size)
    plt.gca().invert_yaxis()
    plt.show()


def save_graph(G, coordinates, color, path_folder, slice_number):
    plt.figure(figsize=(15, 10))
    color = [colored(c) for c in color]
    #color=color.numpy()
    plt.xticks([])
    plt.yticks([])
    node_size=np.full(len(color),3)
    nx.draw_networkx(G, pos=[[c[1],c[0]] for c in coordinates], with_labels=False,
                     node_color=color,  node_size= node_size)
    plt.gca().invert_yaxis()
    plt.savefig(str(time.time())+"_image_"+str(slice_number)+".png")



def visualize_superposed_graph_image(G, coordinates, img, color=None):
    y_lim = img.shape[0]
    x_lim = img.shape[1]
    extent = 0, x_lim, 0, y_lim
    color = [colored(c) for c in color]

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, extent=extent, interpolation='nearest')
    nx.draw_networkx(G, pos=[[c[1], c[0]] for c in coordinates], with_labels=False,
                     node_color=color, cmap="Set2", node_size=3, width=0.2)
    plt.show()

def visualize_superposed_region_image(G, coordinates, img, color=None):
    # img=reshape_ROI(img)
    y_lim = img.shape[0]
    x_lim = img.shape[1]
    extent = 0, x_lim, 0, y_lim
    color = [colored(c) for c in color]

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, extent=extent, interpolation='nearest')
    nx.draw_networkx(G, pos=[[c[1], c[0]] for c in coordinates], with_labels=False,
                     node_color=color, cmap="Set2", node_size=2, width=0.2)
    plt.show()

def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(12, 7))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()

def visualize_one_graph_from_data(data, pred, node_max=None, image=None, save_folder_name = None, image_number=0, image_name=None):
    if node_max is not None:
        coordinates_graph = data.pos[:node_max]
    else:
        coordinates_graph = data.pos

    color = pred[0].numpy()
    #color = torch.ones_like(color)
    graph, coordinates_graph = get_graph_from_range(
        coordinates_graph[:, 0].numpy(), coordinates_graph[:, 1].numpy())

    graph.add_edges_from([(x.item(),y.item()) for x,y in zip(data.edge_index[0],data.edge_index[1])])
    if save_folder_name is None:
        visualize_graph(graph, coordinates_graph,color)

    elif image is not None:
        img = np.flip(np.asarray(image), axis=0)
        visualize_superposed_graph_image(graph, coordinates_graph, img, color=color)
    else:
        #save_graph(graph, coordinates_graph,color, save_folder_name,image_number)

        plt.figure(figsize=(15, 10))
        plt.xticks([])
        plt.yticks([])
        color = [colored(c) for c in color]
        node_size = np.full(len(color), 1)
        nx.draw_networkx(graph, pos=[[c[1], c[0]] for c in coordinates_graph], with_labels=False,
                         node_color=color, node_size=node_size)
        plt.gca().invert_yaxis()
        #plt.show()
        plt.savefig(save_folder_name.joinpath(str(image_name) +".png"))


def visualize_from_dataloader(test_loader, pred, images=None, save_folder_name=None):

    for p in range(len(pred)):
        visualize_one_graph_from_data(test_loader.dataset[p].cpu(), pred, save_folder_name=save_folder_name.joinpath("test_images"), image_number=p)#visualize_one_graph_from_data(test_loader.dataset[p].cpu(), pred)


def show_plot_list(list_metrics, name, avg=20, test_epoch=20):
    import time
    fig = plt.figure()
    avg_list_metrics = []
    time = time.time()
    for list_value in list_metrics:
        avg_list_value = [np.mean(list_value[i:i + avg]) for i in range(0, len(list_value) - avg, avg)]
        plt.plot([i * test_epoch for i in range(0, len(list_value) - avg, avg)], avg_list_value)
        avg_list_metrics.append(avg_list_value)
    plt.title("Curve plotted using the given points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(cfg.url['feature_selection'] + name + ".png")
    np.savetxt(cfg.url['feature_selection'] + name + ".txt", list_metrics, fmt='%.3f')
