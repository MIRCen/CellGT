import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin

from torch_cluster import knn_graph
from torch_cluster import fps
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_mean_pool, knn_interpolate
from torch_geometric.nn.conv import PointTransformerConv
from torch_geometric.nn.pool import knn
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch.nn import Linear, ReLU, LayerNorm

from torch_scatter import scatter_max
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.utils import to_dense_batch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.PEGNN import GridCellSpatialRelationEncoder


pre_transform = T.NormalizeScale()

class TransitionDown(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''

    def __init__(self, in_channels, out_channels, ratio=1/6, k=8):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False, norm='graph_norm')

    def forward(self, x, pos, batch):

        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out, _ = scatter_max(x[id_k_neighbor[1]], id_k_neighbor[0],
                               dim_size=id_clusters.size(0), dim=0)

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class TransitionUp(torch.nn.Module):
    '''
        Reduce features dimensionnality and interpolate back to higher
        resolution and cardinality
    '''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = MLP([in_channels, out_channels], plain_last=False, norm='graph_norm')
        self.mlp = MLP([out_channels, out_channels], plain_last=False, norm='graph_norm')

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=2,#3
                                         batch_x=batch_sub, batch_y=batch)
        x = self.mlp(x) + x_interpolated

        return x




class CellGT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, nb_convs=2, act=F.relu, k=10, emb_hidden_dim=32,
                 emb_dim=8):
        super().__init__()
        self.k = k
        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        self.act = act
        self.nb_convs = nb_convs

        self.spenc = GridCellSpatialRelationEncoder(spa_embed_dim=emb_hidden_dim, ffn=True, min_radius=1e-06,
                                                    max_radius=8)
        self.dec = nn.Sequential(
            nn.Linear(emb_hidden_dim, emb_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 2, emb_dim)
        )

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False,dropout=0.02, norm='graph_norm')

        # backbone layers
        self.down_convs = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()

        self.input_convs = torch.nn.ModuleList()
        num_layers_input = 2
        for i in range(1, num_layers_input + 1):
            conv = GENConv(dim_model[0], dim_model[0], aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            self.input_convs.append(DeepGCNLayer(conv, LayerNorm(dim_model[0], elementwise_affine=True), ReLU(inplace=True), block='res+', dropout=0.04,
                                 ckpt_grad=i % 3))

        self.output_convs = torch.nn.ModuleList()
        num_layers_output = 4
        for i in range(1, num_layers_output + 1):
            conv = GENConv(dim_model[0], dim_model[0], aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            self.output_convs.append(DeepGCNLayer(conv, LayerNorm(dim_model[0], elementwise_affine=True), ReLU(inplace=True), block='res+', dropout=0.04,
                                 ckpt_grad=i % 3))


        #self.input_convs = torch.nn.ModuleList().append(DeepLayers(4, dim_model[0]))

        self.transformers_up = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        self.transition_up = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()
        self.list_k=[10,6,3,2]

        for i in range(0, self.nb_convs):
            # Add Transition Down block followed by a Point Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.list_k[i]))
            #print(self.list_k[i])

            conv = GENConv(dim_model[i + 1], dim_model[i + 1], aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            self.down_convs.append(torch.nn.ModuleList().append(DeepGCNLayer(conv, LayerNorm(dim_model[i + 1], elementwise_affine=True), ReLU(inplace=True), block='res+', dropout=0.04,
                                 ckpt_grad=i % 3)).append(DeepGCNLayer(conv, LayerNorm(dim_model[i + 1], elementwise_affine=True), ReLU(inplace=True), block='res+', dropout=0.04,
                                 ckpt_grad=i % 3)))

            #self.down_convs.append(DeepLayers(4, dim_model[i + 1]))

            # Add Transition Up block followed by Point Transformer block
            self.transition_up.append(
                TransitionUp(in_channels=dim_model[i + 1],
                             out_channels=dim_model[i]))

            conv = GENConv(dim_model[i], dim_model[i], aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')

            self.up_convs.append(torch.nn.ModuleList().append(DeepGCNLayer(conv, LayerNorm(dim_model[i], elementwise_affine=True), ReLU(inplace=True), block='res+', dropout=0.05,
                                 ckpt_grad=i % 3)).append(DeepGCNLayer(conv, LayerNorm(dim_model[i], elementwise_affine=True), ReLU(inplace=True), block='res+', dropout=0.05,
                                 ckpt_grad=i % 3)))

            #self.up_convs.append(DeepLayers(1, dim_model[i]))

        for i in range(self.nb_convs, len(dim_model) - 1):
            # Add Transition Down block followed by a Point Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.list_k[i]))
            #print(self.list_k[i])

            self.transformers_down.append(nn.TransformerEncoder(nn.TransformerEncoderLayer(
                d_model=dim_model[i + 1], nhead=2, dropout=0.01, batch_first=True), num_layers=2))

            # Add Transition Up block followed by Point Transformer block
            self.transition_up.append(
                TransitionUp(in_channels=dim_model[i + 1],
                             out_channels=dim_model[i]))

            self.transformers_up.append(nn.TransformerEncoder(nn.TransformerEncoderLayer(
                d_model=dim_model[i], nhead=2, dropout=0.01, batch_first=True), num_layers=2))

        # summit layers
        self.mlp_summit = MLP([dim_model[-1], dim_model[-1]], norm=None,
                              plain_last=False, dropout=0.01)

        self.transformer_summit = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=dim_model[-1], nhead=2, dropout=0.01, batch_first=True), num_layers=4)
        # self.output_convs = torch.nn.ModuleList().append(DeepLayers(2, dim_model[0]))

        # class score computation
        self.mlp_output = MLP([dim_model[0], out_channels], dropout=0.2
                              , norm='graph_norm')

    def forward(self, data):
        x, c, batch = data.x, data.pos, data.batch
        pos = c
        c = c.float()
        c = c.reshape(1, c.shape[0], c.shape[1])

        emb = self.spenc(c.detach().cpu().numpy())
        emb = emb.reshape(emb.shape[1], emb.shape[2])
        emb = self.dec(emb).float()
        x = torch.cat((x, emb), dim=1)

        out_x = []
        out_pos = []
        out_batch = []

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        #edge_weight = x.new_ones(edge_index.size(1))

        x = self.input_convs[0].conv(x, edge_index)
        for layer in self.input_convs[1:]:
            x = layer(x, edge_index)

        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)

        # backbone down : #reduce cardinality and augment dimensionnality
        #print("debut_down")
        for i in range(len(self.transition_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.list_k[i+1], batch=batch)

            if i in range(self.nb_convs):
                edge_weight = x.new_ones(edge_index.size(1))

                d_c=self.down_convs[i]
                x = d_c[0](x, edge_index).relu() #relu
                x = d_c[1](x, edge_index).relu() #relu
            else:
                x, mask = to_dense_batch(x, batch)
                x = self.transformers_down[i - self.nb_convs](x)
                x = x[mask]

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)

        #print("fin_down")
        # summit
        x, mask = to_dense_batch(x, out_batch[-1])
        x = self.mlp_summit(x)
        x = self.transformer_summit(x)
        x = x[mask]

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.transition_down)
        for i in range(0, n - self.nb_convs):
            x = self.transition_up[-i - 1](x=out_x[-i - 2], x_sub=x,
                                           pos=out_pos[-i - 2],
                                           pos_sub=out_pos[-i - 1],
                                           batch_sub=out_batch[-i - 1],
                                           batch=out_batch[-i - 2])

            x, mask = to_dense_batch(x, out_batch[-i - 2])
            x = self.transformers_up[-i - 1](x)
            x = x[mask]



        for i in range(n - self.nb_convs, n):
            x = self.transition_up[-i - 1](x=out_x[-i - 2], x_sub=x,
                                           pos=out_pos[-i - 2],
                                           pos_sub=out_pos[-i - 1],
                                           batch_sub=out_batch[-i - 1],
                                           batch=out_batch[-i - 2])

            edge_index = knn_graph(out_pos[-i - 2], k=self.list_k[-i-2],
                                   batch=out_batch[-i - 2])
            #print(f"self.list_k[i] {self.list_k[-i-2]}")

            u_c = self.up_convs[-i + n - self.nb_convs - 1]
            x = u_c[0](x, edge_index).relu()  # relu
            x = u_c[1](x, edge_index) # relu
            x = self.act(x) if i < n - 1 else x
        # Class score
        x = self.output_convs[0](x, edge_index)
        for layer in self.output_convs[1:]:
            x = layer(x, edge_index)
        out = self.mlp_output(x)


        return out

