import numpy as np
import torch
import config as cfg
from utils import distance



def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.12):
    """ Randomly perturb the point clouds by small rotations
        Input:
          array, original batch of point clouds and point normals
        Return:
          array, rotated batch of point clouds
    """
    rotated_data = batch_data.cpu().numpy()

    angles = np.clip(angle_sigma*np.random.randn(2), -angle_clip, angle_clip)
    theta = np.radians(angles[0])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    shape_pc = rotated_data[:,-2:]
    rotated_data[:,-2:] = np.dot(shape_pc.reshape((-1, 2)), R)
    return torch.tensor(rotated_data,dtype=torch.float).to(cfg.device)

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx2 array, original batch of point clouds
        Return:
          Nx2 array, jittered batch of point clouds
    """
    B, N = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N), -1*clip, clip)
    jittered_data = torch.tensor(jittered_data,dtype=torch.float)
    jittered_data += batch_data
    return jittered_data


def invert_point_cloud(data,class_number):

    if np.random.choice([0, 1]):
        #visualize_one_graph_from_data(data, [data.y])
        pos=data.pos.clone()
        pos[...,0]= pos[...,0].mean()-pos[...,0]
        data.pos=pos
        inverted_data_y=data.y.clone()
        mask_0= (inverted_data_y >= 0) & (inverted_data_y < ((class_number ) / 2))
        mask_1 = (inverted_data_y >= (class_number ) / 2) & (inverted_data_y <= (class_number ))
        inverted_data_y[mask_0]+=int((class_number ) / 2)
        inverted_data_y[mask_1]-=int((class_number ) / 2)
        data.y =inverted_data_y
        #visualize_one_graph_from_data(data,[data.y])

    return data




def random_point_dropout(data, max_dropout_ratio=0.7):
    dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
    keep_idx = np.where(np.random.random((data.x.shape[0])) > dropout_ratio)[0]
    if len(keep_idx)>0:
        data_sub=data.subgraph(torch.tensor(keep_idx).to(cfg.device))
    return data_sub



def random_cercle_area_dropout(data, max_dropout_cercle_number=5, max_radius_cercle=5000):
    dropout_cercle_number =  np.random.choice(range(max_dropout_cercle_number))
    radius_cercle = np.random.random()*max_radius_cercle

    for i in range(dropout_cercle_number):
        idx_center_dropout_cercle = np.random.choice(range(data.x.shape[0]))
        distance_map=[distance(data.pos[idx_center_dropout_cercle].cpu(),pos.cpu()) for pos in data.pos.cpu()]
        keep_idx = np.where(np.array(distance_map) > radius_cercle)[0]
        if len(keep_idx)>0:
            data=data.subgraph(torch.tensor(keep_idx).to(cfg.device))
    return data


def shift_point_cloud(batch, shift_range=0.05):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          Nx2 array, original batch of point clouds
        Return:
          BxNx2 array, shifted batch of point clouds
    """
    shifts = np.random.uniform(-shift_range, shift_range, (2))
    shifts = torch.tensor(shifts, dtype=torch.float)
    batch+= shifts
    return batch


def random_scale_point_cloud(batch_data, scale_low=0.85, scale_high=1.15):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx2 array, original batch of point clouds
        Return:
            Nx2 array, scaled batch of point clouds
    """
    scaled_data = batch_data.cpu().numpy()
    scales =np.random.uniform(scale_low, scale_high)
    scaled_data[:, -3:] = scaled_data[:, -3:]*scales
    return torch.tensor(scaled_data,dtype=torch.float).to(cfg.device)

