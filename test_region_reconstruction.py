import logging
import os
import time
from pathlib import Path

import torch
import argparse

from graph_processing.concave_hull import compute_masks_label
from graph_processing.graph_post_processing import  multi_polygon_to_image

import config as cfg
from data_management.dataloaders import create_dataloader_mice
from test_classification_model import inference_model
from utils import compute_classification_metrics
from models.CellGT import  CellGT
from models.GCN import  GCN
from models.GraphUnet import  GraphUNet
import matplotlib.pyplot as plt
import numpy as np
import sys

os.environ['TORCH'] = torch.__version__
logger = logging.getLogger(__name__)



def calculate_iou_image(pred_mask, true_mask, num_classes, sum_union,  sum_intersection):
    """
    Computes per-class intersection and union for a single image.

    Args:
        pred_mask (ndarray): Predicted segmentation mask.
        true_mask (ndarray): Ground truth segmentation mask.
        num_classes (int): Total number of classes.
        sum_union (ndarray): Array to accumulate union areas per class.
        sum_intersection (ndarray): Array to accumulate intersection areas per class.

    Returns:
        tuple: Updated sum_union and sum_intersection arrays.
    """
    h1, w1 = pred_mask.shape[:2]
    h2, w2 = true_mask.shape[:2]
    target_h = min(h1, h2)
    target_w = min(w1, w2)
    pred_mask = pred_mask[:target_h, :target_w]
    true_mask = true_mask[:target_h, :target_w]

    for class_id in range(num_classes//2):
        pred_class = np.where(pred_mask == class_id, 1, 0)
        true_class = np.where(true_mask == class_id, 1, 0)
        intersection = np.logical_and(pred_class, true_class)
        union = np.logical_or(pred_class, true_class)
        sum_union[class_id]+=union.sum()
        sum_intersection[class_id] += intersection.sum()
    return sum_union, sum_intersection



def calculate_iou_mouse(pred_mask_list, list_gt_mask, num_classes):
    """
    Computes IoU for a list of predicted and true masks.

    Args:
        pred_mask_list (list of ndarray): List of predicted masks.
        true_mask_list (list of ndarray): List of ground truth masks.
        num_classes (int): Number of classes.

    Returns:
        ndarray: IoU per class.
    """
    sum_union = np.zeros(num_classes)
    sum_intersection = np.zeros(num_classes)
    for pred_mask,gt_mask in zip(pred_mask_list, list_gt_mask):
        sum_union, sum_intersection = calculate_iou_image(pred_mask % 9, gt_mask % 9, num_classes,
                                                                     sum_union=sum_union,
                                                                     sum_intersection=sum_intersection)
    return sum_intersection/sum_union


def region_reconstruction(dataset):
    """
    Reconstructs polygon-based region masks from dataset predictions.

    Args:
        dataset (list): List of prediction objects with attributes `.pred`, `.pos`.

    Returns:
        tuple: List of Shapely geometries and corresponding rasterized images.
    """
    alpha, erosion, dilation, min_perimeter_holes, min_area_polygon = 0.017, -0.006, 0.009, 0.009, 0.0
    reconstructed_imgs=[]
    shapely_figures=[]
    for i, data_test in enumerate(dataset):
        if i in [3, 4]:
            masks_label = compute_masks_label(data_test.pred, alpha, data_test.pos, erosion=erosion,
                                      dilation=dilation)
            im_label = multi_polygon_to_image(masks_label, data_test)
            reconstructed_imgs.append(im_label)
            shapely_figures.append(masks_label)

    return shapely_figures, reconstructed_imgs



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
        default=r'/home/lm279992/projects/Cell-GT/CellGT_GP1-S1_epoch_105.pth',
        help='Path to the trained model checkpoint'
    )
    parser.add_argument(
        '-t', '--test',
        type=str,
        default='mouse_Gp4-S1',
        help='Mouse sample to test'
    )
    args = parser.parse_args()

    # ==== Paths and Config ====
    data_results = Path(__file__).parent / "results_test"
    test_name=f"Test_{args.type_network}_testmouse_{args.test}_{str(time.time())[-7:]}"
    save_folder_name = data_results / test_name
    save_folder_name.mkdir(parents=True, exist_ok=True)
    reconstructed_images_name=save_folder_name/"reconstructed images"
    reconstructed_images_name.mkdir(parents=True, exist_ok=True)
    selected_features = cfg.selected_features
    num_classes = 19


    # ==== Model Definitions ====
    if args.type_network=="Cell-GT":
        model = CellGT(len(selected_features) + 8, num_classes, dim_model=[32, 64, 128, 256], nb_convs=2, k=8).to(cfg.device)
        model.load_state_dict(torch.load(args.model, map_location=cfg.device))
    elif args.type_network=="GCN":
        model = GCN(len(selected_features) ,32,num_classes).to(cfg.device)
        #model.load_state_dict(torch.load(args.model, map_location=cfg.device))
    elif args.type_network=="GUNet":
        model = GraphUNet(len(selected_features),25,19,19).to(cfg.device)
        #model.load_state_dict(torch.load(args.model, map_location=cfg.device))
    model.eval()

    # ==== Dataloading ====
    test_loader = create_dataloader_mice([args.test], Path(args.data_path), selected_features=selected_features,shuffle=False,
                                         knn_nb=10)

    dataset_prediction= inference_model(model, test_loader.dataset)
    overall_accuracy, macro_f1, weighted_f1, f1_scores, accuracy_per_sample=compute_classification_metrics(dataset_prediction)
    shapely_figures, reconstructed_imgs=region_reconstruction(dataset_prediction)
    for i, image in enumerate(reconstructed_imgs):
        plt.imsave(reconstructed_images_name/("reconstructed_image_"+str(i)+".png"), image)

    ground_truth_masks = [data.ROI_image for data in test_loader.dataset]

    IoUs=calculate_iou_mouse(reconstructed_imgs, ground_truth_masks[3:5], num_classes)
    logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s   %(message)s",
                    handlers=[logging.FileHandler(save_folder_name.joinpath("test.log")),
                              logging.StreamHandler(sys.stdout)])
    logger.info(f"=============== Test name : {test_name} ===============\n")
    logger.info(f"=============== Model ===============")
    logger.info(f"{args.type_network} : {args.model}")
    logger.info(f"=============== Prediction ===============")
    for cls, IoU in enumerate(IoUs):
        logger.info(f"  Region {cls}: {IoU:.4f}")
































