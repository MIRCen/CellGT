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
os.environ['TORCH'] = torch.__version__
logger = logging.getLogger(__name__)



@torch.no_grad()
def inference_model(model, dataset):
    """
    Perform inference on a subset of the dataset using the provided model.

    Args:
        model (torch.nn.Module): The trained model used for inference.
        dataset (list): List of data samples to run inference on.

    Returns:
        list: Dataset with added predicted labels (`data_test.pred`).
    """
    with torch.no_grad():
        for i, data_test in enumerate(dataset):
                out = model(data_test.to(cfg.device))
                data_test.pred=out.argmax(dim=1)
                data_test.cpu()
                del out
    return dataset


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


    # ==== Model Definitions ====
    if args.type_network=="Cell-GT":
        model = CellGT(len(selected_features) + 8, num_classes, dim_model=[32, 64, 128, 256], nb_convs=2, k=8).to(cfg.device)
        model.load_state_dict(torch.load(args.model, map_location=cfg.device))
    if args.type_network=="GCN":
        model = GCN(len(selected_features) ,32,num_classes).to(cfg.device)
        #model.load_state_dict(torch.load(args.model, map_location=cfg.device))
    if args.type_network=="GUNet":
        model = GraphUNet(len(selected_features),25,19,19).to(cfg.device)
        #model.load_state_dict(torch.load(args.model, map_location=cfg.device))
    model.eval()
    import time
    test_loader = create_dataloader_mice([args.test], Path(args.data_path), cfg=cfg,
                                         shuffle=False, knn_nb=10)

    dataset_prediction= inference_model(model, test_loader.dataset)
    torch.save(dataset_prediction, save_folder_name / ('dataset_prediction_'+args.test+'.pt'))
    overall_accuracy, macro_f1, weighted_f1, f1_scores, accuracy_per_sample=compute_classification_metrics(dataset_prediction)
    # Print results
    logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s   %(message)s",
                    handlers=[logging.FileHandler(save_folder_name.joinpath("test.log")),
                              logging.StreamHandler(sys.stdout)])
    logger.info(f"=============== Test name : {name_results} ===============\n")
    logger.info(f"=============== Model ===============")
    logger.info(f"{args.type_network} : {args.model}")
    logger.info(f"=============== Prediction ===============")
    logger.info(f"Url : {save_folder_name / ('dataset_prediction_'+args.test+'.pt')}")
    logger.info(f"=============== Performing test: {args.test} ===============")
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logger.info("Accuracy per data sample:")
    for idx, acc in enumerate(accuracy_per_sample):
        logger.info(f"  Slice {idx}: {acc:.4f}")
    logger.info("F1-score per class : \n")
    for cls, f1 in enumerate(f1_scores):
        logger.info(f"  Region {cls}: {f1:.4f}")
    logger.info(f"Macro F1-score: {macro_f1:.4f}")
    logger.info(f"Weighted F1-score: {weighted_f1:.4f}")



























