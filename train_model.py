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
import torch_geometric.transforms as T
from data_management.data_augmentation import invert_point_cloud
from data_management.dataloaders import create_dataloader_mice
from test_classification_model import inference_model
from utils import compute_classification_metrics
from models.CellGT import  CellGT
from models.GCN import  GCN
from models.GraphUnet import  GraphUNet
from torch.optim.lr_scheduler import StepLR
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
        default=r'/home/lm279992/projects/Cell-GT/CellGT_GP1-S_epoch_105.pth',
        help='Path to the trained model checkpoint'
    )
    parser.add_argument(
        '--train_mice',
        nargs='+',
        type=str,
        default=['sub-mouse1'],# 'mouse_Gp5-S2', 'mouse_Gp5-S3', 'mouse_Gp5-S4', 'mouse_Gp5-S5','mouse_Gp4-S3'],
        help='List of training mouse sample names'
    )

    parser.add_argument(
        '--test_mice',

        type=str,
        default='sub-mouse1',
        help='List of test mouse sample names'
    )

    args = parser.parse_args()

    # ==== Paths and Config ====


    # ==== Paths and Config ====
    data_results = Path(args.data_path)/ "Results"
    test_name=f"Train_{args.type_network}_testmouse_{args.test_mice}_{str(time.time())[-7:]}"
    save_folder_name = data_results / test_name
    save_folder_name.mkdir(parents=True, exist_ok=True)
    reconstructed_images_name=save_folder_name/"reconstructed images"
    reconstructed_images_name.mkdir(parents=True, exist_ok=True)
    model_name = save_folder_name / "models"
    model_name.mkdir(parents=True, exist_ok=True)
    selected_features = cfg.selected_features
    num_classes = cfg.num_classes


    # ==== Model Definitions ====
    if args.type_network=="Cell-GT":
        model = CellGT(len(selected_features) + 8, num_classes, dim_model=[32, 64, 128, 256], nb_convs=2, k=8).to(cfg.device)

    elif args.type_network == "GCN":
        model = GCN(len(selected_features), 32, num_classes).to(cfg.device)

    elif args.type_network == "GUNet":
        model = GraphUNet(len(selected_features), 25, 19, 19).to(cfg.device)

    try:
        model.load_state_dict(torch.load(args.model, map_location=cfg.device))
    except Exception as e:
            print(f"Warning: Start training from scratch")
    model_name = model.__class__.__name__

    print("Training mice:", args.train_mice)
    print("Test mice:", args.test_mice)

    # ==== data loading ====

    test_loader = create_dataloader_mice([args.test_mice], Path(args.data_path), cfg=cfg,
                                         shuffle=False, knn_nb=10)
    train_loader = create_dataloader_mice(args.train_mice, Path(args.data_path),
                                          cfg=cfg,
                                          shuffle=True, knn_nb=10, batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train['lr'])  # , weight_decay=0.1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    logger.info(f"=============== Performing training, training: {args.train_mice}, test: {args.test_mice} ===============")
    logger.info(f"Selected_model: {args.model}")
    logger.info(f"lr: {cfg.train['lr']}, epochs: {cfg.train['n_epoch']}, step_size: {cfg.train['step_size']}, warmup_steps: {cfg.train['warmup_steps']}")


    transform = T.Compose([
        T.RandomRotate(5),
        T.RandomShear(0.05),
        T.RandomScale((0.98, 1.08))])

    for epoch in range(1, cfg.train['n_epoch']):
            torch.cuda.empty_cache()
            if epoch % cfg.train['n_test'] == 0:
                torch.cuda.empty_cache()

                dataset_test = inference_model(model, test_loader.dataset)
                overall_accuracy, macro_f1, weighted_f1, f1_scores, accuracy_per_sample = compute_classification_metrics(
                    dataset_test)
                dataset_training= inference_model(model, train_loader.dataset)
                torch.save(model,save_folder_name.joinpath("models", str(model_name) + "_epoch_" + str(epoch) + ".pth"))
                # Print results
                logging.basicConfig(level=logging.INFO,
                                    format="%(asctime)s   %(message)s",
                                    handlers=[logging.FileHandler(save_folder_name.joinpath("test.log")),
                                              logging.StreamHandler(sys.stdout)])
                logger.info(f"{args.type_network} : {args.model}")
                logger.info(f"=============== Prediction ===============")
                logger.info(f"Url : {save_folder_name / ('dataset_prediction_' + args.test_mice + '.pt')}")
                logger.info(f"=============== Performing test: {args.test_mice} ===============")
                logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
                logger.info("Accuracy per data sample:")
                for idx, acc in enumerate(accuracy_per_sample):
                    logger.info(f"  Slice {idx}: {acc:.4f}")
                logger.info("F1-score per class : \n")
                for cls, f1 in enumerate(f1_scores):
                    logger.info(f"  Region {cls}: {f1:.4f}")
                logger.info(f"Macro F1-score: {macro_f1:.4f}")
                logger.info(f"Weighted F1-score: {weighted_f1:.4f}")

            total_loss = 0
            torch.cuda.empty_cache()

            for batch_idx, data in enumerate(train_loader):
                torch.cuda.empty_cache()
                model.train()
                data_train = invert_point_cloud(data, cfg.train['class_number'])
                data_train = transform(data_train).to(cfg.device)
                out = model(data_train) #Perform a single forward pass.
                loss = criterion(out, data_train.y)  # Compute the loss solely based on the training nodes.

                loss.backward()  # Derive gradients.

                optimizer.step()  # Update parameter s based on gradients.
                optimizer.zero_grad()
                total_loss += loss.item()  # * data.num_graphs
                del out, data_train, loss
                torch.cuda.empty_cache()
            scheduler.step()
            logger.info(f"Epoch : {epoch} lr : {optimizer.param_groups[0]['lr']}, loss : {total_loss}")


































