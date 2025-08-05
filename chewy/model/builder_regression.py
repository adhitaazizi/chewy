# Copyright (c) 2025, Chunan Liu and Aurélien Plüsser
# All rights reserved.
#
# This source code is licensed under the CC BY-NC-SA 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import wandb
from loguru import logger

from torch import Tensor
from torch_geometric.data import Batch as PygBatch
from torch.optim import AdamW
from flash.core.optimizers import LinearWarmupCosineAnnealingLR

from chewy.model.encoder import ChewyEncoder
from chewy.model.components.graph_regressor import GraphRegressor

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette("Set2")

def accuracy_metric(pred_rank_label: Tensor, true_rank_label: Tensor) -> Tensor:
    """Calculate accuracy for ranking predictions."""
    return (pred_rank_label == true_rank_label).float().mean()

# Regression model: from WALLE-Affinity with modified encoder
class ChewyRegression(L.LightningModule):
    def __init__(self, num_steps, warm_up_epochs, scheduler=None):
        super().__init__()
        # store the loss for each batch in an epoch
        self.training_loss_epoch = []
        self.training_pred_labels_epoch = []
        self.training_true_labels_epoch = []
        self.validation_loss_epoch = []
        self.validation_pred_labels_epoch = []
        self.validation_true_labels_epoch = []
        self.test_pred_labels_epoch = (
            {}
        )  # e.g. {"generalization": [], "perturbation": []} for multiple test sets
        self.test_true_labels_epoch = (
            {}
        )  # e.g. {"generalization": [], "perturbation": []} for multiple test sets
        self.test_loss_epoch = (
            {}
        )  # e.g. {"generalization": [], "perturbation": []} for multiple test sets
        self.test_metric_epoch = (
            {}
        )  # e.g. {"generalization": [], "perturbation": []} for multiple test sets
        self.test_results = {}  # for storing test results
        
        self.num_steps = num_steps
        self.warm_up_epochs = warm_up_epochs
        self.scheduler = scheduler

        logger.info("Instantiating encoder blocks ...")
        self.B_encoder_block = ChewyEncoder(
                                    node_feat_name="x_b", 
                                    edge_index_name="edge_index_b",
                                    input_dim=512,
                                    dim_list=[512, 512, 512, 512, 512, 512, 512, 128, 64],
                                    heads=4,
                                    dropout=0.3
                                )
        self.G_encoder_block = ChewyEncoder(
                                    node_feat_name="x_b", 
                                    edge_index_name="edge_index_b",
                                    input_dim=1280,
                                    dim_list=[1280, 1280, 1280, 1280, 1280, 1280, 1280, 128, 64],
                                    heads=4,
                                    dropout=0.3
        )
        logger.info(self.B_encoder_block)
        logger.info(self.G_encoder_block)

        logger.info("Instantiating regression component ...")
        self.regressor = GraphRegressor()
        logger.info(self.regressor)

        logger.info("Instantiating losses...")
        self.loss_func_dict = self.configure_loss_func_dict()
        logger.info(f"Using losses: {self.loss_func_dict}")

        logger.info("Instantiating metrics...")
        self.metric_func_dict = self.configure_metric_func_dict()
        logger.info(f"Using metrics: {self.metric_func_dict}")

        self.save_hyperparameters()  # add config to self.hparams

    def encode(self, batch: PygBatch) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch: (PygBatch) batched data returned by PyG DataLoader
        Returns:
            B_z: (Tensor) shape (Nb, C)
            G_z: (Tensor) shape (Ng, C)
        """
        B_z = self.B_encoder_block(
            batch.x_b, batch.edge_index_b
        )  # (batch.x_b.shape[0], C), e.g.  C=64 depends on the config
        G_z = self.G_encoder_block(
            batch.x_g, batch.edge_index_g
        )  # (batch.x_g.shape[0], C), e.g.  C=64 depends on the config

        return B_z, G_z
    
    def forward(self, batch: PygBatch) -> Tensor:
        # encode
        z_ab, z_ag = self.encode(batch)
        # regression
        affinity_pred = self.regressor(z_ab, z_ag, batch)
        affinity_pred = torch.clamp(affinity_pred, -12, 12)
        return affinity_pred

    # --------------------------------------------------------------------------
    # Configure
    # --------------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-5, weight_decay=0.0)
        logger.info(f"Optimizer: {optimizer}")

        if self.scheduler is not None:
            logger.info("Instantiating scheduler...")
            scheduler = {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val/loss",
                "strict": True,
                "name": "learning_rate",

            }
            scheduler["scheduler"] = LinearWarmupCosineAnnealingLR(
                warmup_epochs= self.warm_up_epochs,  
                warmup_start_lr= 0.0,
                max_epochs=self.num_steps,
                optimizer=optimizer
            )
            optimizer_config = {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }
            logger.info(f"optimizer configuration: {optimizer_config}")
            return optimizer_config
        return optimizer

    def configure_loss_func_dict(self):
        """
        Configure the loss function dictionary

        rank loss nn.MarginRankingLoss(margin=0.5)
        Args:
            input1: (Tensor) predicted affinity
            input2: (Tensor) predicted affinity
            target: (Tensor) ranking labels

        """
        # rank loss nn.MarginRankingLoss(margin=0.5)
        return {"mse": nn.MSELoss()}

    def configure_metric_func_dict(self):
        """
        Configure the metric function dictionary
        """

        return {"accuracy": accuracy_metric}
    
    

    # --------------------------------------------------------------------------
    # Custom methods
    # --------------------------------------------------------------------------
    def compute_loss(
        self,
        pred_y: Tensor,
        true_y: Tensor,
        stage: str,
    ) -> Dict[str, Tensor]:
        if pred_y.ndim == 2:
            pred_y = pred_y.squeeze()
        if true_y.ndim == 2:
            true_y = true_y.squeeze()

        # compute loss
        loss_dict = {
            f"{stage}/loss/{k}": v(
                pred_y.float(), true_y.float()
            )
            for k, v in self.loss_func_dict.items()
        }
        return loss_dict

    # --------------------------------------------------------------------------
    # Log
    # --------------------------------------------------------------------------
    def log_step(self, log_dict: Dict[str, float], sync_dist: bool = False):
        # log step
        for k, v in log_dict.items():
            self.log(
                name=k,
                value=v,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=sync_dist,
            )

    def log_epoch(self, log_dict: Dict[str, float], sync_dist: bool = True) -> None:
        # log epoch
        for k, v in log_dict.items():
            self.log(
                name=k,
                value=v,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=sync_dist,
            )

    # --------------------------------------------------------------------------
    # Step Hooks
    # --------------------------------------------------------------------------
    def _one_step(
        self, batch: PygBatch, stage: str
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        # forward
        pred_y: Tensor = self.forward(batch)  # (B, 1)

        # compute loss
        loss_dict = self.compute_loss(
            pred_y=pred_y, true_y=batch.y, stage=stage
        )

        # NOTE: add key value pair `"loss": loss_value` the loss dict to return after each step
        # NOTE: required by Lightning
        # since we only use a single loss function, we only need to return the total
        loss_values = list(loss_dict.values())
        loss_dict["loss"] = loss_values[0]

        return loss_dict, pred_y

    def training_step(
        self, batch: PygBatch, batch_idx: int
    ) -> Tensor:
        # compute loss
        loss_dict, pred_y = self._one_step(
            batch=batch,
            stage="train",
        )
        # store loss for each batch in an epoch
        self.training_loss_epoch.append(loss_dict["loss"])
        # store pred labels for each batch in an epoch
        self.training_pred_y_epoch.append(pred_y.squeeze())  # (B,)
        # store true labels for each batch in an epoch
        self.training_true_y_epoch.append(batch.y.squeeze())  # (B,)
        # determine if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1
        self.log_step(loss_dict, sync_dist=is_distributed)
        return loss_dict["loss"]

    def validation_step(
        self, batch: PygBatch, batch_idx: int
    ) -> Tensor:
        # compute loss
        loss_dict, pred_y = self._one_step(
            batch=batch,
            stage="val",
        )
        # store loss for each batch in an epoch
        self.validation_loss_epoch.append(loss_dict["loss"])
        # store pred labels for each batch in an epoch
        self.validation_pred_y_epoch.append(pred_y.squeeze())  # (B,)
        # store true labels for each batch in an epoch
        self.validation_true_y_epoch.append(batch.y.squeeze())  # (B,)
        # determine if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1
        self.log_step(loss_dict, sync_dist=is_distributed)
        return loss_dict["loss"]

    def test_step(
        self,
        batch: PygBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        """
        Test set is packed in a single batch. This function is called only once
        for the entire test set.
        NOTE: the test set doesn't have true y, we only run a forward pass
        """
        # NOTE: input test dataloader is a dict of two loaders
        # e.g. test_dataloader = {"generalization": dl1, "perturbation": dl2}
        test_name = list(self.test_dataloader.keys())[dataloader_idx]
        names = batch.name
        pred_y: Tensor = self.forward(batch)  # (B, 1)
        d = [(n, p) for n, p in zip(names, pred_y.squeeze())]
        if test_name not in self.test_pred_y_epoch:
            self.test_pred_y_epoch[test_name] = []
        # store pred y for each batch in an epoch
        self.test_pred_y_epoch[test_name].extend(d)

    def predict_step(
        self, batch: PygBatch, batch_idx: int
    ) -> Tensor:
        """
        This function is called during inference.
        """
        # forward
        y_pred = self.forward(batch)  # (B, 1)
        return y_pred

    # --------------------------------------------------------------------------
    # Epoch Hooks
    # --------------------------------------------------------------------------
    def on_train_epoch_end(self) -> None:
        avg_loss = torch.stack(self.training_loss_epoch).mean()

        # Check if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1

        # Log with sync_dist=True if in distributed setting
        sync_dist = is_distributed

        pred_y = torch.cat(self.training_pred_y_epoch, dim=0)
        true_y = torch.cat(self.training_true_y_epoch, dim=0)

        # calculate accuracy
        mse = self.loss_func_dict["mse"](pred_y, true_y)

        # Log the average loss
        self.log_epoch(log_dict={"train/loss/avg": avg_loss}, sync_dist=sync_dist)
        self.log_epoch(log_dict={"train/mse": mse}, sync_dist=sync_dist)

        # Clear lists for the next epoch
        self.training_loss_epoch.clear()
        self.training_pred_y_epoch.clear()
        self.training_true_y_epoch.clear()

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack(self.validation_loss_epoch).mean()

        # Check if we are in a distributed setting
        is_distributed = self.trainer.world_size > 1

        # store pred y for each batch in an epoch
        pred_y = torch.cat(self.validation_pred_y_epoch, dim=0)
        true_y = torch.cat(self.validation_true_y_epoch, dim=0)

        # calculate accuracy
        mse = self.loss_func_dict["mse"](pred_y, true_y)

        # Log with sync_dist=True if in distributed setting
        sync_dist = is_distributed

        # Log the average loss
        self.log_epoch(log_dict={"val/loss/avg": avg_loss}, sync_dist=sync_dist)
        self.log_epoch(log_dict={"val/mse": mse}, sync_dist=sync_dist)

        # Clear lists for the next epoch
        self.validation_loss_epoch.clear()
        self.validation_pred_y_epoch.clear()
        self.validation_true_y_epoch.clear()

    def on_test_epoch_end(self) -> None:
        # NOTE: test set doesn't have true y, we only upload a csv file artifact
        # storing graph pair name and pred y for later usage
        for test_name in self.test_pred_y_epoch:
            # Check if we are in a distributed setting
            is_distributed = self.trainer.world_size > 1

            # store pred y for each batch in an epoch
            names, pred_y = zip(*self.test_pred_y_epoch[test_name])
            names = list(names)
            # move each item in pred_y to cpu and to float
            pred_y = [p.item() for p in pred_y]
            df = pd.DataFrame({"name": names, "pred_y": pred_y})

            # Save DataFrame to a temporary CSV file
            import os
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                df.to_csv(tmp.name, index=False)
                # create an artifact
                artifact = wandb.Artifact(
                    name=f"test_{test_name}",
                    type="test",
                    description=f"Test set for {test_name}",
                )
                artifact.add_file(tmp.name, name=f"{test_name}.csv")
                self.logger.experiment.log_artifact(artifact)
                # Clean up the temporary file
                os.unlink(tmp.name)

            # Clear lists for the next epoch
            self.test_pred_y_epoch[test_name].clear()
