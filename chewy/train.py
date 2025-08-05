import uuid
from pathlib import Path
from typing import Dict, List
import os

import lightning as L
import rootutils
import torch
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from loguru import logger
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.optim import AdamW

from chewy.data.abrank_datamodule import AbRankDataModule
from chewy.data.abrank_regression_datamodule import AbRankRegressionDataModule
from chewy.model.builder_ranking import Chewy
from chewy.model.builder_regression import ChewyRegression
from chewy.utils.callbacks import set_callbacks
from chewy.utils.loggers import set_loggers
from chewy.utils.wandb import upload_ckpts_to_wandb, log_default_root_dir, log_run_dir

# set precision to trade off precision for performance
torch.set_float32_matmul_precision(precision="medium")


# ==================== Function ====================
def _num_training_steps(train_dataloader: PyGDataLoader, trainer: L.Trainer) -> int:
    """
    Returns total training steps inferred from datamodule and devices.

    Args:
        train_dataset: Training dataloader
        trainer: Lightning trainer

    Returns:
        Total number of training steps
    """
    if trainer.max_steps != -1:
        return trainer.max_steps

    dataset_size = (
        trainer.limit_train_batches
        if trainer.limit_train_batches not in {0, 1}
        else len(train_dataloader) * train_dataloader.batch_size
    )

    logger.info(f"Dataset size: {dataset_size}")

    num_devices = max(1, trainer.num_devices)
    effective_batch_size = (
        train_dataloader.batch_size * trainer.accumulate_grad_batches * num_devices
    )
    return (dataset_size // effective_batch_size) * trainer.max_epochs

def train_model(root, device, task, split, ckpt_path, wandb, scheduler=None, seed = None) -> None:
    """
    Train model using Lightning trainer and WandB.

    Args:
        root (str): Root directory path for the dataset and output.
        device (str): Device to use for training (e.g., 'cpu', 'gpu', 'cuda').
        train_split_path (str): Path to the training split CSV file (relative to root/AbRank/splits/Split_AF3/).
        task (str): Task type, should be 'train' to enable training.
        ckpt_path (str): Path to checkpoint file for resuming training (can be None).
        scheduler (str, optional): Scheduler type. If 'CosineAnnealing', uses LinearWarmupCosineAnnealingLR.
        seed (int, optional): Random seed for reproducibility. If None, generates a random seed.
        wandb_project (str, optional): WandB project name for logging experiments.
        wandb_entity (str, optional): WandB entity (username or team name) for logging experiments.

    """
    # set seed for random number generators in pytorch, numpy and python.random
    if seed is None:
       seed = torch.randint(0, 2**32 - 1, (1,)).item()
    logger.info(f"Setting seed {seed} using Lightning.seed_everything")
    L.seed_everything(seed)

    # ----------------------------------------
    # Instantiate datamodule
    # ----------------------------------------
    logger.info(f"Instantiating datamodule: <Train Data Module>...")
    datamodule_args = {
        'root':root,
        'train_split_path':split['train_split_path'],# NOTE: must be set in the main config
        'test_split_path_dict':{
            "generalization": split['test-generalization'],
            "perturbation": split['test-perturbation'],
        }, # NOTE: must be set in the main config
        'seed':seed, # NOTE: if null, will generate automatically; for reproducibility can set in the main config
        'num_workers':2,
        'batch_size':32, # Batch size for dataloader
        'shuffle':True,
        'follow_batch':["x_b", "x_g"],
        'exclude_keys':["metadata", "y", "y_b", "y_g", "edge_index_bg"]

    }
    dm: L.LightningDataModule = AbRankRegressionDataModule(**datamodule_args) if task == "Regression" else AbRankDataModule(**datamodule_args)

    # setup and prepare data
    dm.prepare_data()
    dm.setup()

    # ----------------------------------------
    # Instantiate callbacks, loggers
    # ----------------------------------------
    logger.info("Instantiating callbacks...")
    callbacks = set_callbacks(os.path.join(root, "output", split['name']))

    logger.info("Instantiating loggers...")
    L_logger = [set_loggers(wandb['project'], wandb['entity'], split['name'])]
    # if logger is wandb, initialise it via :func:experiment
    wandb_run = None
    run_name = uuid.uuid4().hex  # in case the wandb run is not initialized
    run_id = uuid.uuid4().hex  # in case the wandb run is not initialized
    for i in L_logger:
        if isinstance(i, WandbLogger):
            wandb_run = i.experiment  # this will initialize the wandb run
            # get wandb run id
            run_name = wandb_run.name  # e.g. "dulcet-sea-50"
            run_id = wandb_run.id  # e.g. "50"
            logger.info(f"Wandb run name: {run_name}")
            logger.info(f"Wandb run id: {run_id}")
            break

    # ----------------------------------------
    # Instantiate trainer
    # ----------------------------------------
    logger.info("Instantiating trainer...")
    logger.info(f"Trainer config: Device: {device}")
    trainer: L.Trainer = L.Trainer(
        accelerator=device,
        callbacks=callbacks,
        logger=L_logger,
        max_epochs = 100,
        check_val_every_n_epoch = 1,
        gradient_clip_val = 1.0,
        gradient_clip_algorithm = "norm",
        accumulate_grad_batches = 1,
        limit_train_batches = 1.0,
        limit_val_batches = 1.0,
        limit_test_batches = 1.0,
    )


    # ----------------------------------------
    # Model
    # ----------------------------------------
    dm.setup()  # type: ignore
    num_steps = _num_training_steps(dm.train_dataloader(), trainer)  # => 10_000
    logger.info(
        f"Setting number of training steps in scheduler to: {num_steps}"
    )

    logger.info("Instantiating model...")
    model_args = {
        'num_steps': num_steps, 
        'warm_up_epochs': trainer.val_check_interval, 
        'scheduler': scheduler
    }
    model: L.LightningModule = ChewyRegression(**model_args) if task == "Regression" else Chewy(**model_args)

    # ----------------------------------------
    # Model initialization
    # ----------------------------------------
    # logger.info("Initializing lazy layers...")
    # with torch.no_grad():
    #     dm.setup(stage="lazy_init")  # type: ignore
    #     batch = next(iter(dm.val_dataloader()))
    #     print(batch)
    #     print(batch[0].x_b)
    #     logger.info(f"Batch: {batch}")
    #     # forward pass
    #     out = model(batch)
    #     logger.info(f"Model output: {out}")
    #     del batch, out

    # ----------------------------------------
    # Training
    # ----------------------------------------
    logger.info("Starting training!")
    trainer.fit(
        model=model,
        datamodule=dm,
        ckpt_path=ckpt_path,  # resume from checkpoint
    )

    # ----------------------------------------
    # Testing
    # Evaluate the best model on the test set
    # ----------------------------------------
    # run test
    logger.info("Running test with the best model checkpoint...")
    # Get the test dataloader(s)
    test_loaders = dm.test_dataloader()
    # Store test dataloader(s) in the model for test_step to access
    if isinstance(test_loaders, dict):
        model.test_dataloader = test_loaders
    trainer.test(
        model=model,
        datamodule=dm,
        ckpt_path=trainer.checkpoint_callback.best_model_path,
    )

    # ----------------------------------------
    # Upload the best model to wandb
    # ----------------------------------------
    logger.info("Uploading checkpoints to wandb...")
    upload_ckpts_to_wandb(
        ckpt_callback=trainer.checkpoint_callback, wandb_run=wandb_run
    )
    logger.info("Uploading checkpoints to wandb... Done")

    # ----------------------------------------
    # Finishing
    # ----------------------------------------
    logger.info("Logging the run directory as an artifact...")
    try:
        log_default_root_dir(trainer.default_root_dir, wandb_run=wandb_run)
    except Exception as e:
        logger.error(f"Failed to log default_root_dir to wandb: {e}")
    try:
        log_run_dir(trainer.default_root_dir, wandb_run=wandb_run)
    except Exception as e:
        logger.error(f"Failed to log run_dir to wandb: {e}")
    logger.info("Logging the run directory as an artifact... Done")