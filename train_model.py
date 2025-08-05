#!/usr/bin/env python3
"""
Main script for training models using the train_model function.
Provides command line interface for all training parameters.
"""

from typing import Dict

import argparse
import logging
import sys
from pathlib import Path
from chewy.train import train_model

# Import your train_model function and other dependencies
# from your_module import train_model  # Uncomment and adjust import path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train model using Lightning trainer and WandB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory path for the dataset and output"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["cpu", "gpu", "tpu"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--split",
        type=dict,
        required=True,
        help="Path to the training split CSV file (relative to root/AbRank/splits/Split_AF3/)"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["train", "test", "validate"],
        help="Task type (use 'train' to enable training)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path to checkpoint file for resuming training"
    )
    
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["CosineAnnealing"],
        help="Scheduler type. Currently supports 'CosineAnnealing'"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not specified, generates a random seed"
    )
    
    parser.add_argument(
        "--wandb",
        type=dict,
        default=None,
        help="WandB project name for logging experiments"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    logger.info("Starting training script")
    
    # Parse arguments
    args = parse_args()
    
    # Log the configuration
    logger.info("Training configuration:")
    logger.info(f"  Root directory: {args.root}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Train split path: {args.train_split_path}")
    logger.info(f"  Task: {args.task}")
    logger.info(f"  Checkpoint path: {args.ckpt_path}")
    logger.info(f"  Scheduler: {args.scheduler}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  WandB project: {args.wandb_project}")
    logger.info(f"  WandB entity: {args.wandb_entity}")
    
    # Call the train_model function
    train_model(
        root=args.root,
        device=args.device,
        split=args.split,
        task=args.task,
        ckpt_path=args.ckpt_path,
        scheduler=args.scheduler,
        seed=args.seed,
        wandb=args.wandb
    )
        

if __name__ == "__main__":
    main()