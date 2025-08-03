#!/usr/bin/env python3
"""
Main script for training models using the train_model function.
Provides command line interface for all training parameters.
"""

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
        choices=["cpu", "gpu", "cuda", "mps", "tpu"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--train-split-path",
        type=str,
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
        "--wandb-project",
        type=str,
        default=None,
        help="WandB project name for logging experiments"
    )
    
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity (username or team name) for logging experiments"
    )
    
    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments."""
    # Check if root directory exists
    root_path = Path(args.root)
    if not root_path.exists():
        logger.error(f"Root directory does not exist: {args.root}")
        sys.exit(1)
    
    # Check if checkpoint path exists (if provided)
    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path)
        if not ckpt_path.exists():
            logger.error(f"Checkpoint file does not exist: {args.ckpt_path}")
            sys.exit(1)
    
    # Check if train split path exists
    train_split_full_path = root_path / "AbRank" / "splits" / "Split_AF3" / args.train_split_path
    if not train_split_full_path.exists():
        logger.warning(f"Training split file may not exist: {train_split_full_path}")
        # Don't exit here as the file might be created by the training process
    
    # Validate seed range
    if args.seed is not None:
        if args.seed < 0 or args.seed >= 2**32:
            logger.error(f"Seed must be between 0 and {2**32-1}")
            sys.exit(1)
    
    logger.info("Argument validation passed")


def main():
    """Main function."""
    logger.info("Starting training script")
    
    # Parse arguments
    args = parse_args()
    
    # Validate arguments
    validate_args(args)
    
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
    
    try:
        # Call the train_model function
        train_model(
            root=args.root,
            device=args.device,
            train_split_path=args.train_split_path,
            task=args.task,
            ckpt_path=args.ckpt_path,
            scheduler=args.scheduler,
            seed=args.seed,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity
        )
        
        # For now, just log that we would call the function
        logger.info("Would call train_model with the provided arguments")
        logger.info("Please uncomment the train_model call and import statement")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        sys.exit(1)
    
    logger.info("Training script completed successfully")


if __name__ == "__main__":
    main()