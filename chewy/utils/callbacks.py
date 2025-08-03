import os
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    LearningRateMonitor
)

def set_callbacks(output_dir: str):
    """
    Instantiate callbacks for AbRank ranking training based on the config files.
    
    Args:
        output_dir: Directory where checkpoints and outputs will be saved
    
    Returns:
        List of instantiated callback objects
    """
    
    # 1. Model Checkpoint Callback
    # Monitors validation accuracy and saves the best model
    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="epoch_{epoch:03d}",
        monitor="val/accuracy",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,  # Override from default config
        verbose=True,
        save_weights_only=False,
        save_on_train_epoch_end=True
    )
    
    # 2. Early Stopping Callback
    # Stops training when validation accuracy stops improving
    early_stopping = EarlyStopping(
        monitor="val/accuracy",
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode="max",
        strict=True,
        check_finite=True,
        stopping_threshold=None,
        divergence_threshold=None,
        check_on_train_epoch_end=False
    )
    
    # 3. Stop on NaN Callback (using EarlyStopping with specific config)
    # Stops training if training loss becomes NaN or infinite
    stop_on_nan = EarlyStopping(
        monitor="train/loss/avg",
        min_delta=0.0,
        patience=10_000_000,  # Very high patience - only stops on NaN/inf
        verbose=True,
        mode="min",
        strict=True,
        check_finite=True,
        stopping_threshold=None,
        divergence_threshold=None,
        check_on_train_epoch_end=False
    )
    
    # 4. Model Summary Callback
    # Displays model architecture summary
    model_summary = RichModelSummary(
        max_depth=-1  # Override from default config (was 1)
    )
    
    # 5. Progress Bar Callback
    # Shows rich progress bar during training
    rich_progress_bar = RichProgressBar()
    
    # 6. Learning Rate Monitor
    # Logs learning rate changes
    learning_rate_monitor = LearningRateMonitor()
    
    # Return all callbacks in a list
    callbacks = [
        model_checkpoint,
        early_stopping,
        stop_on_nan,
        model_summary,
        rich_progress_bar,
        learning_rate_monitor
    ]
    
    return callbacks