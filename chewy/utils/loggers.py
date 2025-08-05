from lightning.pytorch.loggers.wandb import WandbLogger

def set_loggers(wandb_project, wandb_entity, split):
    wandblogger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            save_dir=f"./outputs_wandb/{split}",
            offline=False,
            id=None,
            anonymous=None,
            log_model=False,
            name="",
            prefix="",
            group="",
            tags=[],
            job_type="",
            notes=""
        )
    return wandblogger