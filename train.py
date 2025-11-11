import argparse
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from easydict import EasyDict as edict
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

try:
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    WandbLogger = None  # type: ignore

from src.models.mamba_net import PETCTLightningModule
from src.data.PCLT_datamodule import PETCTDataModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train segmentation model with PyTorch Lightning")

    parser.add_argument("--img_dir", type=str, help='Path to the dataset')
    parser.add_argument("--split_train_val_test", type=str, 
                        help='Path to the train.txt and test.txt files with the split sets')
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("-b", "--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=50, type=int, metavar="N")
    parser.add_argument("--eps", default=1e-8, type=float, help="adam eps")

    parser.add_argument("--lr", default=0.00006, type=float,
                        help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M")
    parser.add_argument("--weight-decay", default=1e-2,
                        type=float, dest="weight_decay")
    parser.add_argument("--print-freq", default=200,
                        type=int, help="logging frequency in steps")
    parser.add_argument("--resume", default="",
                        help="path to checkpoint to resume from (.ckpt or .pth)")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N")
    parser.add_argument("--save-best", default=True, type=bool,
                        help="save best and last epoch weights (.pth files)")
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use mixed precision training when possible")
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N")

    parser.add_argument("--backbone", default="sigma_tiny", choices=["sigma_tiny", "sigma_small", "sigma_base"])
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--decoder", type=str, default="MambaDecoder")
    parser.add_argument("--decoder_embed_dim", type=int, default=512)
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--bn_eps", type=int, default=1e-3)
    parser.add_argument("--bn_momentum", type=int, default=0.1)
    parser.add_argument("--num_classes", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument("--sync_bn", type=bool, default=True)
    parser.add_argument("--warm_up", default=False, type=bool)
    parser.add_argument("--warm_up_epoch", default=5, type=int)
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="cipa",
                        type=str, help="W&B project name")
    parser.add_argument("--wandb_run_name", default=None,
                        type=str, help="W&B run name override")
    parser.add_argument("--devices", default=None, type=int,
                        help="Number of devices to use. Defaults to auto when distributed")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic mode (warn-only fallback for non-deterministic ops)")
    parser.add_argument("--fusion-type", default="context_gate", choices=["context_gate", "other"],  
                        help="Fusion strategy for PET/CT features in the encoder.")
    parser.add_argument("--context-gate-reduction", default=4, type=int,
                        help="Channel reduction ratio for the context gate MLP.")
    parser.add_argument("--context-gate-min-channels", default=32, type=int,
                        help="Minimum hidden units for the context gate MLP.")
    parser.add_argument("--context-gate-no-residual", action="store_true",
                        help="Disable residual scaling inside the context gate.")
    parser.add_argument("--use_cgm", action="store_true",
                        help="Enable CGM.")
    parser.add_argument("--use_dcim", action="store_true",
                        help="Enable DCIM.")
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed, workers=True)

    args.context_gate = edict()
    args.context_gate.reduction = args.context_gate_reduction
    args.context_gate.min_channel = args.context_gate_min_channels
    args.context_gate.residual = not args.context_gate_no_residual

    data_module = PETCTDataModule(args)
    data_module.setup()
    steps_per_epoch = data_module.train_steps_per_epoch

    module_hparams = {
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "warm_up": args.warm_up,
        "warm_up_epoch": args.warm_up_epoch,
        "eps": args.eps,
        "steps_per_epoch": steps_per_epoch,
        "wandb": args.wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name or "",
        "deterministic": args.deterministic,
        "fusion_type": args.fusion_type,
        "context_gate_reduction": args.context_gate.reduction,
        "context_gate_min_channel": args.context_gate.min_channel,
        "context_gate_residual": args.context_gate.residual,
    }

    model = PETCTLightningModule(args, module_hparams)

    wandb_logger: Optional[WandbLogger] = None
    if args.wandb:
        if WandbLogger is None:
            raise ImportError(
                "WandbLogger not available. Please install wandb and pytorch-lightning extras.")
        run_name = args.wandb_run_name
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
        )
        experiment = getattr(wandb_logger, "experiment", None)
        config_obj = getattr(experiment, "config", None)
        if config_obj is not None and hasattr(config_obj, "update"):
            # type: ignore[arg-type]
            config_obj.update(module_hparams, allow_val_change=True)
        else:
            wandb_logger.log_hyperparams(module_hparams)
        wandb_logger.watch(model, log="all", log_freq=args.print_freq)

    ckpt_path: Optional[str] = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.suffix == ".ckpt":
            ckpt_path = str(resume_path.resolve())
        elif resume_path.suffix == ".pth" and resume_path.is_file():
            checkpoint = torch.load(resume_path, map_location="cpu")
            state_dict = checkpoint["model"] if isinstance(
                checkpoint, dict) and "model" in checkpoint else checkpoint
            model.model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unsupported resume file: {resume_path}")

    callbacks = [
        ModelCheckpoint(
            monitor="val_iou_micro",
            mode="max",
            save_top_k=3 if args.save_best else 0,
            save_last=args.save_best,
            filename="cipa_context-{epoch:02d}-{val_dice:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    accelerator = "gpu" if args.device.startswith(
        "cuda") and torch.cuda.is_available() else "cpu"
    if accelerator == "gpu":
        devices = args.devices if args.devices is not None else (
            "auto" if args.distributed else 1)
    else:
        devices = 1

    deterministic_flag: object
    if args.deterministic:
        deterministic_flag = "warn"
    else:
        deterministic_flag = False

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=args.nodes,
        precision="16-mixed" if args.amp and accelerator == "gpu" else "32-true",
        max_epochs=args.epochs,
        log_every_n_steps=args.print_freq,
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=wandb_logger if wandb_logger is not None else True,
        sync_batchnorm=args.sync_bn if accelerator == "gpu" else False,
        deterministic=deterministic_flag,
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
