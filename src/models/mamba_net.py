from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from medpy.metric.binary import hd95

from src.models.builder import EncoderDecoder as segmodel
from src.utils.loss import dice_bce_loss
from src.utils.train_and_eval import create_lr_scheduler
from src.utils.init_func import group_weight


class PETCTLightningModule(pl.LightningModule):
    def __init__(self, cfg: edict, module_hparams: Dict[str, float]) -> None:
        super().__init__()
        self.config = cfg
        self.model = segmodel(cfg=self.config, norm_layer=nn.BatchNorm2d)
        self.criterion = dice_bce_loss()
        self.save_hyperparameters(module_hparams)

        self._micro_stats: Dict[str, Dict[str, torch.Tensor]] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pet = x[:, 0:1, ...].repeat(1, 3, 1, 1)
        ct = x[:, 1:2, ...].repeat(1, 3, 1, 1)
        return self.model(ct, pet)

    def training_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch, "train")
        sync_dist = self._should_sync()
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=sync_dist)
        for key, value in metrics.items():
            self.log(f"train_{key}", value, on_step=False, on_epoch=True, prog_bar=(
                key == "dice"), sync_dist=sync_dist)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch, "val")
        sync_dist = self._should_sync()
        self.log("val_loss", loss, prog_bar=True, sync_dist=sync_dist)
        for key, value in metrics.items():
            self.log(f"val_{key}", value, prog_bar=(
                key in {"dice", "iou"}), sync_dist=sync_dist)
        return {"val_loss": loss, **metrics}

    def configure_optimizers(self):
        params_list = []
        params_list = group_weight(
            params_list, self.model, nn.BatchNorm2d, self.hparams.lr)
        params_to_optimize = [
            p for p in self.model.parameters() if p.requires_grad]

        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                params_to_optimize, lr=self.hparams.lr, eps=self.hparams.eps)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                params_list,
                self.hparams.lr,
                betas=(0.9, 0.999),
                eps=self.hparams.eps,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "SGDM":
            optimizer = torch.optim.SGD(
                params_list,
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                params_to_optimize,
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )

        steps_per_epoch = max(int(self.hparams.steps_per_epoch), 1)
        lr_scheduler = create_lr_scheduler(
            optimizer,
            num_step=steps_per_epoch,
            epochs=self.hparams.epochs,
            warmup=self.hparams.warm_up,
            warmup_epochs=self.hparams.warm_up_epoch,
        )
        scheduler_config = {"scheduler": lr_scheduler,
                            "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self._reset_micro_stats("train")

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        iou_micro = self._micro_iou("train")
        if iou_micro is not None:
            sync_dist = self._should_sync()
            self.log(
                "train_iou_micro",
                iou_micro.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=sync_dist,
            )

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self._reset_micro_stats("val")

    def on_validation_epoch_end(self) -> None:
        iou_micro = self._micro_iou("val")
        if iou_micro is not None:
            sync_dist = self._should_sync()
            self.log(
                "val_iou_micro",
                iou_micro.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=sync_dist,
            )

    def _shared_step(self, batch, stage: str):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(masks, logits)
        preds = torch.sigmoid(logits)
        self._update_micro_stats(stage, preds, masks)
        metrics = self._compute_metrics(preds, masks)
        return loss, metrics

    def _compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        preds = preds.detach()
        targets = targets.detach()
        preds_bin = (preds >= 0.5).float()
        targets_bin = (targets >= 0.5).float()

        tp = torch.sum(preds_bin * targets_bin)
        fp = torch.sum(preds_bin * (1.0 - targets_bin))
        fn = torch.sum((1.0 - preds_bin) * targets_bin)
        tn = torch.sum((1.0 - preds_bin) * (1.0 - targets_bin))

        smooth = torch.tensor(1e-7, device=preds.device)
        dice = (2 * tp) / (2 * tp + fp + fn + smooth)
        iou = tp / (tp + fp + fn + smooth)
        sensitivity = tp / (tp + fn + smooth)
        specificity = tn / (tn + fp + smooth)
        acc = 0.5 * (sensitivity + specificity)

        preds_np = preds_bin.detach().cpu().numpy().astype(np.uint8)
        targets_np = targets_bin.detach().cpu().numpy().astype(np.uint8)
        hd_values = []
        for pred_map, target_map in zip(preds_np, targets_np):
            pred_mask = pred_map[0]
            target_mask = target_map[0]
            if target_mask.sum() == 0 and pred_mask.sum() == 0:
                hd_values.append(0.0)
                continue

            pred_copy = pred_mask.copy()
            target_copy = target_mask.copy()
            h_mid = pred_copy.shape[0] // 2
            w_mid = pred_copy.shape[1] // 2

            if pred_copy.sum() == 0:
                pred_copy[h_mid, w_mid] = 1
            if target_copy.sum() == 0:
                target_copy[h_mid, w_mid] = 1

            try:
                hd_val = float(hd95(pred_copy, target_copy))
            except Exception:
                hd_val = float(
                    np.sqrt(pred_copy.shape[0] ** 2 + pred_copy.shape[1] ** 2))
            hd_values.append(hd_val)

        mean_hd = float(np.mean(hd_values)) if hd_values else 0.0
        hd_tensor = preds.new_tensor(mean_hd)

        return {
            "dice": dice,
            "iou": iou,
            "acc": acc,
            "hd95": hd_tensor,
        }

    def _reset_micro_stats(self, stage: str) -> None:
        device = self.device if isinstance(
            self.device, torch.device) else torch.device(str(self.device))
        self._micro_stats[stage] = {
            "tp": torch.zeros(1, device=device),
            "fp": torch.zeros(1, device=device),
            "fn": torch.zeros(1, device=device),
            "tn": torch.zeros(1, device=device),
        }

    def _update_micro_stats(self, stage: str, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if stage not in {"train", "val"}:
            return
        if stage not in self._micro_stats:
            self._reset_micro_stats(stage)
        stats = self._micro_stats[stage]
        preds_detached = preds.detach()
        targets_detached = targets.detach().float()
        preds_bin = (preds_detached >= 0.5).float()
        stats["tp"] += (preds_bin * targets_detached).sum()
        stats["fp"] += (preds_bin * (1.0 - targets_detached)).sum()
        stats["fn"] += ((1.0 - preds_bin) * targets_detached).sum()
        stats["tn"] += ((1.0 - preds_bin) * (1.0 - targets_detached)).sum()

    def _micro_iou(self, stage: str) -> Optional[torch.Tensor]:
        stats = self._micro_stats.get(stage)
        if not stats:
            return None
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        denom = tp + fp + fn
        if torch.all(denom == 0):
            return torch.ones_like(tp)
        eps = torch.finfo(tp.dtype).eps
        return tp / (denom + eps)

    def _should_sync(self) -> bool:
        return bool(self.trainer and self.trainer.num_devices and self.trainer.num_devices > 1)
