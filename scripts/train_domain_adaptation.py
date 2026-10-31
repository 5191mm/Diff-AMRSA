#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
MRSA Megalodon finetuning entrypoint based on the working weights-only script.

This file is intentionally based on train_mrsa_finetune_weights_only.py logic:
  - Load checkpoint weights through Graph3DInterpolantModel.load_from_checkpoint()
  - By default call trainer.fit(..., ckpt_path=None) so optimizer/scheduler are fresh
  - Optional full training-state resume with MRSA_RESUME_TRAINING_STATE=1
  - Export per-epoch training/validation metrics to CSV while training

Important for Hydra:
  If your checkpoint filename contains '=' such as best-epoch=9-step=221730--mol_stable=0.900.ckpt,
  do NOT pass it as resume=/path/with=sign.ckpt on the CLI.
    1. Copy it to a filename without '=' and pass resume=/path/resume.ckpt
    2. Or set MRSA_RESUME_CKPT=/path/with=sign.ckpt and omit resume=... from the CLI
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

print("BOOT: train_mrsa_finetune_weights_only_loss_log.py started", flush=True)

import hydra
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.data.molecule_datamodule import MoleculeDataModule
from megalodon.data.statistics import Statistics
from megalodon.metrics.molecule_evaluation_callback import MoleculeEvaluationCallback
from megalodon.models.module import Graph3DInterpolantModel


def resolve_checkpoint_path(resume_value: Optional[str]) -> Optional[str]:
    env_resume = os.environ.get("MRSA_RESUME_CKPT", "").strip()
    if env_resume:
        resume_value = env_resume
    if not resume_value:
        return None
    resume_value = str(resume_value)
    if os.path.isdir(resume_value):
        return os.path.join(resume_value, "last.ckpt")
    return resume_value


def get_cfg_value(cfg: DictConfig, key: str, default: Any = None) -> Any:
    return OmegaConf.select(cfg, key, default=default)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return None


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _as_plain(value: Any) -> Any:
    try:
        return OmegaConf.to_container(value, resolve=True)
    except Exception:
        return value


def _close_float(actual: Any, expected: Any, eps: float = 1e-12) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= eps
    except Exception:
        return actual == expected


def _require_equal(errors: list[str], name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, float):
        ok = _close_float(actual, expected)
    else:
        ok = actual == expected
    if not ok:
        errors.append(f"{name}: actual={actual!r}, expected={expected!r}")


def validate_strict_compare(cfg: DictConfig, ckpt_for_weights: Optional[str], resume_training_state: bool) -> Dict[str, Any]:
    strict_cfg = get_cfg_value(cfg, "strict_compare", None)
    if not strict_cfg or not bool(get_cfg_value(cfg, "strict_compare.enabled", False)):
        return {"enabled": False}

    role = str(get_cfg_value(cfg, "strict_compare.role", "")).strip()
    errors: list[str] = []
    if role not in {"scratch_mrsa", "finetune_mrsa"}:
        errors.append(f"strict_compare.role must be scratch_mrsa or finetune_mrsa, got {role!r}")

    _require_equal(errors, "data.dataset_root", str(cfg.data.dataset_root), str(get_cfg_value(cfg, "strict_compare.dataset_root")))
    _require_equal(errors, "train.seed", int(cfg.train.seed), int(get_cfg_value(cfg, "strict_compare.train_seed")))
    _require_equal(errors, "train.n_epochs", int(cfg.train.n_epochs), int(get_cfg_value(cfg, "strict_compare.n_epochs")))
    _require_equal(errors, "data.batch_size", int(cfg.data.batch_size), int(get_cfg_value(cfg, "strict_compare.batch_size")))
    _require_equal(errors, "data.inference_batch_size", int(cfg.data.inference_batch_size), int(get_cfg_value(cfg, "strict_compare.inference_batch_size")))
    _require_equal(errors, "optimizer.lr", float(cfg.optimizer.lr), float(get_cfg_value(cfg, "strict_compare.optimizer_lr")))
    _require_equal(errors, "lr_scheduler.initial_lr", float(cfg.lr_scheduler.initial_lr), float(get_cfg_value(cfg, "strict_compare.scheduler_initial_lr")))
    _require_equal(errors, "lr_scheduler.final_lr", float(cfg.lr_scheduler.final_lr), float(get_cfg_value(cfg, "strict_compare.scheduler_final_lr")))
    _require_equal(errors, "lr_scheduler.num_warmup_steps", int(cfg.lr_scheduler.num_warmup_steps), int(get_cfg_value(cfg, "strict_compare.scheduler_warmup_steps")))
    _require_equal(errors, "interpolant.timesteps", int(cfg.interpolant.timesteps), int(get_cfg_value(cfg, "strict_compare.timesteps")))
    _require_equal(errors, "evaluation.timesteps", int(cfg.evaluation.timesteps), int(get_cfg_value(cfg, "strict_compare.timesteps")))

    if role == "scratch_mrsa" and ckpt_for_weights:
        errors.append("scratch_mrsa must use resume=null and must not load a checkpoint")
    if role == "finetune_mrsa" and not ckpt_for_weights:
        errors.append("finetune_mrsa must provide a GEOM pretrained checkpoint through resume=... or MRSA_RESUME_CKPT")
    if resume_training_state:
        errors.append("strict comparison requires fresh optimizer and scheduler, unset MRSA_RESUME_TRAINING_STATE")

    manifest = {
        "enabled": True,
        "role": role,
        "ckpt_for_weights": ckpt_for_weights,
        "resume_training_state": resume_training_state,
        "dataset_root": str(cfg.data.dataset_root),
        "processed_folder": str(cfg.data.processed_folder),
        "seed": int(cfg.train.seed),
        "n_epochs": int(cfg.train.n_epochs),
        "batch_size": int(cfg.data.batch_size),
        "inference_batch_size": int(cfg.data.inference_batch_size),
        "optimizer": _as_plain(cfg.optimizer),
        "lr_scheduler": _as_plain(cfg.lr_scheduler),
        "interpolant_timesteps": int(cfg.interpolant.timesteps),
        "evaluation_timesteps": int(cfg.evaluation.timesteps),
        "only_intended_difference": "initialization: random for scratch_mrsa, GEOM checkpoint weights for finetune_mrsa",
        "errors": errors,
    }
    if errors:
        raise ValueError("Strict comparison validation failed:\n" + "\n".join(f"- {err}" for err in errors))
    print(f"BOOT: strict comparison validation passed for role={role}", flush=True)
    return manifest


class EpochMetricCsvLogger(Callback):
    """Write callback_metrics to CSV after each train/validation epoch.

    It writes:
      - epoch_metrics_long.csv: one row per metric per hook call
      - epoch_loss_summary.csv: one row per hook call with common loss/quality metrics as columns
      - epoch_metrics_events.jsonl: raw metric dict per hook call
    """

    def __init__(self, log_dir: str, interesting_keywords: Optional[Iterable[str]] = None) -> None:
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.long_csv = self.log_dir / "epoch_metrics_long.csv"
        self.summary_csv = self.log_dir / "epoch_loss_summary.csv"
        self.events_jsonl = self.log_dir / "epoch_metrics_events.jsonl"
        self.interesting_keywords = tuple(interesting_keywords or (
            "loss", "train", "val", "valid", "mol_stable", "atm_stable", "qed",
            "sa", "logp", "lipinski", "diversity", "similarity", "novelty", "lr",
        ))
        self._summary_header_written = self.summary_csv.exists() and self.summary_csv.stat().st_size > 0
        self._long_header_written = self.long_csv.exists() and self.long_csv.stat().st_size > 0
        print(f"BOOT: epoch metric CSV logger enabled at {self.log_dir}", flush=True)

    def _collect_metrics(self, trainer: pl.Trainer) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for key, value in dict(trainer.callback_metrics).items():
            val = _to_float(value)
            if val is not None:
                metrics[str(key)] = val
        # Also try logged metrics in case callback_metrics is sparse
        for key, value in dict(getattr(trainer, "logged_metrics", {}) or {}).items():
            val = _to_float(value)
            if val is not None and str(key) not in metrics:
                metrics[str(key)] = val
        return metrics

    def _write(self, trainer: pl.Trainer, stage: str) -> None:
        metrics = self._collect_metrics(trainer)
        event = {
            "time": _now(),
            "stage": stage,
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "metrics": metrics,
        }
        with self.events_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        # Long CSV
        with self.long_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "stage", "epoch", "global_step", "metric", "value"])
            if not self._long_header_written:
                writer.writeheader()
                self._long_header_written = True
            for key, val in sorted(metrics.items()):
                writer.writerow({
                    "time": event["time"],
                    "stage": stage,
                    "epoch": event["epoch"],
                    "global_step": event["global_step"],
                    "metric": key,
                    "value": val,
                })

        # Summary CSV with a stable useful subset plus all loss-like keys
        summary: Dict[str, Any] = {
            "time": event["time"],
            "stage": stage,
            "epoch": event["epoch"],
            "global_step": event["global_step"],
        }
        for key, val in sorted(metrics.items()):
            key_lower = key.lower()
            if any(word in key_lower for word in self.interesting_keywords):
                safe_key = key.replace("/", "_").replace(" ", "_")
                summary[safe_key] = val
        # Append with dynamic header by rewriting if header expands
        self._append_dynamic_csv(self.summary_csv, summary)
        print(
            f"[{_now()}] LOG {stage}: epoch={event['epoch']} step={event['global_step']} "
            f"metrics={len(metrics)} summary_cols={max(0, len(summary)-4)}",
            flush=True,
        )

    @staticmethod
    def _append_dynamic_csv(path: Path, row: Dict[str, Any]) -> None:
        if not path.exists() or path.stat().st_size == 0:
            fieldnames = list(row.keys())
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return

        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fieldnames = list(reader.fieldnames or [])
            old_rows = list(reader)
        new_fieldnames = old_fieldnames[:]
        for key in row.keys():
            if key not in new_fieldnames:
                new_fieldnames.append(key)
        if new_fieldnames != old_fieldnames:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=new_fieldnames)
                writer.writeheader()
                for old_row in old_rows:
                    writer.writerow(old_row)
                writer.writerow(row)
        else:
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=new_fieldnames)
                writer.writerow(row)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._write(trainer, "train_epoch_end")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._write(trainer, "val_epoch_end")

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        try:
            self._write(trainer, "exception")
        except Exception:
            pass


@hydra.main(version_base=None, config_path="conf/qm9", config_name=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    print("BOOT: Hydra config loaded", flush=True)
    logging.info("\n\n************** MRSA finetune weights-only + loss-log configuration ***********")
    pl.seed_everything(cfg.train.seed)
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    cfg.outdir = os.path.join(cfg.outdir, cfg.run_name)
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(os.path.join(cfg.outdir, "checkpoints"), exist_ok=True)

    loss_log_dir = os.environ.get("MRSA_LOSS_LOG_DIR", "").strip()
    if not loss_log_dir:
        loss_log_dir = os.path.join(cfg.outdir, "mrsa_epoch_logs")
    os.makedirs(loss_log_dir, exist_ok=True)
    with open(os.path.join(loss_log_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))
    with open(os.path.join(loss_log_dir, "train_start_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "time": _now(),
            "cwd": os.getcwd(),
            "loss_log_dir": loss_log_dir,
            "MRSA_RESUME_CKPT": os.environ.get("MRSA_RESUME_CKPT", ""),
            "MRSA_RESUME_TRAINING_STATE": os.environ.get("MRSA_RESUME_TRAINING_STATE", "0"),
            "cli_resume": str(get_cfg_value(cfg, "resume", "")),
        }, f, indent=2, ensure_ascii=False)

    loss_fn = None
    batch_preprocessor = BatchPreProcessor(
        aug_rotations=cfg.data.aug_rotations,
        scale_coords=cfg.data.scale_coords,
    )

    ckpt_for_weights = resolve_checkpoint_path(get_cfg_value(cfg, "resume", None))
    resume_training_state = os.environ.get("MRSA_RESUME_TRAINING_STATE", "0").strip().lower() in {"1", "true", "yes", "y"}
    strict_manifest = validate_strict_compare(cfg, ckpt_for_weights, resume_training_state)
    with open(os.path.join(loss_log_dir, "strict_compare_runtime.json"), "w", encoding="utf-8") as f:
        json.dump(strict_manifest, f, indent=2, ensure_ascii=False)

    if ckpt_for_weights:
        if not os.path.exists(ckpt_for_weights):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_for_weights}")
        strict_load = get_cfg_value(cfg, "finetune.load_strict", True)
        print(f"BOOT: loading checkpoint weights from {ckpt_for_weights}", flush=True)
        print(f"BOOT: strict_load={strict_load}", flush=True)
        print(f"BOOT: MRSA_RESUME_TRAINING_STATE={resume_training_state}", flush=True)

        pl_module = Graph3DInterpolantModel.load_from_checkpoint(
            ckpt_for_weights,
            strict=strict_load,
            loss_fn=loss_fn,
            loss_params=cfg.loss,
            optimizer_params=cfg.optimizer,
            lr_scheduler_params=cfg.lr_scheduler,
            dynamics_params=cfg.dynamics,
            interpolant_params=cfg.interpolant,
            sampling_params=cfg.sample,
            self_cond_params=get_cfg_value(cfg, "self_conditioning", None),
            ema=get_cfg_value(cfg, "ema", True),
            batch_preprocessor=batch_preprocessor,
        )
        fit_ckpt_path = ckpt_for_weights if resume_training_state else None
    else:
        print("BOOT: no checkpoint provided. Training from scratch.", flush=True)
        pl_module = Graph3DInterpolantModel(
            loss_params=cfg.loss,
            optimizer_params=cfg.optimizer,
            lr_scheduler_params=cfg.lr_scheduler,
            dynamics_params=cfg.dynamics,
            interpolant_params=cfg.interpolant,
            sampling_params=cfg.sample,
            self_cond_params=get_cfg_value(cfg, "self_conditioning", None),
            ema=get_cfg_value(cfg, "ema", True),
            loss_fn=loss_fn,
            batch_preprocessor=batch_preprocessor,
        )
        fit_ckpt_path = None

    wandb_resume = cfg.wandb_params.resume if "resume" in cfg.wandb_params else "allow"
    logger = pl.loggers.WandbLogger(
        save_dir=cfg.outdir,
        project=cfg.wandb_params.project,
        group=cfg.wandb_params.group,
        name=cfg.run_name,
        id=cfg.run_name,
        resume=wandb_resume,
        mode=cfg.wandb_params.mode,
    )
    logger.log_hyperparams(cfg)

    datamodule = MoleculeDataModule(
        cfg.data.dataset_root,
        cfg.data.processed_folder,
        cfg.data.batch_size,
        cfg.data.data_loader_type,
        cfg.data.inference_batch_size,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    epoch_metric_logger = EpochMetricCsvLogger(loss_log_dir)

    last_checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.outdir, "checkpoints"),
        save_last=True,
        every_n_train_steps=cfg.train.checkpoint_every_n_train_steps,
        save_on_train_epoch_end=True,
        filename="last-{epoch}-{step}",
    )
    metric_name = cfg.train.checkpoint_monitor.replace("/", "_")
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.outdir, "checkpoints"),
        save_top_k=5,
        monitor=cfg.train.checkpoint_monitor,
        mode=cfg.train.checkpoint_monitor_mode,
        filename="best-{epoch}-{step}--{" + metric_name + ":.3f}",
    )

    if cfg.evaluation.type == "molecules":
        energy_metrics_args = (
            OmegaConf.to_container(cfg.evaluation.energy_metrics_args, resolve=True)
            if cfg.evaluation.energy_metrics_args is not None
            else None
        )
        statistics = Statistics.load_statistics(
            statistics_dir=f"{cfg.data.dataset_root}/{cfg.data.processed_folder}",
            split_name="train",
        )
        evaluation_callback = MoleculeEvaluationCallback(
            n_graphs=cfg.evaluation.n_molecules,
            batch_size=cfg.evaluation.batch_size,
            timesteps=cfg.evaluation.timesteps,
            train_smiles=datamodule.train_dataset.smiles,
            statistics=statistics,
            compute_2D_metrics=cfg.evaluation.compute_2D_metrics,
            compute_3D_metrics=cfg.evaluation.compute_3D_metrics,
            compute_train_data_metrics=cfg.evaluation.compute_train_data_metrics,
            compute_energy_metrics=cfg.evaluation.compute_energy_metrics,
            energy_metrics_args=energy_metrics_args,
            scale_coords=cfg.evaluation.scale_coords,
            preserve_aromatic=get_cfg_value(cfg, "evaluation.preserve_aromatic", True),
        )
    else:
        raise NotImplementedError(f"Unsupported evaluation.type: {cfg.evaluation.type}")

    num_nodes = cfg.train.num_nodes if "num_nodes" in cfg.train else 1

    callbacks = [
        lr_monitor,
        evaluation_callback,
        epoch_metric_logger,
        last_checkpoint_callback,
        best_checkpoint_callback,
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.train.n_epochs,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=cfg.train.enable_progress_bar,
        accelerator="gpu",
        devices=cfg.train.gpus,
        num_nodes=num_nodes,
        strategy=("ddp" if cfg.train.gpus > 1 else "auto"),
        check_val_every_n_epoch=cfg.train.val_freq,
        gradient_clip_val=cfg.train.gradient_clip_value,
        log_every_n_steps=cfg.train.log_freq,
        num_sanity_val_steps=0,
    )

    print("BOOT: building train and validation dataloaders", flush=True)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    if fit_ckpt_path:
        print(f"BOOT: trainer.fit starts with full training-state resume ckpt_path={fit_ckpt_path}", flush=True)
    else:
        print("BOOT: trainer.fit starts with ckpt_path=None, optimizer will be fresh", flush=True)

    trainer.fit(
        model=pl_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=fit_ckpt_path,
    )

    print("BOOT: training finished", flush=True)
    print(f"BOOT: epoch logs written to {loss_log_dir}", flush=True)


if __name__ == "__main__":
    main()
