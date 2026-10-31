

# SPDX-License-Identifier: Apache-2.0

"""
MRSA finetuning entrypoint for Megalodon.

Why this file exists:
- The original scripts/train.py treats cfg.resume as a Lightning training resume checkpoint.
- Lightning then tries to restore optimizer/scheduler states.
- When finetuning from a pretrained checkpoint with a new optimizer/config/data setup,
  optimizer parameter groups may not match and training crashes with:
  "loaded state dict contains a parameter group that doesn't match the size of optimizer's group".

This script loads pretrained model weights, but starts a fresh optimizer/scheduler.
Use it for finetuning from megalodon_large_diffusion.ckpt and for stage-2 finetuning.
"""

import os
import logging
from pathlib import Path

import torch
import hydra
from lightning import pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.data.molecule_datamodule import MoleculeDataModule
from megalodon.data.statistics import Statistics
from megalodon.metrics.molecule_evaluation_callback import MoleculeEvaluationCallback
from megalodon.models.module import Graph3DInterpolantModel


def resolve_checkpoint_path(resume_value):
    if not resume_value:
        return None
    resume_value = str(resume_value)
    if os.path.isdir(resume_value):
        return os.path.join(resume_value, "last.ckpt")
    return resume_value


@hydra.main(version_base=None, config_path="conf/qm9", config_name=None)
def main(cfg: DictConfig) -> None:
    logging.info("\n\n************** MRSA finetune configuration ***********")
    pl.seed_everything(cfg.train.seed)
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    cfg.outdir = os.path.join(cfg.outdir, cfg.run_name)
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(os.path.join(cfg.outdir, "checkpoints"), exist_ok=True)

    loss_fn = None
    batch_preprocessor = BatchPreProcessor(
        aug_rotations=cfg.data.aug_rotations,
        scale_coords=cfg.data.scale_coords,
    )

    ckpt_for_weights = resolve_checkpoint_path(cfg.resume)
    if ckpt_for_weights:
        if not os.path.exists(ckpt_for_weights):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_for_weights}")
        strict_load = OmegaConf.select(cfg, "finetune.load_strict", default=True)
        logging.info("Loading pretrained weights only, not optimizer state.")
        logging.info(f"Weights checkpoint: {ckpt_for_weights}")
        logging.info(f"strict_load={strict_load}")
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
            self_cond_params=OmegaConf.select(cfg, "self_conditioning", default=None),
            ema=OmegaConf.select(cfg, "ema", default=True),
            batch_preprocessor=batch_preprocessor,
        )
        # Critical difference from original scripts/train.py:
        # do NOT pass ckpt_path to trainer.fit, otherwise Lightning restores optimizer/scheduler.
        fit_ckpt_path = None
    else:
        logging.info("No checkpoint provided. Training from scratch.")
        pl_module = Graph3DInterpolantModel(
            loss_params=cfg.loss,
            optimizer_params=cfg.optimizer,
            lr_scheduler_params=cfg.lr_scheduler,
            dynamics_params=cfg.dynamics,
            interpolant_params=cfg.interpolant,
            sampling_params=cfg.sample,
            self_cond_params=OmegaConf.select(cfg, "self_conditioning", default=None),
            ema=OmegaConf.select(cfg, "ema", default=True),
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
            preserve_aromatic=OmegaConf.select(cfg.evaluation, "preserve_aromatic", default=True),
        )
    else:
        raise NotImplementedError

    num_nodes = cfg.train.num_nodes if "num_nodes" in cfg.train else 1

    trainer = pl.Trainer(
        max_epochs=cfg.train.n_epochs,
        logger=logger,
        callbacks=[lr_monitor, evaluation_callback, last_checkpoint_callback, best_checkpoint_callback],
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

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    trainer.fit(
        model=pl_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=fit_ckpt_path,
    )


if __name__ == "__main__":
    main()
