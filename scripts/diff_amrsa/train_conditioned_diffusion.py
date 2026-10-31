#!/usr/bin/env python3
"""Fine-tune Megalodon with *actual in-denoiser* retrieval context injection."""
import argparse
from pathlib import Path
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from megalodon.data.molecule_datamodule import MoleculeDataModule
from diff_amrsa.conditional_model import initialize_conditioned, load_original_weights
from diff_amrsa.retrieval import RetrievalLibrary
from diff_amrsa.dataset import verify_train_only_library


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--backbone-ckpt', required=True)
    p.add_argument('--library', required=True)
    p.add_argument('--split-dir', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--devices', type=int, default=1)
    p.add_argument('--freeze-backbone', action='store_true')
    a = p.parse_args()
    if not Path(a.library).is_file():
        raise FileNotFoundError(a.library)
    verified = verify_train_only_library(RetrievalLibrary.load(a.library), a.split_dir)
    print(f'Verified {verified} train-only retrieval references')
    cfg = OmegaConf.load(a.config)
    cfg.data.batch_size = cfg.data.inference_batch_size = a.batch_size
    cfg.optimizer.lr = a.lr
    cfg.train.n_epochs = a.epochs
    pl.seed_everything(42, workers=True)
    model = initialize_conditioned(cfg, str(Path(a.library).resolve()))
    load_original_weights(model, a.backbone_ckpt)
    if a.freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith('injector.'):
                param.requires_grad_(False)
    data = MoleculeDataModule(dataset_root=cfg.data.dataset_root, processed_folder=cfg.data.processed_folder,
                              batch_size=a.batch_size, inference_batch_size=a.batch_size, data_loader_type=cfg.data.data_loader_type)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    trainer = pl.Trainer(max_epochs=a.epochs, accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         devices=a.devices, logger=False, gradient_clip_val=1.,
                         callbacks=[ModelCheckpoint(dirpath=out, filename='conditioned-{epoch}-{step}',
                                                    monitor='val/loss', mode='min', save_top_k=1),
                                    ModelCheckpoint(dirpath=out, filename='conditioned-last', save_last=True)],
                         num_sanity_val_steps=0)
    trainer.fit(model, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())


if __name__ == '__main__':
    main()
