"""Lightning integration of the conditioned diffusion adapter without editing old code."""
from __future__ import annotations
import torch
from megalodon.models.module import Graph3DInterpolantModel
from .conditioning import ContextInjector
from .retrieval import RetrievalLibrary


class ConditionedMRSA(Graph3DInterpolantModel):
    def __init__(self, *args, library_path: str, context_dim: int = 128,
                 inject_hidden_dim: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters({'library_path': library_path, 'context_dim': context_dim,
                                   'inject_hidden_dim': inject_hidden_dim})
        self.injector = ContextInjector(context_dim, inject_hidden_dim)
        self.injector.attach(self.dynamics)
        self.library_path = library_path
        self._library = None

    @property
    def library(self):
        if self._library is None:
            self._library = RetrievalLibrary.load(self.library_path)
            if self._library.embeddings.shape[1] != self.injector.net[0].in_features:
                raise RuntimeError('Graph-encoder and adapter dimensions do not match')
        return self._library

    def set_query(self, smiles):
        self.injector.set_context(self.library.batch_context([smiles], device=self.device))

    def _set_batch_context(self, batch):
        if not hasattr(batch, 'smiles'):
            raise RuntimeError('Training batch must include smiles attribute')
        smiles = batch.smiles
        if isinstance(smiles, str):
            smiles = [smiles]
        self.injector.set_context(self.library.batch_context(smiles, device=self.device))

    def configure_optimizers(self):
        # The upstream optimizer only sees self.dynamics; include the new adapter explicitly
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=float(self.optimizer_params.lr),
                                       amsgrad=bool(self.optimizer_params.amsgrad),
                                       weight_decay=float(self.optimizer_params.weight_decay))
        if self.lr_scheduler_params and self.lr_scheduler_params.type == 'linear_warmup':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=float(self.lr_scheduler_params.initial_lr) /
                float(self.lr_scheduler_params.final_lr),
                total_iters=int(self.lr_scheduler_params.num_warmup_steps))
            return {'optimizer': optimizer,
                    'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        return optimizer

    def training_step(self, batch, batch_idx):
        self._set_batch_context(batch)
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self._set_batch_context(batch)
        return super().validation_step(batch, batch_idx)

    def sample(self, *args, **kwargs):
        if self.injector._context is None:
            raise RuntimeError('Use set_query(smiles) before generating molecules')
        return super().sample(*args, **kwargs)


def initialize_conditioned(cfg, library_path):
    from megalodon.data.batch_preprocessor import BatchPreProcessor
    from omegaconf import OmegaConf
    return ConditionedMRSA(
        loss_params=cfg.loss, optimizer_params=cfg.optimizer,
        lr_scheduler_params=cfg.lr_scheduler, dynamics_params=cfg.dynamics,
        interpolant_params=cfg.interpolant, sampling_params=cfg.sample,
        self_cond_params=OmegaConf.select(cfg, 'self_conditioning', default=None),
        ema=OmegaConf.select(cfg, 'ema', default=True),
        batch_preprocessor=BatchPreProcessor(aug_rotations=cfg.data.aug_rotations,
                                             scale_coords=cfg.data.scale_coords),
        library_path=str(library_path), context_dim=128,
        inject_hidden_dim=int(cfg.dynamics.model_args.invariant_node_feat_dim),
    )


def load_original_weights(model, checkpoint):
    """Use only local trusted checkpoints, never arbitrary downloaded pickle files."""
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weights = ckpt.get('state_dict', ckpt)
    result = model.load_state_dict(weights, strict=False)
    unexpected = [k for k in result.unexpected_keys if not k.startswith('injector.')]
    missing = [k for k in result.missing_keys if not k.startswith('injector.')]
    if unexpected or missing:
        raise RuntimeError(f'Checkpoint/backbone mismatch missing={missing[:12]}, unexpected={unexpected[:12]}')
    return result
