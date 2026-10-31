# Strict MRSA comparison setup

This package keeps three comparison lines conceptually separate.

## Line 1 Original GEOM direct sampling

No code changes are required for this line. Use the original GEOM checkpoint with the original GEOM config.

```bash
python scripts/sample_domain_adapted.py \
  --compare_group pretrained_baseline \
  --config_path scripts/conf/drugs/megalodon_diffusion.yaml \
  --ckpt_path /autodl-fs/data/megalodon_large_diffusion.ckpt \
  --predictor /autodl-fs/data/custom_mrsa_schemeA_fixed_30_10/processed/mrsa_activity_predictor_schemeA.joblib \
  --n_graphs 2000 \
  --top_k 200 \
  --batch_size 100 \
  --timesteps 500 \
  --statistics_split train \
  --save_path outputs/compare_guided/pretrained_baseline \
  --save_name pretrained_baseline_guided \
  --activity_threshold 0.5 \
  --mic_ceiling 100.0 \
  --w_mic 0.60 \
  --w_activity_prob 0.20 \
  --w_qed 0.10 \
  --w_lipinski 0.05 \
  --w_alert 0.05 \
  --save_raw_pickle
```

## Line 2 Scratch MRSA

Use the same training entrypoint as finetune with `resume=null`.

```bash
PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 python scripts/train_domain_adaptation.py \
  --config-path conf/drugs \
  --config-name mrsa_diffusion_scratch_scheme_a
```

## Line 3 Finetune MRSA

Use the confirmed entrypoint and provide the original GEOM checkpoint.

```bash
unset MRSA_RESUME_TRAINING_STATE
PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 python scripts/train_domain_adaptation.py \
  --config-path conf/drugs \
  --config-name mrsa_diffusion_finetune_scheme_a \
  resume=/autodl-fs/data/megalodon_large_diffusion.ckpt
```

## Strictness checks

Before training:

```bash
python scripts/check_strict_mrsa_compare.py
```

After training:

```bash
python scripts/check_ckpt_hparams_strict.py results/mrsa_strict_scratch_scheme_a/mrsa_strict_scratch_scheme_a/checkpoints/last.ckpt
python scripts/check_ckpt_hparams_strict.py results/mrsa_strict_finetune_scheme_a/mrsa_strict_finetune_scheme_a/checkpoints/last.ckpt
```

The only intended training difference between Scratch MRSA and Finetune MRSA is initialization.
Scratch starts from random initialization. Finetune starts from GEOM checkpoint weights and uses a fresh optimizer and scheduler.
