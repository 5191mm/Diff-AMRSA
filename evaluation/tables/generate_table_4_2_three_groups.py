#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from generate_table_4_2_from_csv import compute_table


def main():
    parser = argparse.ArgumentParser(description='Generate three-group molecular quality comparison table')
    parser.add_argument('--pretrained_baseline_csv', required=True)
    parser.add_argument('--scratch_csv', required=True)
    parser.add_argument('--finetune_csv', required=True)
    parser.add_argument('--training_csv', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--generated_smiles_col', default='smiles')
    parser.add_argument('--training_smiles_col', default='smiles')
    parser.add_argument('--sampled_count', type=int, default=2000)
    parser.add_argument('--top_n', type=int, default=200)
    parser.add_argument('--max_diversity_pairs', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    groups = [
        ('Original Megalodon', Path(args.pretrained_baseline_csv)),
        ('Scratch-MRSA', Path(args.scratch_csv)),
        ('Finetune-MRSA', Path(args.finetune_csv)),
    ]

    summary_rows = []
    long_rows = []
    detail_frames = []
    raw_metrics = {}

    for group_name, csv_path in groups:
        table_df, metrics, detail_df = compute_table(
            generated_csv=csv_path,
            training_csv=Path(args.training_csv),
            generated_smiles_col=args.generated_smiles_col,
            training_smiles_col=args.training_smiles_col,
            sampled_count=args.sampled_count,
            top_n=args.top_n,
            max_diversity_pairs=args.max_diversity_pairs,
            seed=args.seed,
        )

        raw_metrics[group_name] = metrics

        summary_rows.append({
            'Model': group_name,
            'Sampled molecules': metrics['sampled_count'],
            'Valid molecules': metrics['valid_count'],
            'Validity': metrics['validity'],
            'Connected': metrics['connected_ratio'],
            'Unique molecules': metrics['unique_count'],
            'Uniqueness': metrics['uniqueness'],
            'Novel molecules': metrics['novel_count'],
            'Novelty': metrics['novelty'],
            'Diversity': metrics['diversity'],
        })

        table_df.insert(0, 'Model', group_name)
        long_rows.append(table_df)

        detail_df = detail_df.copy()
        detail_df.insert(0, 'Model', group_name)
        detail_frames.append(detail_df)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / 'table_4_2_three_groups_summary.csv', index=False, encoding='utf-8-sig')

    long_df = pd.concat(long_rows, ignore_index=True)
    long_df.to_csv(outdir / 'table_4_2_three_groups_long.csv', index=False, encoding='utf-8-sig')

    detail_df = pd.concat(detail_frames, ignore_index=True)
    detail_df.to_csv(outdir / 'table_4_2_three_groups_detail.csv', index=False, encoding='utf-8-sig')

    with open(outdir / 'table_4_2_three_groups_raw_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(raw_metrics, f, indent=2, ensure_ascii=False)

    print(f'Saved summary table: {outdir / "table_4_2_three_groups_summary.csv"}')
    print(f'Saved long table: {outdir / "table_4_2_three_groups_long.csv"}')
    print(f'Saved detail table: {outdir / "table_4_2_three_groups_detail.csv"}')
    print(f'Saved raw metrics: {outdir / "table_4_2_three_groups_raw_metrics.json"}')


if __name__ == '__main__':
    main()