#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from generate_table_4_4_domain_consistency import compute_domain_table


def main():
    parser = argparse.ArgumentParser(description='Generate three-group anti-MRSA domain consistency table')
    parser.add_argument('--pretrained_baseline_csv', required=True)
    parser.add_argument('--scratch_csv', required=True)
    parser.add_argument('--finetune_csv', required=True)
    parser.add_argument('--reference_csv', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--generated_smiles_col', default='smiles')
    parser.add_argument('--reference_smiles_col', default='smiles')
    parser.add_argument('--top_n', type=int, default=200)
    parser.add_argument('--active_only', action='store_true')
    parser.add_argument('--activity_col', default='activity_label')
    parser.add_argument('--n_bits', type=int, default=2048)
    parser.add_argument('--top_fragments', type=int, default=30)
    parser.add_argument('--high_score_threshold', type=float, default=0.7)
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
    fragment_frames = []
    raw_metrics = {}

    for group_name, csv_path in groups:
        table_df, metrics, generated_df, fragment_df = compute_domain_table(
            generated_csv=csv_path,
            reference_csv=Path(args.reference_csv),
            generated_smiles_col=args.generated_smiles_col,
            reference_smiles_col=args.reference_smiles_col,
            top_n=args.top_n,
            active_only=args.active_only,
            activity_col=args.activity_col,
            n_bits=args.n_bits,
            top_fragments=args.top_fragments,
            high_score_threshold=args.high_score_threshold,
        )

        raw_metrics[group_name] = metrics

        summary_rows.append({
            'Model': group_name,
            'Generated molecules': metrics['generated_molecules'],
            'Reference molecules': metrics['reference_molecules'],
            'Mean maximum Tanimoto': metrics['mean_max_tanimoto'],
            'Median maximum Tanimoto': metrics['median_max_tanimoto'],
            'Scaffold coverage': metrics['scaffold_coverage'],
            'New scaffold ratio': metrics['new_scaffold_ratio'],
            'Key fragment coverage': metrics['key_fragment_coverage'],
            'Mean activity proxy score': metrics['mean_activity_proxy_score'],
            'High-score candidate ratio': metrics['high_score_candidate_ratio'],
            'High-score threshold': metrics['high_score_threshold'],
            'Activity proxy source': metrics['activity_proxy_source'],
        })

        table_df.insert(0, 'Model', group_name)
        long_rows.append(table_df)

        generated_df = generated_df.copy()
        generated_df.insert(0, 'Model', group_name)
        detail_frames.append(generated_df)

        fragment_df = fragment_df.copy()
        fragment_df.insert(0, 'Model', group_name)
        fragment_frames.append(fragment_df)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / 'table_4_4_three_groups_summary.csv', index=False, encoding='utf-8-sig')

    long_df = pd.concat(long_rows, ignore_index=True)
    long_df.to_csv(outdir / 'table_4_4_three_groups_long.csv', index=False, encoding='utf-8-sig')

    detail_df = pd.concat(detail_frames, ignore_index=True)
    detail_df.to_csv(outdir / 'table_4_4_three_groups_generated_detail.csv', index=False, encoding='utf-8-sig')

    fragment_detail_df = pd.concat(fragment_frames, ignore_index=True)
    fragment_detail_df.to_csv(outdir / 'table_4_4_three_groups_fragment_coverage.csv', index=False, encoding='utf-8-sig')

    with open(outdir / 'table_4_4_three_groups_raw_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(raw_metrics, f, indent=2, ensure_ascii=False)

    print(f'Saved summary table: {outdir / "table_4_4_three_groups_summary.csv"}')
    print(f'Saved long table: {outdir / "table_4_4_three_groups_long.csv"}')
    print(f'Saved generated detail: {outdir / "table_4_4_three_groups_generated_detail.csv"}')
    print(f'Saved fragment coverage: {outdir / "table_4_4_three_groups_fragment_coverage.csv"}')
    print(f'Saved raw metrics: {outdir / "table_4_4_three_groups_raw_metrics.json"}')


if __name__ == '__main__':
    main()