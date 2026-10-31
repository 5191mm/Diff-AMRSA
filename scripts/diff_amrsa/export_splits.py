#!/usr/bin/env python3
import argparse
import json
from diff_amrsa.dataset import export_processed_split


def main():
    p = argparse.ArgumentParser(description='Export exact train/val/test canonical structures from Megalodon processed files')
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--activity-table', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--processed-folder', default='processed')
    a = p.parse_args()
    print(json.dumps(export_processed_split(a.dataset_root, a.activity_table, a.output_dir, a.processed_folder), indent=2))


if __name__ == '__main__':
    main()
