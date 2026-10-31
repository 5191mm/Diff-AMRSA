#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check whether the two MRSA training configs only differ in allowed fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ALLOWED_DIFF_PATHS = {
    "run_name",
    "outdir",
    "resume",
    "strict_compare.role",
    "strict_compare.require_resume",
}


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(value, next_prefix))
        return out
    if isinstance(obj, list):
        out: dict[str, Any] = {}
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            out.update(flatten(value, next_prefix))
        return out
    return {prefix: obj}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate strict MRSA scratch and finetune configs")
    parser.add_argument("--scratch", default="scripts/conf/drugs/mrsa_diffusion_scratch_scheme_a.yaml")
    parser.add_argument("--finetune", default="scripts/conf/drugs/mrsa_diffusion_finetune_scheme_a.yaml")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    with open(args.scratch, "r", encoding="utf-8") as f:
        scratch_cfg = yaml.safe_load(f)
    with open(args.finetune, "r", encoding="utf-8") as f:
        finetune_cfg = yaml.safe_load(f)
    scratch_flat = flatten(scratch_cfg)
    finetune_flat = flatten(finetune_cfg)

    all_keys = sorted(set(scratch_flat) | set(finetune_flat))
    unexpected = []
    allowed = []
    for key in all_keys:
        s_val = scratch_flat.get(key, "<MISSING>")
        f_val = finetune_flat.get(key, "<MISSING>")
        if s_val != f_val:
            item = {"path": key, "scratch": s_val, "finetune": f_val}
            if key in ALLOWED_DIFF_PATHS:
                allowed.append(item)
            else:
                unexpected.append(item)

    result = {
        "scratch": str(Path(args.scratch).resolve()),
        "finetune": str(Path(args.finetune).resolve()),
        "allowed_differences": allowed,
        "unexpected_differences": unexpected,
        "passed": len(unexpected) == 0,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if unexpected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
