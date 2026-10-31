#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect checkpoint hyperparameters for the strict MRSA comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


EXPECTED = {
    "optimizer_params.lr": 5e-5,
    "lr_scheduler_params.initial_lr": 1e-5,
    "lr_scheduler_params.final_lr": 5e-5,
    "lr_scheduler_params.num_warmup_steps": 500,
}


def get_nested(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur


def close(actual: Any, expected: Any) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= 1e-12
    except Exception:
        return actual == expected


def main() -> None:
    parser = argparse.ArgumentParser(description="Check strict checkpoint hyperparameters")
    parser.add_argument("ckpt")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})
    checks = []
    for key, expected in EXPECTED.items():
        actual = get_nested(hp, key)
        checks.append({"path": key, "actual": actual, "expected": expected, "passed": close(actual, expected)})
    result = {
        "checkpoint": str(ckpt_path.resolve()),
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
