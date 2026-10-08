"""Compatibility shim for checkpoints or legacy imports that reference ``megalodon``.

Diff-AMRSA uses ``diff_amrsa`` as the canonical Python package name because
hyphens are not valid in Python import names. This shim keeps legacy
Megalodon checkpoint loading and old scripts usable without duplicating code.
New code should import from ``diff_amrsa``.
"""
from __future__ import annotations

import diff_amrsa as _diff_amrsa

__path__ = _diff_amrsa.__path__
__all__ = getattr(_diff_amrsa, "__all__", [])

def __getattr__(name):
    return getattr(_diff_amrsa, name)
