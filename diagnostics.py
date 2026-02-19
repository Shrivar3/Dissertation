# Skilling's Heuristic
from __future__ import annotations
import numpy as np
from typing import Dict

def skilling_sd_logZ(ns_out: Dict[str, object], n_live: int) -> float:
    H = float(ns_out["H"])
    return float(np.sqrt(H / float(n_live)))

def print_ns_summary(ns_out: Dict[str, object], n_live: int) -> None:
    H = float(ns_out["H"])
    print("n_live =", int(n_live))
    print("H =", H)
    print("Skilling sd(logZ) =", float(np.sqrt(H / float(n_live))))
