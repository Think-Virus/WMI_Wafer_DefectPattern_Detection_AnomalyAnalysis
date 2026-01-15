# wmi_triage/ood.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Literal, Optional

import torch

OodMethod = Literal["none", "msp", "energy"]


@dataclass
class OodResult:
    method: str
    msp: float
    energy: float
    threshold: Optional[float]
    temperature: float
    is_ood: Optional[bool]


def msp_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """MSP = max softmax prob (ID일수록 커지는 경향)."""
    prob = torch.softmax(logits, dim=1)
    return prob.max(dim=1).values


def energy_from_logits(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """
    Energy score (Liu et al.):
      E(x) = -T * logsumexp(logits / T)

    일반적으로
      - ID(known): logsumexp가 커져서 energy가 더 '작아짐'(더 negative)
      - OOD(unknown): energy가 더 '커짐'(덜 negative)
    => energy가 클수록 OOD로 보는 규칙이 자연스럽다.
    """
    return -T * torch.logsumexp(logits / T, dim=1)


def decide_ood(
        logits: torch.Tensor,
        method: OodMethod = "energy",
        threshold: Optional[float] = None,
        temperature: float = 1.0,
) -> OodResult:
    """
    threshold가 None이면 점수만 계산하고 is_ood는 None으로 둔다.
    - MSP 기준: is_ood = (msp < threshold)
    - Energy 기준: is_ood = (energy > threshold)
    """
    logits = logits.detach().float()

    msp = float(msp_from_logits(logits)[0].cpu().item())
    energy = float(energy_from_logits(logits, T=temperature)[0].cpu().item())

    is_ood: Optional[bool] = None
    if method == "msp" and threshold is not None:
        is_ood = (msp < float(threshold))
    elif method == "energy" and threshold is not None:
        is_ood = (energy > float(threshold))
    elif method == "none":
        is_ood = None

    return OodResult(
        method=method,
        msp=msp,
        energy=energy,
        threshold=None if threshold is None else float(threshold),
        temperature=float(temperature),
        is_ood=is_ood,
    )


def to_jsonable(r: OodResult) -> Dict[str, Any]:
    return asdict(r)
