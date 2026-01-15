# wmi_triage/preprocess.py
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PreprocessConfig:
    resize: int = 64
    input_mode: str = "polar4"  # "polar4" | "coords4" | "repeat3"
    upsample_mode: str = "nearest"  # waferMap 리사이즈는 nearest 권장
    device: str = "cpu"


_polar_cache = {}  # resize별 (grid,pos)
_coords_cache = {}  # resize별 coords


def _make_polar_grid_and_pos(resize: int, device="cpu", dtype=torch.float32):
    """
    output:
      polar_grid: (1,H,W,2)  grid_sample용 (x,y) in [-1,1]
      polar_pos : (3,H,W)    [r, sin(theta), cos(theta)]
    """
    H = W = resize
    r = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)  # (H,)
    theta = torch.linspace(-math.pi, math.pi, W, device=device, dtype=dtype)  # (W,)

    rr = r[:, None].expand(H, W)  # (H,W)
    tt = theta[None, :].expand(H, W)  # (H,W)

    xx = rr * torch.cos(tt)  # (H,W) in [-1,1]
    yy = rr * torch.sin(tt)  # (H,W) in [-1,1]

    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # (1,H,W,2)
    pos = torch.stack([rr, torch.sin(tt), torch.cos(tt)], 0)  # (3,H,W)
    return grid, pos


def _make_coords(resize: int, device="cpu", dtype=torch.float32):
    g = torch.linspace(-1, 1, resize, device=device, dtype=dtype)
    try:
        yy, xx = torch.meshgrid(g, g, indexing="ij")
    except TypeError:
        yy, xx = torch.meshgrid(g, g)
    rr = torch.sqrt(xx ** 2 + yy ** 2)
    coords = torch.stack([xx, yy, rr], dim=0)  # (3,R,R)
    return coords


def preprocess_wafer(wafer: np.ndarray, cfg: PreprocessConfig) -> torch.Tensor:
    device = cfg.device
    resize = int(cfg.resize)

    wm = torch.as_tensor(np.array(wafer), dtype=torch.float32, device=device)  # (h,w)
    wm = wm.unsqueeze(0).unsqueeze(0)  # (1,1,h,w)

    # resize (discrete map -> nearest)
    if wm.shape[-2:] != (resize, resize):
        wm = F.interpolate(wm, size=(resize, resize), mode=cfg.upsample_mode)

    wm = wm.squeeze(0)  # (1,R,R)

    if cfg.input_mode == "polar4":
        # cache polar grid/pos
        if resize not in _polar_cache:
            _polar_cache[resize] = _make_polar_grid_and_pos(resize, device=device, dtype=wm.dtype)
        polar_grid, polar_pos = _polar_cache[resize]

        # waferMap is discrete -> round -> defect mask
        wm_int = wm.round()
        defect = (wm_int == 2).float()  # (1,R,R)

        # polar transform (nearest)
        polar_defect = F.grid_sample(
            defect.unsqueeze(0),  # (1,1,R,R)
            polar_grid,  # (1,R,R,2)
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)  # (1,R,R)

        x = torch.cat([polar_defect, polar_pos], dim=0)  # (4,R,R)

    elif cfg.input_mode == "coords4":
        if resize not in _coords_cache:
            _coords_cache[resize] = _make_coords(resize, device=device, dtype=wm.dtype)
        coords = _coords_cache[resize]  # (3,R,R)
        x = torch.cat([wm, coords], dim=0)  # (4,R,R)

    elif cfg.input_mode == "repeat3":
        x = wm.repeat(3, 1, 1)  # (3,R,R)

    else:
        raise ValueError(f"Unknown input_mode: {cfg.input_mode}")

    return x.unsqueeze(0)  # (1,C,R,R)
