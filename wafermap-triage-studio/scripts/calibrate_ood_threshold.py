# scripts/calibrate_ood_threshold.py
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from wmi_triage.config import Paths
from wmi_triage.preprocess import PreprocessConfig, preprocess_wafer
from wmi_triage.inference import load_model


# -----------------------------
# Dataset
# -----------------------------
class IndexDataset(Dataset):
    """
    df.index에 원본 df_index가 있는 전제에서,
    주어진 df_index 리스트를 꺼내서 preprocess까지 해서 텐서로 반환.
    """

    def __init__(self, df: pd.DataFrame, df_indices: np.ndarray, cfg: PreprocessConfig):
        self.df = df
        self.df_indices = df_indices.astype(np.int64)
        self.cfg = cfg

    def __len__(self):
        return len(self.df_indices)

    def __getitem__(self, i: int):
        dfi = int(self.df_indices[i])
        row = self.df.loc[dfi]
        wafer = row["waferMap"]
        if isinstance(wafer, list):
            wafer = np.array(wafer)
        else:
            wafer = np.array(wafer)

        x = preprocess_wafer(wafer, self.cfg)  # (1,C,H,W)
        x = x.squeeze(0)  # (C,H,W) -> DataLoader가 batch로 (B,C,H,W)
        return x, dfi


@torch.no_grad()
def infer_scores(
        model,
        loader: DataLoader,
        device: str,
        method: str,
        temperature: float,
) -> np.ndarray:
    """
    method:
      - "energy": score가 클수록 OOD
      - "msp":    score가 작을수록 OOD (여기서는 그대로 MSP 반환)
    """
    scores: List[np.ndarray] = []

    for x, _dfi in loader:
        x = x.to(device, non_blocking=True)
        logits, _emb = model(x)  # (B, num_classes)

        if method == "energy":
            # energy = -T * logsumexp(logits/T)
            t = float(temperature)
            e = -t * torch.logsumexp(logits / t, dim=1)  # (B,)
            scores.append(e.detach().cpu().numpy().astype(np.float32))

        elif method == "msp":
            prob = F.softmax(logits, dim=1)
            msp = prob.max(dim=1).values  # (B,)
            scores.append(msp.detach().cpu().numpy().astype(np.float32))

        else:
            raise ValueError(f"Unknown method: {method}")

    return np.concatenate(scores, axis=0)


def quantiles(x: np.ndarray, qs: List[float]) -> Dict[str, float]:
    out = {}
    for q in qs:
        out[str(q)] = float(np.quantile(x, q))
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--method", type=str, default="energy", choices=["energy", "msp"])
    ap.add_argument("--temperature", type=float, default=1.0)

    ap.add_argument("--target-fpr", type=float, default=0.20, help="known을 OOD로 오판할 비율")
    ap.add_argument("--n-known", type=int, default=2000, help="0이면 전부 사용")
    ap.add_argument("--n-unknown", type=int, default=1000, help="0이면 전부 사용")

    ap.add_argument("--resize", type=int, default=64)
    ap.add_argument("--input-mode", type=str, default="polar4")
    ap.add_argument("--upsample-mode", type=str, default="nearest")

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)  # Windows 안전 기본값
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    ap.add_argument("--out-json", type=str, default=None)
    args = ap.parse_args()

    P = Paths()

    lswmd_pkl = P.root / "data" / "LSWMD.pkl"
    ckpt = P.models / "20260111_074510_resnet18_R64_polar4_K6_best_e05_weights.pt"
    class_to_idx_path = P.models / "class_to_idx.json"

    known_npz = P.emb_db / "known_embeddings.npz"
    unknown_npz = P.emb_db / "unknown_embeddings.npz"

    if ckpt is None:
        raise RuntimeError("--ckpt는 꼭 지정해줘 (GitHub 용량 이슈 때문에 파일명이 바뀌기 쉬움).")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("[device]", device)
    print("[paths] pkl:", lswmd_pkl)
    print("[paths] ckpt:", ckpt)
    print("[paths] class_to_idx:", class_to_idx_path)
    print("[paths] known_npz:", known_npz)
    print("[paths] unknown_npz:", unknown_npz)

    # load df
    df = pd.read_pickle(lswmd_pkl)
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="first")].copy()
    print("[data] df rows:", len(df), "index dtype:", df.index.dtype)

    # load meta
    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    print("[meta] num_classes:", num_classes)

    # load indices
    kobj = np.load(known_npz, allow_pickle=True)
    uobj = np.load(unknown_npz, allow_pickle=True)
    known_idx_all = kobj["df_index"].astype(np.int64)
    unknown_idx_all = uobj["df_index"].astype(np.int64)

    # filter by df coverage (small dataset에서 중요)
    known_idx_all = known_idx_all[np.isin(known_idx_all, df.index.values)]
    unknown_idx_all = unknown_idx_all[np.isin(unknown_idx_all, df.index.values)]

    rng = np.random.RandomState(int(args.seed))

    def sample(arr: np.ndarray, n: int) -> np.ndarray:
        if n <= 0 or n >= len(arr):
            return arr
        return rng.choice(arr, size=n, replace=False).astype(np.int64)

    known_idx = sample(known_idx_all, int(args.n_known))
    unknown_idx = sample(unknown_idx_all, int(args.n_unknown))

    print("[sample] known:", len(known_idx), "/", len(known_idx_all))
    print("[sample] unknown:", len(unknown_idx), "/", len(unknown_idx_all))

    # model
    model = load_model(str(ckpt), num_classes=num_classes, device=device)
    model.eval()

    # preprocess cfg (build_known/unknown_db와 동일 계열)
    cfg = PreprocessConfig(
        resize=int(args.resize),
        input_mode=str(args.input_mode),
        upsample_mode=str(args.upsample_mode),
        device=device,
    )

    # loaders
    known_ds = IndexDataset(df, known_idx, cfg)
    unknown_ds = IndexDataset(df, unknown_idx, cfg)

    known_loader = DataLoader(
        known_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device == "cuda"),
    )
    unknown_loader = DataLoader(
        unknown_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device == "cuda"),
    )

    method = str(args.method)
    T = float(args.temperature)

    print("[infer] scoring known ...")
    known_scores = infer_scores(model, known_loader, device=device, method=method, temperature=T)

    print("[infer] scoring unknown ...")
    unknown_scores = infer_scores(model, unknown_loader, device=device, method=method, temperature=T)

    # threshold by target FPR
    target_fpr = float(args.target_fpr)

    if method == "energy":
        # OOD if energy > thr
        thr = float(np.quantile(known_scores, 1.0 - target_fpr))
        fpr = float(np.mean(known_scores > thr))
        tpr = float(np.mean(unknown_scores > thr))

        # AUROC needs "higher => more OOD"
        from sklearn.metrics import roc_auc_score
        y = np.concatenate([np.zeros_like(known_scores), np.ones_like(unknown_scores)])
        s = np.concatenate([known_scores, unknown_scores])
        auroc = float(roc_auc_score(y, s))

    else:  # msp
        # OOD if msp < thr
        thr = float(np.quantile(known_scores, target_fpr))
        fpr = float(np.mean(known_scores < thr))
        tpr = float(np.mean(unknown_scores < thr))

        # AUROC: higher => more OOD => use (1 - msp)
        from sklearn.metrics import roc_auc_score
        y = np.concatenate([np.zeros_like(known_scores), np.ones_like(unknown_scores)])
        s = np.concatenate([1.0 - known_scores, 1.0 - unknown_scores])
        auroc = float(roc_auc_score(y, s))

    print("[result] method:", method)
    print("[result] threshold:", thr)
    print("[result] target_fpr:", target_fpr, "measured_fpr:", fpr)
    print("[result] tpr(unknown recall):", tpr)
    print("[result] auroc:", auroc)

    out = {
        "method": method,
        "threshold": thr,
        "temperature": T,
        "target_fpr": target_fpr,
        "measured_fpr": fpr,
        "tpr": tpr,
        "auroc": auroc,
        "n_known": int(len(known_scores)),
        "n_unknown": int(len(unknown_scores)),
        "score_quantiles": {
            "known": quantiles(known_scores, [0.01, 0.05, 0.5, 0.95, 0.99]),
            "unknown": quantiles(unknown_scores, [0.01, 0.05, 0.5, 0.95, 0.99]),
        },
        "paths": {
            "lswmd_pkl": str(lswmd_pkl),
            "ckpt": str(ckpt),
            "class_to_idx": str(class_to_idx_path),
            "known_npz": str(known_npz),
            "unknown_npz": str(unknown_npz),
        },
        "preprocess": {
            "resize": int(args.resize),
            "input_mode": str(args.input_mode),
            "upsample_mode": str(args.upsample_mode),
        },
        "seed": int(args.seed),
    }

    out_json = P.models / f"ood_threshold_{method}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[saved]", out_json)


if __name__ == "__main__":
    main()
