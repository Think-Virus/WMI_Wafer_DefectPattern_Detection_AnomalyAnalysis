# scripts/build_unlabeled_db.py
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# --- make imports stable no matter where you run from ---
ROOT = Path(__file__).resolve().parents[1]  # wafermap-triage-studio/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmi_triage.config import Paths
from wmi_triage.preprocess import PreprocessConfig, preprocess_wafer
from wmi_triage.inference import load_model, load_class_to_idx, predict


# -----------------------------
# Helpers
# -----------------------------
def failuretype_to_label(ft) -> Optional[str]:
    """
    WM-811K failureType는 보통 list 형태거나 빈 list/None일 수 있음.
    return:
      - 라벨 있으면 문자열
      - 없으면 None
    """
    if ft is None:
        return None
    if isinstance(ft, np.ndarray):
        ft = ft.tolist()
    if isinstance(ft, (list, tuple)):
        if len(ft) == 0:
            return None
        a0 = ft[0]
        if isinstance(a0, (list, tuple, np.ndarray)):
            if len(a0) == 0:
                return None
            a0 = a0[0]
        s = str(a0).strip()
        return s if s else None
    s = str(ft).strip()
    return s if s else None


def pick_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def top1_cosine_to_known(
    q_emb: np.ndarray,
    ref_emb: np.ndarray,
    ref_df_index: np.ndarray,
    chunk: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    q_emb: (N,D) float32
    ref_emb: (M,D) float32
    return:
      best_sim: (N,)
      best_ref_df_index: (N,)
    """
    qn = l2norm(q_emb.astype(np.float32, copy=False))
    rn = l2norm(ref_emb.astype(np.float32, copy=False))

    best_sim = np.full((qn.shape[0],), -1.0, dtype=np.float32)
    best_idx = np.full((qn.shape[0],), -1, dtype=np.int64)

    # compute in chunks to avoid huge memory spikes
    for s in range(0, qn.shape[0], chunk):
        e = min(qn.shape[0], s + chunk)
        sims = qn[s:e] @ rn.T  # (B,M)
        arg = np.argmax(sims, axis=1)
        val = sims[np.arange(e - s), arg]
        best_sim[s:e] = val.astype(np.float32)
        best_idx[s:e] = ref_df_index[arg].astype(np.int64)

    return best_sim, best_idx


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, default="", help="data/LSWMD.pkl path (default: <root>/data/LSWMD.pkl)")
    ap.add_argument("--ckpt", type=str, default="", help="checkpoint path (default: pick one under artifacts/models)")
    ap.add_argument("--class-to-idx", type=str, default="", help="class_to_idx.json path (default: <models>/class_to_idx.json)")
    ap.add_argument("--out-npz", type=str, default="", help="output npz (default: <emb_db>/unlabeled_embeddings.npz)")
    ap.add_argument("--out-meta", type=str, default="", help="output meta json (default: <emb_db>/unlabeled_db_meta.json)")

    ap.add_argument("--n", type=int, default=5000, help="number of unlabeled samples to embed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)

    ap.add_argument("--resize", type=int, default=64)
    ap.add_argument("--input-mode", type=str, default="polar4", choices=["polar4", "coords4", "repeat3"])
    ap.add_argument("--upsample-mode", type=str, default="nearest", choices=["nearest", "bilinear", "bicubic"])
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")

    ap.add_argument("--known-db", type=str, default="", help="known_embeddings.npz (default: <emb_db>/known_embeddings.npz)")
    ap.add_argument("--no-known-sim", action="store_true", help="skip best_known_sim computation")
    ap.add_argument("--known-sim-chunk", type=int, default=512)

    args = ap.parse_args()

    P = Paths()
    device = pick_device(args.device)

    pkl_path = Path(args.pkl) if args.pkl else (P.root / "data" / "LSWMD.pkl")

    # default ckpt: pick first *.pt under models if not given
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        pts = sorted(P.models.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No .pt found under {P.models}")
        ckpt_path = pts[0]

    class_to_idx_path = Path(args.class_to_idx) if args.class_to_idx else (P.models / "class_to_idx.json")
    out_npz = Path(args.out_npz) if args.out_npz else (P.emb_db / "unlabeled_embeddings.npz")
    out_meta = Path(args.out_meta) if args.out_meta else (P.emb_db / "unlabeled_db_meta.json")

    known_db_path = Path(args.known_db) if args.known_db else (P.emb_db / "known_embeddings.npz")

    out_npz.parent.mkdir(parents=True, exist_ok=True)

    print("[device]", device)
    print("[paths] pkl:", pkl_path)
    print("[paths] ckpt:", ckpt_path)
    print("[paths] class_to_idx:", class_to_idx_path)
    print("[paths] out_npz:", out_npz)

    # 1) load df
    df = pd.read_pickle(pkl_path)
    print("[data] total:", len(df))

    # 2) unlabeled indices (failureType 없음)
    ft = df["failureType"] if "failureType" in df.columns else None
    if ft is None:
        raise KeyError("df에 'failureType' 컬럼이 없어. unlabeled 정의를 바꿔야 함.")

    # apply is OK at this scale; do it once
    labels = ft.map(failuretype_to_label)
    unlabeled_mask = labels.isna()
    unlabeled_idx = df.index[unlabeled_mask].to_numpy()
    print("[data] unlabeled:", int(unlabeled_mask.sum()))

    # sample
    rng = np.random.default_rng(int(args.seed))
    n = int(min(args.n, len(unlabeled_idx)))
    pick_idx = rng.choice(unlabeled_idx, size=n, replace=False).astype(np.int64)
    print("[sample] n:", n, "seed:", int(args.seed))

    # 3) load model
    class_to_idx = load_class_to_idx(str(class_to_idx_path))
    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    print("[meta] num_classes:", num_classes)

    model = load_model(str(ckpt_path), num_classes=num_classes, device=device)

    # 4) embed loop
    cfg = PreprocessConfig(
        resize=int(args.resize),
        input_mode=str(args.input_mode),
        upsample_mode=str(args.upsample_mode),
        device=device,
    )

    bs = int(args.batch_size)
    embs: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    confs: List[np.ndarray] = []
    dfis: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for s in range(0, n, bs):
            batch_idx = pick_idx[s : s + bs]
            xs = []
            for dfi in batch_idx.tolist():
                wafer = df.loc[int(dfi), "waferMap"]
                if isinstance(wafer, list):
                    wafer = np.array(wafer)
                x = preprocess_wafer(np.array(wafer), cfg)  # (1,C,R,R) on cfg.device
                xs.append(x)
            x_batch = torch.cat(xs, dim=0)  # (B,C,R,R)

            out = predict(model, x_batch, device=device)
            emb = out["emb"].numpy().astype(np.float32)          # (B,512)
            pred_idx = out["pred"].numpy().astype(np.int64)      # (B,)
            conf = out["conf"].numpy().astype(np.float32)        # (B,)

            embs.append(emb)
            preds.append(pred_idx)
            confs.append(conf)
            dfis.append(batch_idx.astype(np.int64))

            if (s // bs) % 20 == 0:
                print(f"[embed] {s:6d}/{n} ...")

    emb_all = np.concatenate(embs, axis=0)
    pred_idx_all = np.concatenate(preds, axis=0)
    conf_all = np.concatenate(confs, axis=0)
    df_index_all = np.concatenate(dfis, axis=0)

    pred_label_all = np.array([idx_to_class.get(int(i), "UNKNOWN_IDX") for i in pred_idx_all], dtype=object)

    # 5) best_known_sim (optional but very useful)
    best_known_sim = None
    best_known_df_index = None

    if (not args.no_known_sim) and known_db_path.exists():
        kob = np.load(known_db_path, allow_pickle=True)
        ref_emb = kob["emb"].astype(np.float32)
        ref_df_index = kob["df_index"].astype(np.int64)

        print("[known_sim] computing top1 cosine to known DB ...")
        best_known_sim, best_known_df_index = top1_cosine_to_known(
            emb_all,
            ref_emb,
            ref_df_index,
            chunk=int(args.known_sim_chunk),
        )
        print("[known_sim] done.")
    else:
        print("[known_sim] skipped (flag or missing known DB)")

    # 6) save
    save_kwargs: Dict[str, Any] = dict(
        emb=emb_all,
        df_index=df_index_all,
        pred_idx=pred_idx_all,
        pred_label=pred_label_all,
        conf=conf_all,
    )
    if best_known_sim is not None and best_known_df_index is not None:
        save_kwargs["best_known_sim"] = best_known_sim.astype(np.float32)
        save_kwargs["best_known_df_index"] = best_known_df_index.astype(np.int64)

    np.savez_compressed(out_npz, **save_kwargs)

    meta = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n": int(n),
        "seed": int(args.seed),
        "device": device,
        "pkl_path": str(pkl_path),
        "ckpt_path": str(ckpt_path),
        "class_to_idx_path": str(class_to_idx_path),
        "preprocess": {
            "resize": int(args.resize),
            "input_mode": str(args.input_mode),
            "upsample_mode": str(args.upsample_mode),
        },
        "outputs": {
            "npz": str(out_npz),
            "meta": str(out_meta),
        },
        "known_sim": {
            "enabled": bool(best_known_sim is not None),
            "known_db": str(known_db_path),
            "chunk": int(args.known_sim_chunk),
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[saved] npz:", out_npz)
    print("[saved] meta:", out_meta)
    print("[npz] emb:", emb_all.shape, "df_index:", df_index_all.shape, "pred_label:", pred_label_all.shape)

    # quick peek
    print("[peek] pred_label top5:")
    from collections import Counter
    c = Counter(pred_label_all.tolist())
    for k, v in c.most_common(5):
        print("  -", k, v)


if __name__ == "__main__":
    main()
