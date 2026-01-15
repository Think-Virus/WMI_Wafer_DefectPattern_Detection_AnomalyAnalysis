# scripts/build_lswmd_streamlit.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd
from PIL import Image

from wmi_triage.config import Paths


def _load_df_index_from_npz(npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        return np.empty((0,), dtype=np.int64)
    obj = np.load(npz_path, allow_pickle=True)
    if "df_index" not in obj.files:
        return np.empty((0,), dtype=np.int64)
    return obj["df_index"].astype(np.int64)


def _collect_needed_indices(emb_db_dir: Path) -> np.ndarray:
    candidates = [
        emb_db_dir / "known_embeddings.npz",
        emb_db_dir / "unknown_embeddings.npz",
        emb_db_dir / "unlabeled_embeddings.npz",
        emb_db_dir / "unknown_cluster.npz",
        emb_db_dir / "unlabeled_cluster.npz",
    ]
    s: Set[int] = set()
    for p in candidates:
        arr = _load_df_index_from_npz(p)
        s.update(arr.tolist())
    return np.array(sorted(s), dtype=np.int64)


def _failuretype_to_label(ft):
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


def _resize_wafer_to_u8(w: np.ndarray, resize: int) -> np.ndarray:
    x = np.array(w)
    if x.ndim != 2:
        x = x.squeeze()
    if x.dtype != np.uint8:
        x = np.rint(x).astype(np.uint8, copy=False)
    pil = Image.fromarray(x, mode="L")
    pil = pil.resize((resize, resize), resample=Image.NEAREST)
    return np.array(pil, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resize", type=int, default=64, help="waferMap 고정 리사이즈")
    ap.add_argument("--seed", type=int, default=42, help="추가 샘플링 seed")
    ap.add_argument(
        "--target-rows",
        type=int,
        default=450000,
        help="0이면 필요한 idx만. >0이면 (필요 idx + 랜덤 추가)로 최종 rows를 target으로 맞춤",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="(옵션) 필요 idx를 앞에서부터 cap. target-rows와 같이 쓰지 않는 걸 권장",
    )
    args = ap.parse_args()

    P = Paths()
    pkl_path = P.root / "data" / "LSWMD.pkl"
    emb_db_dir = P.emb_db
    out_path = P.root / "data" / "LSWMD_small.pkl.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    need_idx = _collect_needed_indices(emb_db_dir)

    # (구버전 동작 유지: cap)
    if args.max_rows and args.max_rows > 0:
        need_idx = need_idx[: args.max_rows]

    print(f"[need] base indices: {len(need_idx)}")
    print("[load] reading:", pkl_path)
    df = pd.read_pickle(pkl_path)

    # ✅ target-rows: 필요한 idx는 유지 + 부족분은 랜덤으로 채움
    if args.target_rows and args.target_rows > 0:
        target = int(args.target_rows)
        if target < len(need_idx):
            print(f"[warn] target_rows({target}) < needed({len(need_idx)}). needed만 사용.")
        elif target > len(need_idx):
            n_extra = target - len(need_idx)
            rng = np.random.RandomState(int(args.seed))

            # 원본 df는 RangeIndex(0..N-1)라 iloc index space는 [0..len(df)-1]
            all_pos = np.arange(len(df), dtype=np.int64)
            mask = np.ones(len(df), dtype=bool)
            mask[need_idx] = False
            pool = all_pos[mask]

            if n_extra > len(pool):
                raise RuntimeError(f"extra too large: extra={n_extra}, pool={len(pool)}")

            extra = rng.choice(pool, size=n_extra, replace=False).astype(np.int64)
            need_idx = np.concatenate([need_idx, extra])
            need_idx = np.array(sorted(set(need_idx.tolist())), dtype=np.int64)

            print(f"[need] +extra: {n_extra} -> total indices: {len(need_idx)}")

    if len(need_idx) == 0:
        raise RuntimeError("need_idx is empty. emb_db에 npz들이 있는지 확인해줘.")

    sub = df.iloc[need_idx].copy()
    sub["df_index"] = need_idx

    wafers = []
    true_labels = []
    for _, row in sub.iterrows():
        wafers.append(_resize_wafer_to_u8(row["waferMap"], args.resize))
        true_labels.append(_failuretype_to_label(row.get("failureType", None)))

    out_df = pd.DataFrame({"waferMap": wafers, "failureType": true_labels})
    out_df.index = sub["df_index"].astype(np.int64)

    out_df.to_pickle(out_path, compression="gzip", protocol=4)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[saved] {out_path} ({size_mb:.2f} MB)")
    print("[peek] rows:", len(out_df), "cols:", list(out_df.columns))
    print("[peek] index dtype:", out_df.index.dtype)
    print("[peek] index min/max:", int(out_df.index.min()), int(out_df.index.max()))


if __name__ == "__main__":
    main()
