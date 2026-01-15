# scripts/build_cluster_bank.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmi_triage.config import Paths


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def build_known_centroids(known_npz: Path, idx_to_class: Dict[int, str]) -> Tuple[np.ndarray, List[str], List[str], np.ndarray, np.ndarray]:
    obj = np.load(known_npz, allow_pickle=True)
    emb = obj["emb"].astype(np.float32)
    y = obj["y"].astype(np.int64)
    df_index = obj["df_index"].astype(np.int64)

    emb_n = l2norm(emb)

    centroids = []
    names = []
    types = []
    reps = []
    counts = []

    for cls_idx in sorted(np.unique(y).tolist()):
        m = (y == cls_idx)
        E = emb_n[m]
        D = df_index[m]
        if len(D) == 0:
            continue

        c = l2norm(E.mean(axis=0, keepdims=True))[0]
        sim = E @ c
        rep_df = int(D[int(np.argmax(sim))])

        centroids.append(c)
        names.append(f"known:{idx_to_class.get(int(cls_idx), str(int(cls_idx)))}")
        types.append("known_class")
        reps.append(rep_df)
        counts.append(int(len(D)))

    return np.stack(centroids), names, types, np.array(reps, np.int64), np.array(counts, np.int32)


def build_unknown_cluster_centroids(
        unk_emb_npz: Path,
        unk_cluster_npz: Path,
        unk_reps_json: Path,
) -> Tuple[np.ndarray, List[str], List[str], np.ndarray, np.ndarray]:
    eobj = np.load(unk_emb_npz, allow_pickle=True)
    emb = eobj["emb"].astype(np.float32)
    df_index = eobj["df_index"].astype(np.int64)
    emb_n = l2norm(emb)

    # df_index -> row
    dfi2pos = {int(dfi): i for i, dfi in enumerate(df_index.tolist())}

    cobj = np.load(unk_cluster_npz, allow_pickle=True)
    c_df = cobj["df_index"].astype(np.int64)
    cid = cobj["cluster_id"].astype(np.int32)

    reps_map = {}
    if unk_reps_json.exists():
        reps_map = json.loads(unk_reps_json.read_text(encoding="utf-8"))

    centroids = []
    names = []
    types = []
    reps = []
    counts = []

    for k in sorted(np.unique(cid).tolist()):
        k = int(k)
        if k == -1:
            continue
        members = c_df[cid == k].tolist()
        pos = [dfi2pos.get(int(dfi), None) for dfi in members]
        pos = [p for p in pos if p is not None]
        if not pos:
            continue

        E = emb_n[pos]
        c = l2norm(E.mean(axis=0, keepdims=True))[0]

        # 대표 df_index는 reps_json 우선, 없으면 centroid-closest
        rep_df = reps_map.get(str(k), reps_map.get(k, None))
        if rep_df is None:
            sim = E @ c
            rep_pos = int(np.argmax(sim))
            rep_df = int(df_index[pos[rep_pos]])
        rep_df = int(rep_df)

        centroids.append(c)
        names.append(f"unknown_cluster:{k}")
        types.append("unknown_cluster")
        reps.append(rep_df)
        counts.append(int(len(pos)))

    return np.stack(centroids), names, types, np.array(reps, np.int64), np.array(counts, np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="default: <emb_db>/cluster_bank.npz")
    args = ap.parse_args()

    P = Paths()

    known_npz = P.emb_db / "known_embeddings.npz"
    unk_emb_npz = P.emb_db / "unknown_embeddings.npz"
    unk_cluster_npz = P.emb_db / "unknown_cluster.npz"
    unk_reps_json = P.emb_db / "unknown_cluster_reps.json"
    class_to_idx_json = P.models / "class_to_idx.json"

    if not known_npz.exists():
        raise FileNotFoundError(f"missing: {known_npz}")
    if not unk_emb_npz.exists():
        raise FileNotFoundError(f"missing: {unk_emb_npz}")
    if not unk_cluster_npz.exists():
        raise FileNotFoundError(f"missing: {unk_cluster_npz}")
    if not class_to_idx_json.exists():
        raise FileNotFoundError(f"missing: {class_to_idx_json}")

    class_to_idx = json.loads(class_to_idx_json.read_text(encoding="utf-8"))
    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}

    kc, kn, kt, kr, kcnt = build_known_centroids(known_npz, idx_to_class)
    uc, un, ut, ur, ucnt = build_unknown_cluster_centroids(unk_emb_npz, unk_cluster_npz, unk_reps_json)

    centroids = np.concatenate([kc, uc], axis=0).astype(np.float32)
    names = np.array(kn + un, dtype=object)
    types = np.array(kt + ut, dtype=object)
    rep_df_index = np.concatenate([kr, ur], axis=0).astype(np.int64)
    counts = np.concatenate([kcnt, ucnt], axis=0).astype(np.int32)

    out = Path(args.out) if args.out else (P.emb_db / "cluster_bank.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, centroid=centroids, name=names, type=types, rep_df_index=rep_df_index, count=counts)

    print("[saved]", out)
    print("[bank] n_clusters:", len(names))
    print("[bank] types:", {t: int((types == t).sum()) for t in np.unique(types)})


if __name__ == "__main__":
    main()
