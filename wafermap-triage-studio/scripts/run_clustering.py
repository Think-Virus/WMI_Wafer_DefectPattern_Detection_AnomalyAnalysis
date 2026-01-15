# scripts/run_clustering.py
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # wafermap-triage-studio/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmi_triage.config import Paths


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def reps_by_centroid(emb_n: np.ndarray, cluster_id: np.ndarray) -> dict[int, int]:
    """
    cluster별 centroid(평균)와 cosine(sim=dot) 최대 샘플을 rep으로 선택
    return: {cid: member_pos}  # member_pos: emb row index
    """
    reps: dict[int, int] = {}
    for cid in sorted(set(cluster_id.tolist())):
        if cid == -1:
            continue
        members = np.where(cluster_id == cid)[0]
        if len(members) == 0:
            continue
        centroid = emb_n[members].mean(axis=0, keepdims=True)
        centroid = l2norm(centroid)[0]
        sims = emb_n[members] @ centroid
        rep_pos = int(members[int(np.argmax(sims))])
        reps[int(cid)] = rep_pos
    return reps


def run_dbscan(emb_n: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    return db.fit_predict(emb_n).astype(np.int32)


def run_kmeans(emb_n: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """
    l2-normalized emb에 KMeans(Euclidean) -> cosine-ish clustering 효과
    """
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    return km.fit_predict(emb_n).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "dbscan"])
    ap.add_argument("--k", type=int, default=2, help="kmeans clusters (method=kmeans)")
    ap.add_argument("--eps", type=float, default=0.05, help="dbscan eps (method=dbscan)")
    ap.add_argument("--min-samples", type=int, default=4, help="dbscan min_samples (method=dbscan)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    P = Paths()
    unk_npz = P.emb_db / "unknown_embeddings.npz"
    out_npz = P.emb_db / "unknown_cluster.npz"
    out_reps = P.emb_db / "unknown_cluster_reps.json"
    out_meta = P.emb_db / "unknown_cluster_meta.json"

    if not unk_npz.exists():
        raise FileNotFoundError(f"missing: {unk_npz}")

    obj = np.load(unk_npz, allow_pickle=True)
    emb = obj["emb"].astype(np.float32)
    df_index = obj["df_index"].astype(np.int64)
    true_label = obj["true_label"] if "true_label" in obj.files else None

    emb_n = l2norm(emb)

    # clustering
    if args.method == "dbscan":
        cluster_id = run_dbscan(emb_n, eps=args.eps, min_samples=args.min_samples)
        params = {"eps": float(args.eps), "min_samples": int(args.min_samples)}
    else:
        cluster_id = run_kmeans(emb_n, k=args.k, seed=args.seed)
        params = {"k": int(args.k), "seed": int(args.seed)}

    cnt = Counter(cluster_id.tolist())
    n_clusters = len([k for k in cnt.keys() if k != -1])
    noise = cnt.get(-1, 0)

    print("[load] emb:", emb.shape, "df_index:", df_index.shape)
    if true_label is not None:
        print("[labels]", Counter([str(x) for x in true_label.tolist()]))
    print(f"[cluster] method={args.method} -> clusters(excl noise)={n_clusters}, noise={noise}({noise / len(cluster_id):.1%})")
    print("[cluster] top counts:", dict(sorted(cnt.items(), key=lambda kv: kv[1], reverse=True)[:10]))

    # reps
    reps_pos = reps_by_centroid(emb_n, cluster_id)
    reps_df_index = {int(cid): int(df_index[pos]) for cid, pos in reps_pos.items()}

    # save
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        cluster_id=cluster_id.astype(np.int32),
        df_index=df_index.astype(np.int64),
        method=np.array([args.method], dtype="<U10"),
        **{k: np.array([v]) for k, v in params.items()},
    )
    out_reps.write_text(json.dumps(reps_df_index, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "source_npz": str(unk_npz),
        "saved_npz": str(out_npz),
        "saved_reps_json": str(out_reps),
        "method": args.method,
        "params": params,
        "num_samples": int(len(cluster_id)),
        "num_clusters_excluding_noise": int(n_clusters),
        "noise_count": int(noise),
        "cluster_counts_top10": {int(k): int(v) for k, v in sorted(cnt.items(), key=lambda kv: kv[1], reverse=True)[:10]},
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[saved]", out_npz)
    print("[saved]", out_reps)
    print("[saved]", out_meta)


if __name__ == "__main__":
    main()
