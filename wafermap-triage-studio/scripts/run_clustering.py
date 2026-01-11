# scripts/run_clustering.py
from __future__ import annotations

import argparse
import json
from collections import Counter
import numpy as np
from pathlib import Path

from sklearn.cluster import DBSCAN

from wmi_triage.config import Paths


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def choose_representatives(emb_n: np.ndarray, cluster_id: np.ndarray) -> dict[int, int]:
    """
    노트북 방식:
    - cluster별 centroid(평균) 구하고
    - centroid와 cosine(sim=dot) 가장 큰 샘플을 rep으로 선택
    return: {cluster_id: member_pos}  # member_pos는 emb 배열의 row index
    """
    reps: dict[int, int] = {}
    for cid in sorted(set(cluster_id.tolist())):
        if cid == -1:
            continue
        members = np.where(cluster_id == cid)[0]
        if len(members) == 0:
            continue

        centroid = emb_n[members].mean(axis=0, keepdims=True)
        centroid = l2norm(centroid)[0]  # (D,)
        sims = emb_n[members] @ centroid  # (n_members,)
        rep_pos = int(members[int(np.argmax(sims))])
        reps[int(cid)] = rep_pos
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.25, help="DBSCAN eps (cosine distance)")
    ap.add_argument("--min-samples", type=int, default=10, help="DBSCAN min_samples")
    args = ap.parse_args()

    P = Paths()
    unk_npz = P.emb_db / "unknown_embeddings.npz"
    out_npz = P.emb_db / "unknown_cluster.npz"
    out_reps = P.emb_db / "unknown_cluster_reps.json"
    out_meta = P.emb_db / "unknown_cluster_meta.json"

    if not unk_npz.exists():
        raise FileNotFoundError(f"unknown_embeddings.npz not found: {unk_npz}")

    obj = np.load(unk_npz, allow_pickle=True)
    emb = obj["emb"].astype(np.float32)  # (N,D)
    df_index = obj["df_index"].astype(np.int64)  # (N,)
    true_label = obj["true_label"] if "true_label" in obj.files else None

    print("[load] emb:", emb.shape, "df_index:", df_index.shape)

    emb_n = l2norm(emb)

    # DBSCAN(metric="cosine") expects cosine distance = 1 - cosine_sim
    db = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="cosine")
    cluster_id = db.fit_predict(emb_n).astype(np.int32)

    cnt = Counter(cluster_id.tolist())
    n_clusters = len([k for k in cnt.keys() if k != -1])
    print("[cluster] num_clusters(excl noise):", n_clusters, "noise:", cnt.get(-1, 0), "total:", len(cluster_id))
    print("[cluster] top counts:", dict(sorted(cnt.items(), key=lambda kv: kv[1], reverse=True)[:10]))

    reps_pos = choose_representatives(emb_n, cluster_id)
    print("[reps] num reps:", len(reps_pos), "sample:", list(reps_pos.items())[:10])

    # reps를 df_index로 바꿔 저장 (나중에 원본 df에서 바로 찾기 좋음)
    reps_df_index = {int(cid): int(df_index[pos]) for cid, pos in reps_pos.items()}

    # 저장
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        cluster_id=cluster_id,
        df_index=df_index,
        eps=np.array([args.eps], dtype=np.float32),
        min_samples=np.array([args.min_samples], dtype=np.int32),
    )

    out_reps.write_text(json.dumps(reps_df_index, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "source_npz": str(unk_npz),
        "saved_npz": str(out_npz),
        "saved_reps_json": str(out_reps),
        "eps": float(args.eps),
        "min_samples": int(args.min_samples),
        "num_samples": int(len(cluster_id)),
        "num_clusters_excluding_noise": int(n_clusters),
        "noise_count": int(cnt.get(-1, 0)),
        "cluster_counts_top10": {int(k): int(v) for k, v in sorted(cnt.items(), key=lambda kv: kv[1], reverse=True)[:10]},
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ saved:", out_npz)
    print("✅ saved:", out_reps)
    print("✅ saved:", out_meta)


if __name__ == "__main__":
    main()
