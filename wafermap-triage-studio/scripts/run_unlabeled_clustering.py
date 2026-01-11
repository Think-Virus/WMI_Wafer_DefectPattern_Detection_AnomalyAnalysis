# scripts/run_unlabeled_clustering.py
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]  # wafermap-triage-studio/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmi_triage.config import Paths


def _norm_str(x) -> str:
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="ignore")
    return str(x).strip()


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def pick_representatives(emb_n: np.ndarray, df_index: np.ndarray, cluster_id: np.ndarray) -> Dict[str, int]:
    """
    cluster centroid(평균) 기준 cosine 최대 샘플을 대표로 선택
    emb_n: (N,D) l2-normalized
    """
    reps: Dict[str, int] = {}
    for cid in np.unique(cluster_id):
        m = cluster_id == cid
        E = emb_n[m]
        D = df_index[m]
        if len(D) == 0:
            continue
        c = l2norm(E.mean(axis=0, keepdims=True))[0]  # (D,)
        sim = E @ c
        j = int(np.argmax(sim))
        reps[str(int(cid))] = int(D[j])
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-npz", type=str, default="", help="default: <emb_db>/unlabeled_embeddings.npz")
    ap.add_argument("--out-npz", type=str, default="", help="default: <emb_db>/unlabeled_cluster.npz")
    ap.add_argument("--out-reps", type=str, default="", help="default: <emb_db>/unlabeled_cluster_reps.json")
    ap.add_argument("--out-meta", type=str, default="", help="default: <emb_db>/unlabeled_cluster_meta.json")

    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)

    # 후보군 필터
    ap.add_argument("--min-conf", type=float, default=0.70)

    # best_known_sim 절대 컷(있으면 적용). 보통 지금 상황에선 0.85 같은 값은 의미 없음.
    ap.add_argument("--max-known-sim", type=float, default=None)

    # (추천) best_known_sim 하위 퍼센타일로 후보군 선정 (예: 10 => 하위 10%만)
    ap.add_argument("--novel-pct", type=float, default=10.0)

    # pred_label로 제한하고 싶을 때 (comma-separated)
    ap.add_argument("--pred-label", type=str, default="", help="e.g. Loc,Edge-Loc")

    args = ap.parse_args()
    P = Paths()

    in_npz = Path(args.input_npz) if args.input_npz else (P.emb_db / "unlabeled_embeddings.npz")
    out_npz = Path(args.out_npz) if args.out_npz else (P.emb_db / "unlabeled_cluster.npz")
    out_reps = Path(args.out_reps) if args.out_reps else (P.emb_db / "unlabeled_cluster_reps.json")
    out_meta = Path(args.out_meta) if args.out_meta else (P.emb_db / "unlabeled_cluster_meta.json")
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    if not in_npz.exists():
        raise FileNotFoundError(f"missing: {in_npz}")

    obj = np.load(in_npz, allow_pickle=True)
    emb = obj["emb"].astype(np.float32)
    df_index = obj["df_index"].astype(np.int64)

    conf = obj["conf"].astype(np.float32) if "conf" in obj.files else np.zeros((len(df_index),), np.float32)
    best_known_sim = obj["best_known_sim"].astype(np.float32) if "best_known_sim" in obj.files else None
    pred_label = obj["pred_label"] if "pred_label" in obj.files else None
    if pred_label is not None:
        pred_label = np.array([_norm_str(x) for x in pred_label], dtype=object)

    # --- base mask
    m = conf >= float(args.min_conf)

    # pred_label filter
    if args.pred_label.strip():
        if pred_label is None:
            raise RuntimeError("pred_label 필터를 쓰려면 input npz에 pred_label이 있어야 함.")
        want = {s.strip() for s in args.pred_label.split(",") if s.strip()}
        m &= np.isin(pred_label, list(want))

    # novelty filter by percentile (recommended)
    thr = None
    if best_known_sim is not None:
        sims = best_known_sim[m]
        if sims.size == 0:
            raise RuntimeError("필터 적용 후 남은 샘플이 0개야. min-conf/pred-label을 완화해봐.")

        if args.novel_pct is not None:
            q = float(args.novel_pct) / 100.0
            q = min(max(q, 0.0), 1.0)
            thr = float(np.quantile(sims, q))
            m &= best_known_sim <= thr

        # absolute cutoff (optional)
        if args.max_known_sim is not None:
            m &= best_known_sim <= float(args.max_known_sim)

    emb_f = emb[m]
    dfi_f = df_index[m]
    conf_f = conf[m]
    kn_f = best_known_sim[m] if best_known_sim is not None else None
    pl_f = pred_label[m] if pred_label is not None else None

    print("[load] total:", len(df_index), "filtered:", len(dfi_f))
    if thr is not None:
        print("[novel] best_known_sim quantile threshold:", thr)
    if kn_f is not None and len(kn_f) > 0:
        q = np.quantile(kn_f, [0.0, 0.1, 0.5, 0.9, 1.0]).tolist()
        print("[novel] filtered best_known_sim quantiles:", [float(x) for x in q])

    if pl_f is not None:
        from collections import Counter
        print("[pred_label] top5:", Counter(pl_f.tolist()).most_common(5))

    if len(dfi_f) < max(10, int(args.k)):
        raise RuntimeError(f"filtered sample too small: {len(dfi_f)} (k={args.k}). 필터를 완화해봐.")

    # --- KMeans on normalized embeddings (cosine-ish)
    emb_n = l2norm(emb_f)
    km = KMeans(n_clusters=int(args.k), random_state=int(args.seed), n_init=10)
    cid = km.fit_predict(emb_n).astype(np.int32)

    reps = pick_representatives(emb_n, dfi_f, cid)

    np.savez_compressed(out_npz, df_index=dfi_f.astype(np.int64), cluster_id=cid.astype(np.int32))
    out_reps.write_text(json.dumps(reps, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_npz": str(in_npz),
        "method": "kmeans",
        "k": int(args.k),
        "seed": int(args.seed),
        "filters": {
            "min_conf": float(args.min_conf),
            "novel_pct": float(args.novel_pct) if args.novel_pct is not None else None,
            "max_known_sim": float(args.max_known_sim) if args.max_known_sim is not None else None,
            "pred_label": args.pred_label.strip() or None,
        },
        "counts": {
            "total": int(len(df_index)),
            "filtered": int(len(dfi_f)),
            "clusters": int(len(np.unique(cid))),
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[saved] npz:", out_npz)
    print("[saved] reps:", out_reps)
    print("[saved] meta:", out_meta)
    print("[cluster] clusters:", int(len(np.unique(cid))))


if __name__ == "__main__":
    main()
