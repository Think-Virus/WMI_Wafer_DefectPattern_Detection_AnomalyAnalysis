# scripts/run_case.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # wafermap-triage-studio/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from wmi_triage.config import Paths
from wmi_triage.preprocess import PreprocessConfig, preprocess_wafer
from wmi_triage.inference import load_model, predict
from wmi_triage.retrieval import (
    load_known_db,
    load_unknown_db,
    retrieve_known_topk,
    retrieve_unknown_topk,
)


def failuretype_to_label(ft):
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


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df-index", type=int, default=None, help="원본 df index. 없으면 unknown DB에서 랜덤 선택")
    ap.add_argument("--k-known", type=int, default=5)
    ap.add_argument("--k-unk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--cluster-topn", type=int, default=5)
    args = ap.parse_args()

    P = Paths()
    cand = [
        # P.root / "data" / "LSWMD.pkl",
        P.root / "data" / "LSWMD_small.pkl",
        P.root / "data" / "LSWMD_small.pkl.gz",
    ]
    lswmd_pkl = str(next(p for p in cand if p.exists()))
    ckpt = P.models / "20260111_074510_resnet18_R64_polar4_K6_best_e05_weights.pt"
    class_to_idx_path = P.models / "class_to_idx.json"

    known_npz = P.emb_db / "known_embeddings.npz"
    unk_npz = P.emb_db / "unknown_embeddings.npz"
    cluster_npz = P.emb_db / "unknown_cluster.npz"
    cluster_reps_json = P.emb_db / "unknown_cluster_reps.json"
    cluster_meta_json = P.emb_db / "unknown_cluster_meta.json"

    rng = np.random.RandomState(args.seed)

    # device / model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device)

    class_to_idx = json.loads(class_to_idx_path.read_text(encoding="utf-8"))
    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}

    model = load_model(str(ckpt), num_classes=len(class_to_idx), device=device)

    # DB load
    known_db = load_known_db(str(known_npz))
    unk_db = load_unknown_db(str(unk_npz))

    # pick df_index
    if args.df_index is None:
        df_index = int(rng.choice(unk_db.df_index))
        print("[pick] random unknown df_index:", df_index)
    else:
        df_index = int(args.df_index)
        print("[pick] df_index:", df_index)

    # load wafer
    df = pd.read_pickle(lswmd_pkl)
    if df_index not in df.index:
        raise KeyError(f"df_index {df_index} not found in df.index")

    row = df.loc[df_index]
    wafer = np.array(row["waferMap"])
    true_label = failuretype_to_label(row.get("failureType", None))

    # preprocess + inference (polar4/R64)
    cfg = PreprocessConfig(resize=64, input_mode="polar4", upsample_mode="nearest", device="cpu")
    x = preprocess_wafer(wafer, cfg)  # (1,4,64,64) on cpu

    out = predict(model, x, device=device)
    prob = out["prob"][0].numpy()
    q_emb = out["emb"][0].numpy()

    # model top3
    topi = np.argsort(prob)[::-1][:3]
    topv = prob[topi]
    model_top3 = [
        {"rank": r + 1, "class": idx_to_class.get(int(i), f"IDX_{i}"), "idx": int(i), "prob": float(v)}
        for r, (i, v) in enumerate(zip(topi.tolist(), topv.tolist()))
    ]

    # retrieval
    known_topk = retrieve_known_topk(q_emb, known_db, idx_to_class, k=args.k_known)
    unknown_topk = retrieve_unknown_topk(q_emb, unk_db, k=args.k_unk, exclude_df_index=df_index)

    # ----------------------------
    # cluster lookup (unknown + unlabeled)
    # ----------------------------
    unk_cluster_npz = P.emb_db / "unknown_cluster.npz"
    unk_cluster_reps_json = P.emb_db / "unknown_cluster_reps.json"
    unk_cluster_meta_json = P.emb_db / "unknown_cluster_meta.json"

    unl_cluster_npz = P.emb_db / "unlabeled_cluster.npz"
    unl_cluster_reps_json = P.emb_db / "unlabeled_cluster_reps.json"
    unl_cluster_meta_json = P.emb_db / "unlabeled_cluster_meta.json"
    unl_emb_npz = P.emb_db / "unlabeled_embeddings.npz"

    def _load_cluster_mapping(cluster_npz: Path):
        if not cluster_npz.exists():
            return None, None, None
        cobj = np.load(cluster_npz, allow_pickle=True)
        c_df_index = cobj["df_index"].astype(np.int64)
        c_cid = cobj["cluster_id"].astype(np.int32)
        method = str(cobj["method"][0]) if "method" in cobj.files else "unknown"
        m = {int(dfi): int(cid) for dfi, cid in zip(c_df_index.tolist(), c_cid.tolist())}
        return m, method, cobj

    def _load_json(p: Path) -> dict:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    def _cluster_info_for_df_index(df_index: int, cluster_npz: Path, reps_json: Path, meta_json: Path, dataset: str):
        m, method, _ = _load_cluster_mapping(cluster_npz)
        if m is None:
            return {"available": False, "dataset": dataset}

        cid = m.get(int(df_index), None)
        reps = _load_json(reps_json)
        meta = _load_json(meta_json)

        rep_df_index = None
        if cid is not None:
            rep_df_index = reps.get(str(cid), reps.get(cid, None))

        return {
            "available": True,
            "dataset": dataset,  # "unknown" or "unlabeled"
            "method": method,
            "cluster_id": int(cid) if cid is not None else None,
            "rep_df_index": int(rep_df_index) if rep_df_index is not None else None,
            "meta": meta,
            "cluster_npz": str(cluster_npz),
        }, m

    # (A) unknown_topk에 unknown cluster_id 붙이기 + query의 unknown cluster 찾기
    cluster_info_unknown, unk_map = _cluster_info_for_df_index(
        df_index, unk_cluster_npz, unk_cluster_reps_json, unk_cluster_meta_json, dataset="unknown"
    )
    if isinstance(unk_map, dict):
        for it in unknown_topk:
            dfi = int(it["df_index"])
            it["cluster_id"] = unk_map.get(dfi, None)

    # (B) query의 unlabeled cluster 찾기
    cluster_info_unlabeled, unl_map = _cluster_info_for_df_index(
        df_index, unl_cluster_npz, unl_cluster_reps_json, unl_cluster_meta_json, dataset="unlabeled"
    )

    # (추천) unlabeled_cluster에 df_index가 직접 없을 때도 “가장 가까운 unlabeled cluster rep”로 할당
    if (
            cluster_info_unlabeled.get("available", False)
            and cluster_info_unlabeled.get("cluster_id") is None
            and unl_emb_npz.exists()
            and unl_cluster_reps_json.exists()
    ):
        u = np.load(unl_emb_npz, allow_pickle=True)
        u_dfi = u["df_index"].astype(np.int64)
        u_emb = u["emb"].astype(np.float32)

        pos_map = {int(d): i for i, d in enumerate(u_dfi.tolist())}
        qpos = pos_map.get(int(df_index), None)
        if qpos is not None:
            qv = u_emb[qpos]
            qv = qv / (np.linalg.norm(qv) + 1e-12)

            reps = _load_json(unl_cluster_reps_json)  # {"0": 281674, "1": ...}
            rep_cids = []
            rep_dfis = []
            rep_vecs = []
            for k, rep_dfi in reps.items():
                cid = int(k)
                rep_dfi = int(rep_dfi)
                rpos = pos_map.get(rep_dfi, None)
                if rpos is None:
                    continue
                rv = u_emb[rpos]
                rv = rv / (np.linalg.norm(rv) + 1e-12)
                rep_cids.append(cid)
                rep_dfis.append(rep_dfi)
                rep_vecs.append(rv)

            if rep_vecs:
                C = np.stack(rep_vecs, axis=0)
                sim = (C @ qv).astype(np.float32)
                j = int(np.argmax(sim))
                cluster_info_unlabeled["cluster_id"] = int(rep_cids[j])
                cluster_info_unlabeled["rep_df_index"] = int(rep_dfis[j])
                cluster_info_unlabeled["assign_mode"] = "nearest_rep"
                cluster_info_unlabeled["assign_cosine_sim"] = float(sim[j])

    # (C) summary에 넣을 "cluster"는: unknown에 속하면 unknown, 아니면 unlabeled 우선
    #     (둘 다 없으면 cluster_id=None)
    if cluster_info_unknown.get("cluster_id") is not None:
        cluster_info = cluster_info_unknown
    elif cluster_info_unlabeled.get("cluster_id") is not None:
        cluster_info = cluster_info_unlabeled
    else:
        # 둘 다 못 찾은 경우도 available=True로 두면 UI에서 '없음' 표시 가능
        cluster_info = {
            "available": True,
            "dataset": None,
            "method": None,
            "cluster_id": None,
            "rep_df_index": None,
            "cluster_npz": None,
        }

    # save summary
    case_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_df{df_index}"
    out_dir = P.cases / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- unified cluster ranking (known classes + unknown clusters) ---
    bank_npz = P.emb_db / "cluster_bank.npz"
    cluster_rank_top = []

    if bank_npz.exists():
        b = np.load(bank_npz, allow_pickle=True)
        C = b["centroid"].astype(np.float32)  # (M,512), already normalized
        names = [str(x) for x in b["name"].tolist()]
        types = [str(x) for x in b["type"].tolist()]
        reps = b["rep_df_index"].astype(np.int64)
        cnts = b["count"].astype(np.int32)

        qn = q_emb.astype(np.float32)
        qn = qn / (np.linalg.norm(qn) + 1e-12)

        sim = (C @ qn).astype(np.float32)  # (M,)
        prob = softmax(sim / float(args.tau))

        order = np.argsort(prob)[::-1][: int(args.cluster_topn)]
        for r, j in enumerate(order.tolist()):
            cluster_rank_top.append({
                "rank": r + 1,
                "type": types[j],
                "name": names[j],
                "prob": float(prob[j]),
                "cosine_sim": float(sim[j]),
                "rep_df_index": int(reps[j]),
                "count": int(cnts[j]),
            })

    summary = {
        "case_id": case_id,
        "query": {"df_index": int(df_index), "true_label": true_label, "model_top3": model_top3},
        "known_topk": known_topk,
        "unknown_topk": unknown_topk,
        "cluster": cluster_info,
        "cluster_rank_top5": cluster_rank_top,
        "cluster_rank_tau": float(args.tau),
        "paths": {
            "lswmd_pkl": str(lswmd_pkl),
            "ckpt": str(ckpt),
            "known_db": str(known_npz),
            "unknown_db": str(unk_npz),
            "cluster_npz": str(cluster_npz),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[saved]", out_dir / "summary.json")


if __name__ == "__main__":
    main()
