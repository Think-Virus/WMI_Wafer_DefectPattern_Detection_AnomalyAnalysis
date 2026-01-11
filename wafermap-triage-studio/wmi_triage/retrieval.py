# wmi_triage/retrieval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def cosine_topk(db_emb_n: np.ndarray, q_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    db_emb_n: (N,D) L2-normalized
    q_emb: (D,) or (1,D)
    return: top_idx (k,), top_sim (k,)
    """
    if q_emb.ndim == 2:
        q = q_emb[0]
    else:
        q = q_emb

    q = q.astype(np.float32, copy=False)
    q = q / (np.linalg.norm(q) + 1e-12)

    sim = db_emb_n @ q  # (N,)
    k = int(min(k, sim.shape[0]))

    top_idx = np.argpartition(-sim, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-sim[top_idx])]
    top_sim = sim[top_idx]
    return top_idx.astype(np.int64), top_sim.astype(np.float32)


@dataclass
class KnownDB:
    emb_n: np.ndarray  # (N,512) normalized
    y: np.ndarray  # (N,)
    df_index: np.ndarray  # (N,)


@dataclass
class UnknownDB:
    emb_n: np.ndarray  # (N,512) normalized
    df_index: np.ndarray  # (N,)
    true_label: Optional[np.ndarray] = None  # (N,)


def load_known_db(npz_path: str) -> KnownDB:
    obj = np.load(npz_path, allow_pickle=True)
    emb = obj["emb"].astype(np.float32)
    y = obj["y"].astype(np.int64)
    df_index = obj["df_index"].astype(np.int64)
    return KnownDB(emb_n=l2norm(emb), y=y, df_index=df_index)


def load_unknown_db(npz_path: str) -> UnknownDB:
    obj = np.load(npz_path, allow_pickle=True)
    emb = obj["emb"].astype(np.float32)
    df_index = obj["df_index"].astype(np.int64)
    true_label = obj["true_label"] if "true_label" in obj.files else None
    return UnknownDB(emb_n=l2norm(emb), df_index=df_index, true_label=true_label)


def retrieve_known_topk(
        q_emb: np.ndarray,
        db: KnownDB,
        idx_to_class: Dict[int, str],
        k: int = 5,
) -> List[dict]:
    top_idx, top_sim = cosine_topk(db.emb_n, q_emb, k=k)
    out: List[dict] = []
    for rank, (i, s) in enumerate(zip(top_idx.tolist(), top_sim.tolist()), start=1):
        yi = int(db.y[i])
        out.append(
            {
                "rank": rank,
                "df_index": int(db.df_index[i]),
                "y": yi,
                "label": idx_to_class.get(yi, f"IDX_{yi}"),
                "cosine_sim": float(s),
            }
        )
    return out


def retrieve_unknown_topk(
        q_emb: np.ndarray,
        db: UnknownDB,
        k: int = 5,
        exclude_df_index: Optional[int] = None,
) -> List[dict]:
    # exclude 때문에 +1 여유로 뽑기
    top_idx, top_sim = cosine_topk(db.emb_n, q_emb, k=min(k + 1, db.emb_n.shape[0]))

    out: List[dict] = []
    for i, s in zip(top_idx.tolist(), top_sim.tolist()):
        dfi = int(db.df_index[i])
        if exclude_df_index is not None and dfi == int(exclude_df_index):
            continue

        item = {"df_index": dfi, "cosine_sim": float(s)}
        if db.true_label is not None:
            item["true_label"] = str(db.true_label[i])
        out.append(item)

        if len(out) >= k:
            break

    for r, item in enumerate(out, start=1):
        item["rank"] = r
    return out
