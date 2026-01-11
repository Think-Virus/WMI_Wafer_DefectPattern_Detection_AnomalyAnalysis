# scripts/build_unknown_db.py
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from wmi_triage.config import Paths
from wmi_triage.preprocess import PreprocessConfig, preprocess_wafer
from wmi_triage.inference import load_model

RANDOM_STATE = 42


def failuretype_to_label(ft) -> Optional[str]:
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
        if a0 is None:
            return None
        s = str(a0).strip()
        return s if s != "" else None

    if isinstance(ft, str):
        s = ft.strip()
        return s if s != "" else None

    s = str(ft).strip()
    return s if s != "" else None


class UnknownDataset(Dataset):
    """
    unknown_df에서 임베딩 추출용 Dataset
    - 반환: (x(C,R,R), df_index(int), true_label(str))
    """

    def __init__(self, unknown_df: pd.DataFrame, cfg_cpu: PreprocessConfig):
        self.df = unknown_df.reset_index()  # 원래 df index를 컬럼으로 보관
        self.cfg_cpu = cfg_cpu

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        wafer = np.array(row["waferMap"])
        x = preprocess_wafer(wafer, self.cfg_cpu).squeeze(0)  # (C,R,R)
        df_index = int(row["index"])  # 원본 df index
        true_label = str(row["label"])  # Scratch 등
        return x, df_index, true_label


@torch.no_grad()
def extract_embeddings(model, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    embs, idxs, labels = [], [], []
    for x, df_index, true_label in loader:
        x = x.to(device, non_blocking=True)
        logits, emb = model(x)  # emb: (B,512)
        embs.append(emb.detach().cpu().numpy())
        idxs.append(df_index.numpy())
        labels.extend(list(true_label))  # batch의 문자열들

    emb = np.concatenate(embs, axis=0).astype(np.float32)
    df_index = np.concatenate(idxs, axis=0).astype(np.int64)
    true_label = np.array(labels, dtype="<U32")
    return emb, df_index, true_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="(optional) ckpt filename under artifacts/models/")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0, help="Windows는 0 권장")
    ap.add_argument("--exclude", nargs="*", default=["none"], help="label에서 제외할 값들 (예: none)")
    args = ap.parse_args()

    P = Paths()
    lswmd_pkl = P.root / "data" / "LSWMD.pkl"
    class_to_idx_path = P.models / "class_to_idx.json"

    if args.ckpt is None:
        ckpt_path = P.models / "20260111_074510_resnet18_R64_polar4_K6_best_e05.pt"
    else:
        ckpt_path = P.models / args.ckpt

    P.emb_db.mkdir(parents=True, exist_ok=True)

    # 1) 데이터 로드 + label 생성
    df = pd.read_pickle(lswmd_pkl).copy()
    df["label"] = df["failureType"].apply(failuretype_to_label)

    # 2) known set = class_to_idx.json 기준 (노트북 원칙)
    class_to_idx = json.loads(class_to_idx_path.read_text(encoding="utf-8"))
    known_classes = set(class_to_idx.keys())

    # labeled만
    labeled_df = df[df["label"].notna()].copy()
    labeled_df = labeled_df[~labeled_df["label"].isin(args.exclude)].copy()

    # unknown = labeled인데 known_classes에 없는 것 (노트북 unknown_df 대응)
    unknown_df = labeled_df[~labeled_df["label"].isin(list(known_classes))].copy()

    if len(unknown_df) == 0:
        raise RuntimeError("unknown_df가 비었어. label 생성/known_classes가 맞는지 확인해줘.")

    print("[data] total:", len(df), "labeled:", len(labeled_df), "unknown:", len(unknown_df))
    unk_dist = Counter(unknown_df["label"].tolist())
    # 너무 길면 상위만
    top10 = dict(sorted(unk_dist.items(), key=lambda kv: kv[1], reverse=True)[:10])
    print("[data] unknown label top10:", top10)

    # 3) 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device)
    model = load_model(str(ckpt_path), num_classes=len(class_to_idx), device=device)

    # 4) 전처리/임베딩 추출
    cfg_cpu = PreprocessConfig(
        resize=64,
        input_mode="polar4",
        upsample_mode="nearest",
        device="cpu",
    )

    ds = UnknownDataset(unknown_df, cfg_cpu=cfg_cpu)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    emb, df_index, true_label = extract_embeddings(model, dl, device=device)

    out_npz = P.emb_db / "unknown_embeddings.npz"
    np.savez_compressed(out_npz, emb=emb, df_index=df_index, true_label=true_label)

    meta = {
        "ckpt": ckpt_path.name,
        "num_classes": int(len(class_to_idx)),
        "known_classes": sorted(list(known_classes)),
        "preprocess": asdict(cfg_cpu),
        "exclude": args.exclude,
        "unknown_df_size": int(len(unknown_df)),
        "saved_npz": str(out_npz),
        "fields": {
            "emb": "float32 (N,512)",
            "df_index": "int64 (N,) original df index",
            "true_label": "str (N,) from failureType-derived label",
        },
        "unknown_label_dist_top10": top10,
    }
    (P.emb_db / "unknown_db_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ saved:", out_npz)
    print("✅ meta :", P.emb_db / "unknown_db_meta.json")
    print("[npz] emb:", emb.shape, "df_index:", df_index.shape, "true_label:", true_label.shape)


if __name__ == "__main__":
    main()
