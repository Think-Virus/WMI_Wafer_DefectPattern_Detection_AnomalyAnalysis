# scripts/build_known_db.py
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
from sklearn.model_selection import train_test_split

from wmi_triage.config import Paths
from wmi_triage.preprocess import PreprocessConfig, preprocess_wafer
from wmi_triage.inference import load_model

RANDOM_STATE = 42


def failuretype_to_label(ft) -> Optional[str]:
    """
    노트북에서 failureType이 list 형태/빈 리스트/문자열 등 섞여있던 걸 안전하게 label로 변환.
    - [] -> None
    - ["Center"] -> "Center"
    - [["Center"]] 같은 중첩도 첫 원소를 따라가서 문자열로
    """
    if ft is None:
        return None

    # numpy array도 list처럼 처리
    if isinstance(ft, np.ndarray):
        ft = ft.tolist()

    if isinstance(ft, (list, tuple)):
        if len(ft) == 0:
            return None
        a0 = ft[0]
        # 중첩 list 형태도 처리
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

    # 기타 타입은 문자열화
    s = str(ft).strip()
    return s if s != "" else None


class KnownRefDataset(Dataset):
    """
    known ref_df에서 임베딩 추출용 Dataset
    - 반환: (x(C,R,R), y(int), df_index(int))
    """

    def __init__(self, ref_df: pd.DataFrame, class_to_idx: dict, cfg_cpu: PreprocessConfig):
        self.df = ref_df.reset_index()  # 원래 df index를 컬럼으로 보관
        self.class_to_idx = class_to_idx
        self.cfg_cpu = cfg_cpu

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        wafer = np.array(row["waferMap"])
        x = preprocess_wafer(wafer, self.cfg_cpu).squeeze(0)  # (C,R,R)
        y = int(self.class_to_idx[str(row["label"])])
        df_index = int(row["index"])  # 원본 df의 index
        return x, y, df_index


@torch.no_grad()
def extract_embeddings(model, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    embs, ys, idxs = [], [], []
    for x, y, df_index in loader:
        x = x.to(device, non_blocking=True)
        logits, emb = model(x)  # emb: (B,512)
        embs.append(emb.detach().cpu().numpy())
        ys.append(y.numpy())
        idxs.append(df_index.numpy())

    emb = np.concatenate(embs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    df_index = np.concatenate(idxs, axis=0).astype(np.int64)
    return emb, y, df_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="(optional) ckpt filename under artifacts/models/")
    ap.add_argument("--max-per-class", type=int, default=1500, help="ref_df에서 클래스별 최대 샘플 수")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0, help="Windows는 0 권장")
    ap.add_argument("--exclude", nargs="*", default=["none"], help="label에서 제외할 값들 (예: none)")
    args = ap.parse_args()

    P = Paths()
    lswmd_pkl = P.root / "data" / "LSWMD.pkl"
    class_to_idx_path = P.models / "class_to_idx.json"

    if args.ckpt is None:
        # 기본: 가장 최근에 쓰던 ckpt 파일명(네가 이미 smoke_test에서 쓰는 것)
        ckpt_path = P.models / "20260111_074510_resnet18_R64_polar4_K6_best_e05.pt"
    else:
        ckpt_path = P.models / args.ckpt

    P.emb_db.mkdir(parents=True, exist_ok=True)

    # 1) 데이터 로드 + label 생성
    df = pd.read_pickle(lswmd_pkl).copy()
    df["label"] = df["failureType"].apply(failuretype_to_label)

    # 2) known set = class_to_idx.json 기준 (노트북 원칙)
    class_to_idx = json.loads(class_to_idx_path.read_text(encoding="utf-8"))
    known_classes = list(class_to_idx.keys())

    # labeled만
    labeled_df = df[df["label"].notna()].copy()
    labeled_df = labeled_df[~labeled_df["label"].isin(args.exclude)].copy()

    # known만
    known_df = labeled_df[labeled_df["label"].isin(known_classes)].copy()

    if len(known_df) == 0:
        raise RuntimeError("known_df가 비었어. label 생성/known_classes가 맞는지 확인해줘.")

    print("[data] total:", len(df), "labeled:", len(labeled_df), "known:", len(known_df))
    print("[data] known label dist:", dict(Counter(known_df["label"].tolist())))

    # 3) 노트북처럼 split (train/val/test_known)
    train_df, temp_df = train_test_split(
        known_df,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=known_df["label"],
    )
    val_df, test_known_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
    )

    print("[split] train:", len(train_df), "val:", len(val_df), "test_known:", len(test_known_df))

    # 4) 노트북처럼 ref_df 구성: train_df에서 클래스별 최대 N개 샘플
    ref_parts = []
    for c in known_classes:
        sub = train_df[train_df["label"] == c]
        if len(sub) == 0:
            continue
        n = min(len(sub), args.max_per_class)
        ref_parts.append(sub.sample(n=n, random_state=RANDOM_STATE))

    ref_df = pd.concat(ref_parts, axis=0).sample(frac=1.0, random_state=RANDOM_STATE)  # shuffle
    print("[ref_df] size:", len(ref_df), "per-class:", dict(Counter(ref_df["label"].tolist())))

    # 5) 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device)
    model = load_model(str(ckpt_path), num_classes=len(class_to_idx), device=device)

    # 6) 전처리/임베딩 추출 (노트북 최종: R64 polar4, nearest)
    cfg_cpu = PreprocessConfig(
        resize=64,
        input_mode="polar4",
        upsample_mode="nearest",
        device="cpu",
    )

    ds = KnownRefDataset(ref_df, class_to_idx=class_to_idx, cfg_cpu=cfg_cpu)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    emb, y, df_index = extract_embeddings(model, dl, device=device)

    out_npz = P.emb_db / "known_embeddings.npz"
    np.savez_compressed(out_npz, emb=emb, y=y, df_index=df_index)

    meta = {
        "ckpt": ckpt_path.name,
        "num_classes": int(len(class_to_idx)),
        "class_to_idx": class_to_idx,
        "known_classes": known_classes,
        "preprocess": asdict(cfg_cpu),
        "random_state": RANDOM_STATE,
        "split_sizes": {"train": int(len(train_df)), "val": int(len(val_df)), "test_known": int(len(test_known_df))},
        "ref_df_size": int(len(ref_df)),
        "max_per_class": int(args.max_per_class),
        "exclude": args.exclude,
        "saved_npz": str(out_npz),
        "fields": {"emb": "float32 (N,512)", "y": "int64 (N,)", "df_index": "int64 (N,) original df index"},
    }
    (P.emb_db / "known_db_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[saved]", out_npz)
    print("[meta]", P.emb_db / "known_db_meta.json")
    print("[npz] emb:", emb.shape, "y:", y.shape, "df_index:", df_index.shape)


if __name__ == "__main__":
    main()
