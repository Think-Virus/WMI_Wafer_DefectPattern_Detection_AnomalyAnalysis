# scripts/smoke_test_inference.py
import json
import numpy as np
import pandas as pd
import torch

from wmi_triage.config import Paths
from wmi_triage.preprocess import PreprocessConfig, preprocess_wafer
from wmi_triage.inference import load_model, predict


def main():
    P = Paths()

    LSWMD_PKL = P.root / "data" / "LSWMD.pkl"
    CKPT_PATH = P.models / "20260111_074510_resnet18_R64_polar4_K6_best_e05.pt"
    CLASS_TO_IDX_PATH = P.models / "class_to_idx.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device)
    if device == "cuda":
        print("[cuda]", torch.cuda.get_device_name(0))

    print("Loading dataset...")
    df = pd.read_pickle(LSWMD_PKL)
    wafer = np.array(df["waferMap"].iloc[0])
    print("[sample] wafer shape:", wafer.shape, "dtype:", wafer.dtype)

    with open(CLASS_TO_IDX_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print("[meta] num_classes:", num_classes)

    model = load_model(str(CKPT_PATH), num_classes=num_classes, device=device)

    cfg = PreprocessConfig(
        resize=64,
        input_mode="polar4",
        upsample_mode="nearest",
        device=device,
    )
    x = preprocess_wafer(wafer, cfg)
    print("[preprocess] x shape:", tuple(x.shape), "dtype:", x.dtype, "device:", x.device)

    out = predict(model, x, device=device)
    logits, prob, emb = out["logits"], out["prob"], out["emb"]
    pred, conf = out["pred"].item(), out["conf"].item()

    print("[infer] logits:", tuple(logits.shape))
    print("[infer] prob:", tuple(prob.shape))
    print("[infer] emb:", tuple(emb.shape))
    print("[infer] pred idx:", pred, "pred class:", idx_to_class.get(pred, "UNKNOWN_IDX"), "conf:", conf)

    topv, topi = torch.topk(prob[0], k=3)
    print("[infer] Top-3:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print("   -", idx_to_class.get(i, "UNKNOWN_IDX"), f"(idx={i})", f"prob={v:.4f}")


if __name__ == "__main__":
    main()
