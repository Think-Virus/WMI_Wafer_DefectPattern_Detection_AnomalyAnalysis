# scripts/compact_ckpt.py

import torch

from wmi_triage.config import Paths

P = Paths()
SRC = P.models / "20260111_074510_resnet18_R64_polar4_K6_best_e05.pt"
DST = P.models / "20260111_074510_resnet18_R64_polar4_K6_best_e05_weights.pt"


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        # 이미 state_dict 형태일 수도 있음
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    # model object로 저장된 경우
    if hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()
    raise RuntimeError("Cannot extract state_dict from checkpoint.")


def main():
    ckpt = torch.load(SRC, map_location="cpu")
    sd = extract_state_dict(ckpt)

    # (선택) 불필요 prefix 정리: model./module./resnet. 등
    cleaned = {}
    for k, v in sd.items():
        kk = k
        for p in ("model.", "module.", "net.", "resnet."):
            if kk.startswith(p):
                kk = kk[len(p):]
        cleaned[kk] = v.detach().cpu()

    # weights-only로 저장 (옵션/스케줄러/optimizer 제거)
    torch.save(cleaned, DST)

    mb = DST.stat().st_size / (1024 * 1024)
    print(f"[saved] {DST} ({mb:.2f} MB)")
    print("[peek] keys:", list(cleaned.keys())[:10])


if __name__ == "__main__":
    main()
