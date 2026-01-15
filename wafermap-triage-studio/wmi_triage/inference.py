# wmi_triage/inference.py
import json
import torch
import torch.nn.functional as F
from .model import ResNet18TVEmbed


def load_class_to_idx(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj:
            return ckpt_obj["state_dict"]
        if "model_state_dict" in ckpt_obj:
            return ckpt_obj["model_state_dict"]
    return ckpt_obj


def _clean_state_dict_keys(sd: dict) -> dict:
    cleaned = {}
    for k, v in sd.items():
        for p in ("model.", "module.", "net.", "resnet."):
            if k.startswith(p):
                k = k[len(p):]
        cleaned[k] = v
    return cleaned


def load_model(ckpt_path: str, num_classes: int, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = _clean_state_dict_keys(_extract_state_dict(ckpt))

    if "conv1.weight" not in sd:
        raise RuntimeError(f"conv1.weight가 state_dict에 없음. keys sample={list(sd.keys())[:20]}")

    in_chans = int(sd["conv1.weight"].shape[1])

    model = ResNet18TVEmbed(num_classes=num_classes, in_chans=in_chans).to(device)

    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        # 디버깅에 도움되는 정보를 같이 출력
        fc_w = sd.get("fc.weight", None)
        fc_out = None if fc_w is None else int(fc_w.shape[0])
        raise RuntimeError(
            f"state_dict 로드 실패. ckpt fc_out={fc_out}, expected num_classes={num_classes}. "
            f"in_chans={in_chans}. 원본 에러: {e}"
        )

    model.eval()
    return model


@torch.no_grad()
def predict(model, x, device="cpu"):
    x = x.to(device)
    logits, emb = model(x)
    prob = F.softmax(logits, dim=1)
    conf, pred = prob.max(dim=1)
    return {
        "logits": logits.detach().cpu(),
        "prob": prob.detach().cpu(),
        "pred": pred.detach().cpu(),
        "conf": conf.detach().cpu(),
        "emb": emb.detach().cpu(),
    }
