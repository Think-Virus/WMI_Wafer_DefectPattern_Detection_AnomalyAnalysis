# app/streamlit_app.py
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from wmi_triage.config import Paths


# -----------------------------
# Utils
# -----------------------------
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


def wafer_to_pil(wafer: np.ndarray, scale: int = 8) -> Image.Image:
    """
    wafer values: 보통 0/1/2
    - 0: background
    - 1: wafer
    - 2: defect
    """
    w = np.array(wafer)
    if w.ndim != 2:
        w = w.squeeze()
    w = w.astype(np.uint8, copy=False)

    # simple grayscale mapping
    img = np.zeros_like(w, dtype=np.uint8)
    img[w == 0] = 0
    img[w == 1] = 110
    img[w == 2] = 255

    pil = Image.fromarray(img, mode="L")
    pil = pil.resize((pil.size[0] * scale, pil.size[1] * scale), resample=Image.NEAREST)
    return pil


def safe_read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def list_case_summaries(cases_dir: Path) -> List[Path]:
    if not cases_dir.exists():
        return []
    paths = sorted(cases_dir.glob("*/summary.json"), key=lambda x: x.parent.name, reverse=True)
    return paths


def majority_label(items: List[dict], key: str) -> Tuple[Optional[str], Dict[str, int]]:
    if not items:
        return None, {}
    vals = [str(it.get(key)) for it in items if it.get(key) is not None]
    if not vals:
        return None, {}
    from collections import Counter
    c = Counter(vals)
    top = c.most_common(1)[0][0]
    return top, dict(c)


# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_resource
def load_df_cached(pkl_path: str) -> pd.DataFrame:
    return pd.read_pickle(pkl_path)


@st.cache_resource
def load_unknown_cluster_cached(npz_path: str) -> Optional[dict]:
    p = Path(npz_path)
    if not p.exists():
        return None
    obj = np.load(p, allow_pickle=True)
    out = {k: obj[k] for k in obj.files}
    return out


# -----------------------------
# UI Components
# -----------------------------
def render_wafer_card(df: pd.DataFrame, df_index: int, title: str, subtitle: str = "", scale: int = 8):
    if df_index not in df.index:
        st.warning(f"{title}: df_index {df_index} not found in df")
        return

    row = df.loc[df_index]
    wafer = row["waferMap"]
    wafer = np.array(wafer)
    true_label = failuretype_to_label(row.get("failureType", None))

    st.markdown(f"### {title}")
    st.caption(subtitle)
    st.write(f"- df_index: `{df_index}`")
    if true_label is not None:
        st.write(f"- true_label: `{true_label}`")
    st.write(f"- shape: `{tuple(wafer.shape)}`")

    st.image(wafer_to_pil(wafer, scale=scale), use_container_width=True)


def render_topk_thumbs(df: pd.DataFrame, items: List[dict], title: str, cols: int = 5, scale: int = 6):
    st.markdown(f"### {title}")

    if not items:
        st.info("No items")
        return

    # table
    st.dataframe(pd.DataFrame(items), use_container_width=True)

    # thumbs
    n = len(items)
    grid = st.columns(cols)
    for i, it in enumerate(items):
        c = grid[i % cols]
        dfi = int(it["df_index"])
        sim = float(it.get("cosine_sim", 0.0))
        extra = []
        if "label" in it:
            extra.append(str(it["label"]))
        if "true_label" in it:
            extra.append(f"true={it['true_label']}")
        caption = f"#{it.get('rank', i + 1)} df={dfi} sim={sim:.3f} " + (" | ".join(extra) if extra else "")

        with c:
            if dfi in df.index:
                wafer = np.array(df.loc[dfi]["waferMap"])
                st.image(wafer_to_pil(wafer, scale=scale), caption=caption, use_container_width=True)
            else:
                st.write(caption)
                st.warning("df_index not found")


def run_case_subprocess(project_root: Path, df_index: Optional[int], k_known: int, k_unk: int) -> Tuple[int, str]:
    """
    streamlit 안에서 새 케이스 생성(선택 기능).
    """
    cmd = [sys.executable, str(project_root / "scripts" / "run_case.py"),
           "--k-known", str(k_known), "--k-unk", str(k_unk)]
    if df_index is not None:
        cmd += ["--df-index", str(df_index)]

    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        shell=False
    )
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, out


# -----------------------------
# Main
# -----------------------------
def main():
    st.set_page_config(page_title="Wafermap Triage Studio", layout="wide")
    P = Paths()

    st.title("Wafermap Triage Studio")
    st.caption("case summary.json 기반 뷰어 (Query / Known / Unknown / Cluster)")

    # Sidebar
    st.sidebar.header("Controls")

    mode = st.sidebar.radio(
        "Mode",
        ["Browse saved cases", "Open summary.json", "Run new case (optional)"],
        index=0
    )

    # Load DF once (big)
    lswmd_pkl = str(P.root / "data" / "LSWMD.pkl")
    with st.sidebar.expander("Dataset", expanded=False):
        st.write(f"LSWMD.pkl: `{lswmd_pkl}`")
        load_df = st.button("Load dataset now (cache)")
    if load_df or True:
        # 캐시로 한번만 로드되게
        df = load_df_cached(lswmd_pkl)

    summary: Optional[Dict[str, Any]] = None
    summary_path: Optional[Path] = None

    if mode == "Browse saved cases":
        cases = list_case_summaries(P.cases)
        if not cases:
            st.warning(f"No cases found in: {P.cases}")
            st.stop()

        labels = [p.parent.name for p in cases]
        pick = st.sidebar.selectbox("Select case", labels, index=0)
        summary_path = next(p for p in cases if p.parent.name == pick)
        summary = safe_read_json(summary_path)

    elif mode == "Open summary.json":
        up = st.sidebar.file_uploader("Upload summary.json", type=["json"])
        if up is None:
            st.info("Upload a summary.json to view.")
            st.stop()
        summary = json.loads(up.read().decode("utf-8"))

    else:
        st.sidebar.write("Generate a new case by calling scripts/run_case.py")
        df_index = st.sidebar.text_input("df_index (optional)", value="")
        k_known = st.sidebar.number_input("k_known", min_value=1, max_value=50, value=5, step=1)
        k_unk = st.sidebar.number_input("k_unk", min_value=1, max_value=50, value=5, step=1)
        if st.sidebar.button("Run"):
            dfi = int(df_index) if df_index.strip() else None
            code, log = run_case_subprocess(P.root, dfi, int(k_known), int(k_unk))
            st.sidebar.code(log)
            if code != 0:
                st.error("run_case failed. See log in sidebar.")
                st.stop()

            # 최신 case 자동 로드
            cases = list_case_summaries(P.cases)
            if not cases:
                st.error("Case created but cannot find summary.json under artifacts/cases")
                st.stop()
            summary_path = cases[0]
            summary = safe_read_json(summary_path)

    if summary is None:
        st.stop()

    # Header info
    case_id = summary.get("case_id", "unknown_case")
    q = summary.get("query", {})
    q_df_index = int(q.get("df_index", -1))
    q_true = q.get("true_label", None)
    model_top3 = q.get("model_top3", [])

    st.subheader(f"Case: {case_id}")
    cols = st.columns(3)
    with cols[0]:
        st.write(f"**Query df_index**: `{q_df_index}`")
        st.write(f"**Query true_label**: `{q_true}`")
    with cols[1]:
        st.write("**Model Top-3 (known classes)**")
        if model_top3:
            st.dataframe(pd.DataFrame(model_top3), use_container_width=True)
        else:
            st.info("No model_top3 in summary")
    with cols[2]:
        if summary_path is not None:
            st.write("**summary.json**")
            st.code(str(summary_path))

    # Triage hint
    known_topk = summary.get("known_topk", [])
    unk_topk = summary.get("unknown_topk", [])
    known_major, known_dist = majority_label(known_topk, "label")
    unk_major, unk_dist = majority_label(unk_topk, "true_label")

    st.markdown("## Triage summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("Known Top-K majority")
        st.write(f"- majority: `{known_major}`")
        st.json(known_dist)
    with c2:
        st.write("Unknown Top-K majority")
        st.write(f"- majority: `{unk_major}`")
        st.json(unk_dist)
    with c3:
        # quick comparison of best similarities
        best_known = float(known_topk[0]["cosine_sim"]) if known_topk else None
        best_unk = float(unk_topk[0]["cosine_sim"]) if unk_topk else None
        st.write("Similarity (top1)")
        st.write(f"- best known sim: `{best_known}`")
        st.write(f"- best unknown sim: `{best_unk}`")
        if best_known is not None and best_unk is not None:
            if best_unk > best_known:
                st.success("Unknown side is closer than Known side → likely novel/unknown pattern candidate")
            else:
                st.info("Known side is closer or similar → could be near-known / ambiguous")

    st.divider()

    # Main panels
    left, mid, right = st.columns([1.1, 1.2, 1.1], gap="large")

    with left:
        render_wafer_card(
            df=df,
            df_index=q_df_index,
            title="Query wafer",
            subtitle="원본 waferMap을 그대로 렌더링",
            scale=10,
        )

    with mid:
        render_topk_thumbs(df, known_topk, title="Known Top-K (reference cases)", cols=5, scale=6)
        st.divider()
        render_topk_thumbs(df, unk_topk, title="Unknown Top-K (similar unknown cases)", cols=5, scale=6)

    with right:
        st.markdown("## Cluster (new pattern candidates)")
        cluster = summary.get("cluster", {})
        if not cluster or not cluster.get("available", False):
            st.info("Cluster info not available")
        else:
            st.write(f"- method: `{cluster.get('method')}`")
            st.write(f"- cluster_id: `{cluster.get('cluster_id')}`")
            st.write(f"- rep_df_index: `{cluster.get('rep_df_index')}`")

            rep_df = cluster.get("rep_df_index", None)
            if rep_df is not None:
                render_wafer_card(
                    df=df,
                    df_index=int(rep_df),
                    title="Cluster representative",
                    subtitle="cluster centroid에 가장 가까운 대표 샘플",
                    scale=10,
                )

            # show cluster members (optional)
            cid = cluster.get("cluster_id", None)
            cnpz = load_unknown_cluster_cached(str(P.emb_db / "unknown_cluster.npz"))
            if cnpz is not None and cid is not None:
                all_df_index = cnpz["df_index"].astype(np.int64)
                all_cid = cnpz["cluster_id"].astype(np.int32)
                members = all_df_index[all_cid == int(cid)]
                st.write(f"**members count**: `{len(members)}`")

                n_show = st.slider("Show members (sample)", 0, 30, 10)
                if n_show > 0 and len(members) > 0:
                    # deterministic sample
                    sel = members[: min(n_show, len(members))]
                    cols_m = st.columns(5)
                    for i, dfi in enumerate(sel.tolist()):
                        with cols_m[i % 5]:
                            if int(dfi) in df.index:
                                wafer = np.array(df.loc[int(dfi)]["waferMap"])
                                st.image(
                                    wafer_to_pil(wafer, scale=6),
                                    caption=f"df={int(dfi)}",
                                    use_container_width=True,
                                )

        with st.expander("Raw summary.json", expanded=False):
            st.json(summary)


if __name__ == "__main__":
    main()
