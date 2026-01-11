# app/streamlit_app.py
from __future__ import annotations

import json
import os
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


@st.cache_resource
def load_unknown_pool_cached(npz_path: str) -> Dict[str, List[int]]:
    p = Path(npz_path)
    if not p.exists():
        return {}

    obj = np.load(p, allow_pickle=True)
    df_index = obj["df_index"].astype(np.int64)

    if "true_label" in obj.files:
        raw = obj["true_label"]

        def norm(x):
            if isinstance(x, bytes):
                x = x.decode("utf-8", errors="ignore")
            return str(x).strip()

        labels = np.array([norm(x) for x in raw], dtype=object)
    else:
        labels = np.array(["UNKNOWN"] * len(df_index), dtype=object)

    out: Dict[str, List[int]] = {}
    for lab in np.unique(labels):
        out[str(lab)] = df_index[labels == lab].tolist()
    return out


@st.cache_resource
def load_npz_cached(npz_path: str) -> Optional[dict]:
    p = Path(npz_path)
    if not p.exists():
        return None
    obj = np.load(p, allow_pickle=True)
    return {k: obj[k] for k in obj.files}


@st.cache_resource
def load_json_cached(json_path: str) -> Optional[dict]:
    p = Path(json_path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


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


def render_cluster_explorer(df: pd.DataFrame, P: Paths):
    st.header("Cluster Explorer")

    # 현재는 unknown 클러스터부터 (이미 만들어져 있으니까)
    cluster_npz = str(P.emb_db / "unknown_cluster.npz")
    emb_npz = str(P.emb_db / "unknown_embeddings.npz")
    reps_json = str(P.emb_db / "unknown_cluster_reps.json")

    cobj = load_npz_cached(cluster_npz)
    eobj = load_npz_cached(emb_npz)
    reps = load_json_cached(reps_json) or {}

    if cobj is None or eobj is None:
        st.error("unknown cluster/embedding files not found.")
        st.code(f"need:\n- {cluster_npz}\n- {emb_npz}")
        st.stop()

    df_idx = cobj["df_index"].astype(np.int64)
    cid = cobj["cluster_id"].astype(np.int32)

    # label map (unknown_embeddings.npz의 true_label 활용)
    label_key = "true_label" if "true_label" in eobj else None
    dfi2label = {}
    if label_key:
        e_dfi = eobj["df_index"].astype(np.int64)
        e_lab = eobj[label_key]

        def _norm(x):
            if isinstance(x, bytes):
                x = x.decode("utf-8", errors="ignore")
            return str(x).strip()

        for i in range(len(e_dfi)):
            dfi2label[int(e_dfi[i])] = _norm(e_lab[i])

    # cluster list (noise=-1 제외)
    valid = cid != -1
    uniq = np.unique(cid[valid])
    # 큰 클러스터부터 정렬
    uniq = sorted([int(x) for x in uniq], key=lambda k: int((cid == k).sum()), reverse=True)

    # summary table
    rows = []
    for k in uniq:
        members = df_idx[cid == k]
        n = int(len(members))
        rep = reps.get(str(k), reps.get(k, None))
        rep = int(rep) if rep is not None else int(members[0])

        # label distribution/purity (가능하면)
        purity = None
        top_label = None
        if dfi2label:
            labs = [dfi2label.get(int(x), "NA") for x in members.tolist()]
            from collections import Counter
            cc = Counter(labs)
            top_label, top_cnt = cc.most_common(1)[0]
            purity = float(top_cnt) / float(n) if n > 0 else None

        rows.append({
            "cluster_id": k,
            "count": n,
            "rep_df_index": rep,
            "top_label": top_label,
            "purity": purity,
        })

    st.subheader("Clusters")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # pick cluster
    options = [f"cid={r['cluster_id']} (n={r['count']})" for r in rows]
    pick = st.selectbox("Select cluster", options, index=0)
    pick_cid = int(pick.split()[0].split("=")[1])

    members = df_idx[cid == pick_cid].astype(np.int64)
    st.write(f"- cluster_id: `{pick_cid}` / members: `{len(members)}`")

    rep_df = next((r["rep_df_index"] for r in rows if r["cluster_id"] == pick_cid), int(members[0]))

    st.subheader("Representative")
    render_wafer_card(df, int(rep_df), title="Cluster representative", subtitle="대표 샘플", scale=10)

    st.subheader("Member gallery (sample)")
    seed = st.number_input("seed", min_value=0, max_value=10_000_000, value=42, step=1)
    n_show = st.slider("num samples", 0, 60, 20)
    rng = np.random.RandomState(int(seed))

    if n_show > 0 and len(members) > 0:
        sel = members if len(members) <= n_show else rng.choice(members, size=n_show, replace=False)
        cols = st.columns(5)
        for i, dfi in enumerate(sel.tolist()):
            with cols[i % 5]:
                wafer = np.array(df.loc[int(dfi)]["waferMap"])
                lab = dfi2label.get(int(dfi), None)
                cap = f"df={int(dfi)}" + (f" | {lab}" if lab else "")
                st.image(wafer_to_pil(wafer, scale=6), caption=cap, use_container_width=True)


def run_case_subprocess(
        project_root: Path,
        df_index: Optional[int],
        k_known: int,
        k_unk: int,
        seed: int,
) -> Tuple[int, str]:
    """
    streamlit 안에서 scripts/run_case.py를 subprocess로 실행
    - PYTHONPATH에 project_root를 넣어서 wmi_triage import 안정화
    - seed를 넘겨서 랜덤 선택도 재현 가능
    """
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_case.py"),
        "--k-known", str(k_known),
        "--k-unk", str(k_unk),
        "--seed", str(seed),
    ]
    if df_index is not None:
        cmd += ["--df-index", str(df_index)]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        shell=False,
        env=env,
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
        ["Browse saved cases", "Open summary.json", "Run new case (optional)", "Explore clusters"],
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

    elif mode == "Explore clusters":
        render_cluster_explorer(df, P)
        st.stop()

    else:
        st.sidebar.write("Generate a new case by calling scripts/run_case.py")

        unk_npz = str(P.emb_db / "unknown_embeddings.npz")
        unk_pool = load_unknown_pool_cached(unk_npz)

        # 선택 모드
        pick_mode = st.sidebar.radio(
            "Pick query",
            ["Random (any unknown)", "Random by unknown type", "By df_index"],
            index=1,
        )

        # 옵션들
        k_known = st.sidebar.number_input("k_known", min_value=1, max_value=50, value=5, step=1)
        k_unk = st.sidebar.number_input("k_unk", min_value=1, max_value=50, value=5, step=1)
        seed = st.sidebar.number_input("seed", min_value=0, max_value=10_000_000, value=42, step=1)

        df_index_text = ""
        unk_type = None

        # Scratch/Donut 우선 노출, 없으면 자동 fallback
        preferred = [t for t in ["Scratch", "Donut"] if t in unk_pool and len(unk_pool[t]) > 0]
        all_types = sorted([t for t, lst in unk_pool.items() if len(lst) > 0])

        if pick_mode == "By df_index":
            df_index_text = st.sidebar.text_input("df_index", value="")

        elif pick_mode == "Random by unknown type":
            options = preferred if preferred else all_types
            if not options:
                st.sidebar.warning("unknown pool이 비어있어. unknown_embeddings.npz를 확인해줘.")
            else:
                unk_type = st.sidebar.selectbox("unknown type", options, index=0)
                st.sidebar.caption(f"pool size: {len(unk_pool.get(unk_type, []))}")

        # -----------------
        # Preview (Run 전에 보여주기)
        # -----------------
        preview_dfi: Optional[int] = None
        preview_note: str = ""

        rng = np.random.RandomState(int(seed))

        if pick_mode == "By df_index":
            if df_index_text.strip():
                try:
                    cand = int(df_index_text)
                    if cand in df.index:
                        preview_dfi = cand
                        preview_note = "manual df_index"
                    else:
                        preview_note = "df_index not found in df"
                except ValueError:
                    preview_note = "invalid df_index"

        elif pick_mode == "Random by unknown type" and unk_type is not None:
            candidates = unk_pool.get(unk_type, [])
            if candidates:
                preview_dfi = int(rng.choice(candidates))
                preview_note = f"type={unk_type}"
            else:
                preview_note = f"no candidates for type={unk_type}"

        else:
            # Random(any unknown): 전체 unknown pool에서 랜덤
            all_candidates = []
            for lst in unk_pool.values():
                all_candidates.extend(lst)
            if all_candidates:
                preview_dfi = int(rng.choice(all_candidates))
                preview_note = "any unknown"
            else:
                preview_note = "unknown pool is empty"

        st.sidebar.markdown("### Preview")
        if preview_dfi is not None and preview_dfi in df.index:
            wafer = np.array(df.loc[preview_dfi]["waferMap"])
            st.sidebar.write(f"- df_index: `{preview_dfi}`")
            st.sidebar.write(f"- note: `{preview_note}`")
            st.sidebar.image(wafer_to_pil(wafer, scale=6), use_container_width=True)
        else:
            st.sidebar.info(preview_note if preview_note else "No preview")

        # -----------------
        # Run (Preview에서 확정된 df_index를 그대로 사용)
        # -----------------
        if st.sidebar.button("Run"):
            # 실행 df_index 결정
            if pick_mode == "By df_index":
                dfi = int(df_index_text) if df_index_text.strip() else None
            elif pick_mode == "Random by unknown type":
                dfi = preview_dfi  # ✅ preview와 동일하게
            else:
                dfi = preview_dfi  # ✅ preview와 동일하게

            code, log = run_case_subprocess(P.root, dfi, int(k_known), int(k_unk), int(seed))
            st.sidebar.code(log)
            if code != 0:
                st.error("run_case failed. See log in sidebar.")
                st.stop()

            # 최신 case 자동 로드
            cases = list_case_summaries(P.cases)
            if not cases:
                st.error("Case created but cannot find summary.json under artifacts/cases")
                st.stop()  # ✅ 괄호 꼭!
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
