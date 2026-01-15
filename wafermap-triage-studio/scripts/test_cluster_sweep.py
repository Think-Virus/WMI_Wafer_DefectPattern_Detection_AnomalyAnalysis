# scripts/test_cluster_sweep.py
from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from wmi_triage.config import Paths


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def run_dbscan_cosine(emb_n: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    return db.fit_predict(emb_n).astype(np.int32)


def run_kmeans_on_cosine(emb_n: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """
    emb를 l2-normalize한 뒤 KMeans(Euclidean)를 하면
    cosine 기반 분할과 거의 같은 효과를 냄.
    """
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    return km.fit_predict(emb_n).astype(np.int32)


def purity_stats(cluster_id: np.ndarray, true_label: np.ndarray) -> dict:
    by_cluster = defaultdict(list)
    for cid, lab in zip(cluster_id.tolist(), true_label.tolist()):
        by_cluster[int(cid)].append(str(lab))

    weighted_sum = 0.0
    weighted_n = 0
    per = {}

    for cid, labs in by_cluster.items():
        n = len(labs)
        c = Counter(labs)
        top_lab, top_cnt = c.most_common(1)[0]
        p = top_cnt / n if n else 0.0
        per[cid] = {"n": n, "top_label": top_lab, "top_ratio": p, "dist": dict(c)}
        if cid != -1:
            weighted_sum += p * n
            weighted_n += n

    return {
        "per_cluster": per,
        "weighted_purity_excl_noise": (weighted_sum / weighted_n) if weighted_n else 0.0,
        "noise_count": int((cluster_id == -1).sum()),
    }


def brief_clusters(pur: dict, max_clusters: int = 6) -> str:
    per = pur["per_cluster"]
    items = [(cid, info["n"], info["top_label"], info["top_ratio"], info["dist"])
             for cid, info in per.items() if cid != -1]
    items.sort(key=lambda t: t[1], reverse=True)

    lines = []
    for cid, n, top_lab, top_ratio, dist in items[:max_clusters]:
        lines.append(f"    - cid={cid:>3} n={n:<4} top={top_lab}({top_ratio:.2f}) dist={dist}")
    if len(items) > max_clusters:
        lines.append(f"    ... ({len(items) - max_clusters} more clusters)")
    return "\n".join(lines) if lines else "    (no clusters excl noise)"


def label_separation_probe(emb_n: np.ndarray, true_label: np.ndarray, a: str = "Scratch", b: str = "Donut",
                           max_per_class: int = 400, seed: int = 42) -> None:
    """
    Scratch/Donut이 임베딩에서 분리 가능한지 빠른 감(평균 cosine 유사도) 체크.
    - intra(A,A), intra(B,B), inter(A,B) 평균 cos sim 출력
    """
    rng = np.random.RandomState(seed)

    labels = np.array([str(x) for x in true_label.tolist()])
    ia = np.where(labels == a)[0]
    ib = np.where(labels == b)[0]
    if len(ia) == 0 or len(ib) == 0:
        print("[probe] missing label group:", a, len(ia), b, len(ib))
        return

    ia = rng.choice(ia, size=min(len(ia), max_per_class), replace=False)
    ib = rng.choice(ib, size=min(len(ib), max_per_class), replace=False)

    A = emb_n[ia]
    B = emb_n[ib]

    # cosine sim = dot (normalized)
    intra_A = (A @ A.T)
    intra_B = (B @ B.T)
    inter_AB = (A @ B.T)

    # 대각 제거 평균
    def offdiag_mean(M):
        if M.shape[0] <= 1:
            return float(M.mean())
        return float((M.sum() - np.trace(M)) / (M.size - M.shape[0]))

    print("\n=== label separation probe (mean cosine sim) ===")
    print(f"intra {a}-{a}: {offdiag_mean(intra_A):.4f}")
    print(f"intra {b}-{b}: {offdiag_mean(intra_B):.4f}")
    print(f"inter {a}-{b}: {float(inter_AB.mean()):.4f}")
    print("-> inter가 intra보다 많이 낮으면 분리 여지 큼 / 비슷하면 임베딩이 겹친다는 신호")


def main():
    P = Paths()
    unk_npz = P.emb_db / "unknown_embeddings.npz"

    obj = np.load(unk_npz, allow_pickle=True)
    emb = obj["emb"].astype(np.float32)
    true_label = obj["true_label"] if "true_label" in obj.files else None
    if true_label is None:
        raise RuntimeError("unknown_embeddings.npz에 true_label이 없어.")

    emb_n = l2norm(emb)
    print("[load] emb:", emb.shape, "labels:", Counter([str(x) for x in true_label.tolist()]))

    # 0) 분리력 감 체크
    label_separation_probe(emb_n, true_label, a="Scratch", b="Donut")

    # 1) DBSCAN sweep: eps를 더 작게까지 내려봄
    eps_list = [0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03]
    min_samples_list = [10, 8, 6, 4, 3]

    best_db = None  # (score, eps, min_samples, n_clusters, noise_ratio, purity)
    print("\n=== DBSCAN sweep (cosine) ===")
    for ms in min_samples_list:
        for eps in eps_list:
            cid = run_dbscan_cosine(emb_n, eps=eps, min_samples=ms)
            cnt = Counter(cid.tolist())
            n_clusters = len([k for k in cnt.keys() if k != -1])
            noise = cnt.get(-1, 0)
            noise_ratio = noise / len(cid)

            pur = purity_stats(cid, true_label)
            wp = pur["weighted_purity_excl_noise"]

            # 스코어: purity 높고 noise 낮고, cluster>=2 선호
            score = wp - 0.6 * noise_ratio - (0.25 if n_clusters < 2 else 0.0)

            if best_db is None or score > best_db[0]:
                best_db = (score, eps, ms, n_clusters, noise_ratio, wp)

            print(
                f"\n[DBSCAN] eps={eps:.2f} ms={ms:<2} -> clusters={n_clusters}, noise={noise}({noise_ratio:.1%}), wpurity={wp:.3f}, score={score:.3f}")
            print(brief_clusters(pur, max_clusters=4))

    # 2) KMeans sweep: 강제 분할로 분리 가능한지 확인
    k_list = [2, 3, 4, 5, 6, 8]
    best_km = None  # (score, k, purity)
    print("\n=== KMeans sweep on normalized emb (cosine-ish) ===")
    for k in k_list:
        cid = run_kmeans_on_cosine(emb_n, k=k, seed=42)
        cnt = Counter(cid.tolist())
        n_clusters = len(cnt.keys())
        pur = purity_stats(cid, true_label)
        wp = pur["weighted_purity_excl_noise"]

        # kmeans는 noise가 없으니 purity 중심
        score = wp - 0.02 * (k - 2)  # 너무 큰 k 약간 페널티

        if best_km is None or score > best_km[0]:
            best_km = (score, k, wp)

        print(f"\n[KMeans] k={k:<2} -> clusters={n_clusters}, wpurity={wp:.3f}, score={score:.3f}")
        print(brief_clusters(pur, max_clusters=4))

    print("\n=== BEST suggestions ===")
    if best_db:
        score, eps, ms, n_clusters, noise_ratio, wp = best_db
        print(f"[DBSCAN best] eps={eps:.2f} ms={ms} clusters={n_clusters} noise={noise_ratio:.1%} wpurity={wp:.3f} score={score:.3f}")
        print(f"  -> run: python scripts/run_clustering.py --eps {eps:.2f} --min-samples {ms}")
    if best_km:
        score, k, wp = best_km
        print(f"[KMeans best] k={k} wpurity={wp:.3f} score={score:.3f}")
        print("  -> (추천) DBSCAN이 계속 1덩어리면 KMeans로 '후보군 분해' 하는 게 MVP에 더 유리함.")


if __name__ == "__main__":
    main()
