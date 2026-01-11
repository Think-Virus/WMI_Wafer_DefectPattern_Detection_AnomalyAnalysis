# Open-set Wafer Map Triage (WM-811K) — MVP 프로젝트

WM-811K(LSWMD.pkl) wafer map 데이터로  
1) **Known 결함 패턴 분류**를 학습하고  
2) **학습에 없는 결함(Unknown)을 OOD 점수로 탐지**하며  
3) Unknown 발생 시 **유사사례/군집/시각화로 트리아지**를 제공하는 것을 목표로 하는 프로젝트입니다.

> 핵심 컨셉: “분류 정확도”가 아니라 **Unknown을 ‘Unknown’으로 거부(reject)하고 이후 대응을 돕는 워크플로우** 구현

---

## 0) 최근 변경사항 요약 (V2 개선)
- ✅ **`none` 제외 학습**: known 분류/지표 해석이 왜곡되지 않도록 학습 split에서 `none` 제거
- ✅ **좌표 채널(coords4) 입력 + ResNet conv1 수정(4ch)**: `Loc`/`Edge-Loc` 등 유사 패턴 구분 개선 목적
- ✅ **체크포인트 기반 분석 안정화**
  - ckpt의 `class_to_idx/known_classes/MODEL_CFG`를 우선 사용
  - 구버전(3ch repeat3) ckpt와 신버전(4ch coords4) ckpt를 **같은 노트에서 번갈아 로드해도 채널 mismatch 없이 분석**
- ✅ **Random threshold 스윕(후처리) 추가**: Random 과대예측을 줄이기 위한 post-hoc rule + val 기반 t 선택
- ✅ **UMAP 시각화 개선**: 클래스별 legend 표시(known + unknown overlay)
- ✅ **결과/리포트 자동 저장(reports/)**: runmeta, CM, classification_report, UMAP 좌표/이미지 저장
- ✅ **Thin Scratch 커버리지 진단(MATCH/NO MATCH) 추가**
  - Scratch unknown 중 `line_score` 상위 20%를 **thin-scratch query**로 정의
  - Loc-family(Loc+Edge-Loc) ref 3000개 중 `line_score` 상위 1%(43개)를 **line-like ref(선형 exemplar 후보)**로 정의
  - 각 thin-scratch query에 대해 Loc-family **Local Top-50** 안에 line-like ref가 존재하면 **MATCH**, 없으면 **NO MATCH(coverage gap)** 로 분리
  - 산출물: `reports/thin_scratch_match_nomatch_cases.csv`, `assets/thin_scratch_match_nomatch/{match,nomatch}/`

---

## 1) 프로젝트 목표 (작품/제출 관점)
- 단순 wafer map 분류(Closed-set)는 선행 사례가 많아 novelty가 약할 수 있음
- 제조 현실에서는 **학습에 없던 신종 결함 패턴(Unknown)**이 발생 가능
- 본 프로젝트는 다음을 목표로 함
  - **Unknown 과신(잘못된 확신) 오분류를 줄이고**
  - Unknown 발생 후 엔지니어가 빠르게 판단하도록 **트리아지(정리/의사결정 지원)** 제공
- 제출물: **시연 영상 + 포스터 + 슬라이드**

---

## 2) 현재 상태 (진행 단계)
### ✅ MVP-1 (완료): Known 분류 + OOD(Unknown 탐지) 베이스라인
- labeled 데이터(`failureType` 존재)만 사용
- 클래스 홀드아웃으로 Open-set 평가 구성 (예: Donut/Scratch → Unknown)
- ResNet18 기반 known 분류
- OOD 점수 **MSP vs Energy**로 AUROC/AUPR 평가
- 체크포인트 저장/로드 유틸(Drive의 `checkpoints/`)

### ✅ MVP-2 (진행): Triage(트리아지) 구현/고도화
- ✅ **임베딩 기반 2D 시각화(UMAP)**: known vs unknown overlay
- ✅ **유사사례 Top-K 검색(임베딩 기반 retrieval)**
- ✅ unknown→unknown 유사사례(Top-K) 및 클러스터링(DBSCAN)
- ✅ assets/triage_unknown/에 군집 대표 패널 및 요약 CSV 저장
- ✅ reports/<run_id>/에 분석 산출물 저장(runmeta/report/cm/umap…)
- ✅ **Thin Scratch 커버리지 진단(MATCH/NO MATCH)**: thin-scratch query에서 Loc-family 선형 exemplar 존재 여부를 분리(MATCH/NO MATCH)

### ▶ MVP-3 (개선/고도화): 성능 및 OOD 강화
...
- 학습/검증/테스트(known): 홀드아웃 제외 클래스
- 테스트(unknown): 홀드아웃 클래스만 모아 Unknown으로 평가

> WM-811K에는 원래 Unknown 라벨이 없으므로,  
> **“학습에서 제외한 클래스를 Unknown으로 간주”**하여 객관적으로 평가합니다.

### `none` 처리 정책 (V2)
- 기본 정책: **학습 split에서 `none` 제외**
  - `EXCLUDE_CLASSES = ["none"]`
- 단, 과거 ckpt(구버전) 재현/비교 분석 시:
  - ckpt의 `class_to_idx`에 `none`이 포함되어 있으면 **분석 시 exclude를 완화**하거나
  - 최소한 **loader가 ckpt mapping 기준으로 필터링되도록 유지**합니다.

---

## 4) 구현 개요
### 모델 (Known 분류)
- Backbone: `torchvision.models.resnet18` + `fc` 재정의
- 입력 전처리:
  - wafer map → (1, H, W) → resize(기본 64×64, nearest)
  - **입력 모드 2종**
    - `repeat3`: (1ch) → 3ch 복제 (구버전/호환)
    - `coords4`: (1ch) + 좌표채널(x,y,r) → 4ch (신버전)
- 증강(Train only): 좌우/상하 flip + 90도 회전

> **중요: ckpt마다 입력 채널 수가 다를 수 있으므로**  
> loader/dataset은 `input_mode`(coords4 vs repeat3)를 모델(conv1.in_channels) 또는 MODEL_CFG 기준으로 자동 선택해야 합니다.

### OOD(Unknown 탐지)
- MSP(Max Softmax Probability)
- Energy score (logits 기반)
- AUROC/AUPR로 known vs unknown 분리 성능 평가

### Random threshold (후처리, 선택)
- “argmax가 Random”인데 Random 확률이 낮으면(임계값 미만) **2등 클래스로 이동**
- val에서 t를 스윕해 macro-F1 기준 최적 t를 선택 → test에 적용

### Triage(트리아지)
- 임베딩 추출: ResNet18의 fc 직전 특징(avgpool 출력 등)
- UMAP: known/unknown 분포 및 혼재 정도 확인(정성 분석)
- retrieval: unknown별 Top-K 유사 known(또는 unknown) 예시 제공
- clustering: unknown→unknown 군집화(DBSCAN 등), 군집 대표 패널 저장
- **Thin Scratch 커버리지(MATCH/NO MATCH)**
  - thin-scratch query에 대해 Loc-family Local Top-50 안에 line-like ref가 존재하면 MATCH, 없으면 NO MATCH(coverage gap)
  - 각 케이스에 대해 Local Top-K vs line-like pool Top-K를 나란히 저장하여 근거(정성) 확보

---

## 5) 결과 스냅샷 (예시)
> 실험/시드/홀드아웃 조합에 따라 달라질 수 있습니다.  
> 프로젝트에서는 accuracy보다 **macro-F1 + OOD AUROC** 중심으로 보고합니다.

- Known 분류(test_known): macro-F1 ~ 0.90대 (최근 run 기준)
- OOD(holdout Donut/Scratch): AUROC(MSP/Energy) ~ 0.85~0.86대 (최근 run 기준)
- UMAP 관찰:
  - unknown이 known과 일부 섞여도 **Top-K/군집 대표 패널로 사람이 빠르게 판단**하도록 돕는 방향이 설득력 있음.

---

## 6) 실행 방법 (Google Colab 권장)
1. Google Drive에 `LSWMD.pkl`을 준비합니다.
2. Colab에서 노트북을 열고 **Run all** 실행합니다.
3. 데이터 경로(`pd.read_pickle(...)`)는 본인 Drive 구조에 맞게 수정합니다.

노트북:
- `WMI_Wafer_DefectPattern_Detection_AnomalyAnalysis.ipynb`

---

## 7) 체크포인트(checkpoints) 저장/로드
- 체크포인트는 Google Drive의 `checkpoints/` 폴더에 저장합니다.
- GitHub에는 체크포인트를 올리지 않습니다(레포 용량/관리 이슈).
- PyTorch 2.6+ 환경에서 구형 ckpt 로드시 `torch.load(..., weights_only=False)`가 필요할 수 있습니다.
  - **주의:** 신뢰 가능한(본인이 생성한) ckpt에만 사용

---

## 8) 산출물 저장 구조 (권장)
- `assets/triage/` : known 기반 데모 패널/요약
- `assets/triage_unknown/` : unknown 군집 대표 패널/요약
- `assets/thin_scratch_match_nomatch/`
  - `match/` : Local Top-50에 line-like ref가 존재하는 thin-scratch 케이스
  - `nomatch/` : Local Top-50에 line-like ref가 없는(coverage gap) thin-scratch 케이스
- `reports/`
  - `thin_scratch_match_nomatch_cases.csv` : MATCH/NO MATCH 케이스 요약(unk_i, q_line_score, best_line_like_rank 등)
- `reports/<run_id>/`
  - `runmeta.json` (cfg/seed/ckpt/class mapping)
  - `test_report.txt`
  - `test_cm_row_norm.png`
  - `umap_known_unknown.png`, `umap_Z.npy`, `umap_y.npy`
  - (선택) `ref_emb.npy`, `unk_emb.npy` 등