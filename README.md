# Open-set Wafer Map Triage (WM-811K) — MVP 프로젝트

WM-811K(LSWMD.pkl) wafer map 데이터로  
1) **Known 결함 패턴 분류**를 학습하고  
2) **학습에 없는 결함(Unknown)을 OOD 점수로 탐지**하며  
3) (다음 단계) Unknown 발생 시 **유사사례/군집/시각화로 트리아지**를 제공하는 것을 목표로 하는 프로젝트입니다.

> 핵심 컨셉: “분류 정확도”가 아니라 **Unknown을 ‘Unknown’으로 거부(reject)하고 이후 대응을 돕는 워크플로우**를 구현합니다.

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
### ✅ MVP-1 (완료): Known 분류 + OOD(Unknown 탐지) 평가
- labeled 데이터(`failureType` 존재)만 사용
- 클래스 홀드아웃으로 Open-set 평가 구성 (예: Donut/Scratch → Unknown)
- ResNet18(경량 CNN) 기반 known 분류
- OOD 점수 **MSP vs Energy**로 AUROC/AUPR 평가
- 체크포인트 저장/로드 유틸(Drive의 `checkpoints/`)

### ▶ MVP-2 (다음 목표): Triage(트리아지) 구현
- Unknown 판정 이후
  - **유사사례 Top-K 검색(임베딩 기반 retrieval)**
  - **UMAP/t-SNE 시각화**
  - (가능 시) **클러스터링**으로 “신규 패턴 후보군” 묶기

### ▶ MVP-3 (개선/고도화): 성능 및 OOD 강화
- 클래스 불균형 대응(샘플러/가중치/손실함수)
- 해상도 비교(64→96/128)
- pretrained backbone 비교
- OOD 강화: **임베딩 거리(kNN distance)** 기반 OOD 점수 추가

### (선택) 확장 과제
- 라벨 부족을 활용하는 자기지도/반지도 학습(시간 여력 시)

---

## 3) 데이터셋 및 평가 설계 (Open-set Protocol)
### 데이터
- WM-811K (LSWMD.pkl)
- 주요 컬럼:
  - `waferMap`: 2D numpy array (보통 0/1/2 값)
  - `failureType`: 결함 라벨(list 형태 또는 빈 list)

### MVP-1 범위
- `failureType`이 있는 샘플만 사용 (`failureType == []`는 제외)

### Open-set 구성(클래스 홀드아웃)
- `UNKNOWN_CLASSES = ["Donut", "Scratch"]` (예시)
- 학습/검증/테스트(known): 홀드아웃 제외 클래스
- 테스트(unknown): 홀드아웃 클래스만 모아 Unknown으로 평가

> 디펜스 포인트: WM-811K에는 원래 Unknown 라벨이 없으므로,  
> **“학습에서 제외한 클래스를 Unknown으로 간주”**하여 객관적으로 평가합니다.

---

## 4) 구현 개요 (MVP-1)
### 모델 (Known 분류)
- Backbone: `torchvision.models.resnet18` (weights=None) + `fc` 재정의
- 입력 전처리:
  - wafer map → (1, H, W) → resize(기본 64×64, nearest)
  - ResNet 입력 맞추기 위해 1채널을 3채널로 복제
- 증강(Train only): 좌우/상하 flip + 90도 회전

### OOD(Unknown 탐지)
- MSP(Max Softmax Probability)
- Energy score (logits 기반)
- AUROC/AUPR로 known vs unknown 분리 성능 평가

---

## 5) 결과 스냅샷 (현재 베이스라인, MVP-1)
### Known 분류 (test_known)
- macro-F1: **0.8715**
- 관찰:
  - `none` 클래스가 매우 많아(지원 샘플 수 큼) accuracy/weighted avg는 높게 나올 수 있음
  - `Loc`, `Edge-Loc` 등 일부 결함 클래스가 상대적으로 약함

### OOD(Unknown 탐지) (test_unknown = holdout Donut/Scratch)
- AUROC (MSP): **0.8619**
- AUPR  (MSP): **0.9886** *(known=positive 설정 기준)*
- AUROC (Energy): **0.8524**
- AUPR  (Energy): **0.9877** *(known=positive 설정 기준)*

---

## 6) (중요) 현재 단계(MVP-1)의 한계와 개선 방향
### 1) 클래스 불균형 영향이 큼 (`none` 과다)
- `none`이 많아 전체 accuracy가 높아 보일 수 있으나,
  결함 클래스(예: `Loc`, `Edge-Loc`) 성능이 상대적으로 낮을 수 있음.

**개선 방향**
- `WeightedRandomSampler`로 균형 배치 구성(가성비 최고)
- 또는 class-weighted loss / focal loss 적용
- 보고 지표는 accuracy보다 **macro-F1** 중심 유지 (+ 필요 시 `none` 제외 macro-F1 병기)

### 2) Loc vs Edge-Loc 등 유사 패턴 혼동
- 시각적으로 유사한 결함은 오분류가 발생할 수 있음.

**개선 방향**
- resize 64 → 96/128 비교
- pretrained backbone 비교(가능 시)

### 3) 소수 클래스(Near-full 등) 지표 변동성
- 샘플 수가 매우 적으면 지표가 흔들릴 수 있음.

**개선 방향**
- seed 고정 및 반복 실험(여력 시)
- 홀드아웃 조합 여러 개로 검증(확장 시)

### 4) OOD는 베이스라인(MSP/Energy) 수준
- 구현이 간단한 대신, 데이터/모델에 따라 한계가 있을 수 있음.

**개선 방향**
- 임베딩 거리(kNN distance) 기반 OOD 점수 추가 (트리아지와 자연스럽게 연결)

### 5) 현재는 “탐지”까지, 트리아지는 다음 단계(MVP-2)
- Unknown을 “Unknown”으로 판정하는 단계는 완료
- Unknown 이후 “정리/의사결정 지원(트리아지)”는 구현 예정

---

## 7) 실행 방법 (Google Colab 권장)
1. Google Drive에 `LSWMD.pkl`을 준비합니다.
2. Colab에서 아래 노트북을 열고 **Run all** 실행합니다.
3. 데이터 경로(`pd.read_pickle(...)`)는 본인 Drive 구조에 맞게 수정합니다.

노트북:
- `WMI_Wafer_DefectPattern_Detection_AnomalyAnalysis.ipynb`

---

## 8) 체크포인트(checkpoints) 저장 정책
- 모델 체크포인트는 Google Drive의 `checkpoints/` 폴더에 저장합니다.
- GitHub에는 체크포인트를 올리지 않습니다(레포 용량/관리 이슈).

