# 베이즈 추론 기반 낙상 감지 알고리즘 (Fall Detection System)

[![Language: Python](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 💡 프로젝트 개요 및 배경

본 프로젝트는 **YOLOv8** 객체 감지 모델과 **MediaPipe Pose**를 결합하고, 바운딩 박스 특징에 대해 **베이즈 추론(로그-오즈 누적)** 을 적용하여 낙상 사고를 실시간으로 감지하는 시스템입니다.

### 문제 해결: 원근 문제와 오탐지 방지

1. **원근 문제 해결**: 순수 MediaPipe 뼈대 데이터는 인물이 카메라와 멀어지거나 가까워질 때 영상상 크기 변화(원근 변화)에 취약합니다. 이를 해결하기 위해 **YOLO의 바운딩 박스(BBox) 중심점과 종횡비**를 메인 지표로 활용하여, 원근 변화에 강인한 낙상 지표를 확보했습니다.
2. **오탐지 방지**: 단순히 넘어지는 동작(False Positive)과 **실제 움직임 없는 사고**를 구분하기 위해, 시간 기반의 **다단계 상태 추론 로직 (Stillness Tracking)** 을 핵심적으로 구현했습니다.

---

## ⚙️ 시스템 아키텍처 및 기술 스택

시스템은 기능별 모듈로 분리되어 유기적으로 연결된 파이프라인 구조를 갖습니다.

### 1. 주요 기술 스택

| 역할 | 모델/라이브러리 | 목적 |
| :----- | :----- | :----- |
| **객체 감지 (BBox)** | `ultralytics` (YOLOv8n) | 사람 객체 감지 및 추적 |
| **자세 추정 (Pose)** | `mediapipe` | 인체 랜드마크(관절 좌표) 추정 및 시각화 (보조 역할) |
| **핵심 로직** | Python (Bayesian + FSM) | 속도/비율/정지 여부의 **우도 누적(Posterior)** 로 사고 확정 |
| **실행 환경** | `opencv-python (cv2)` | 비디오 스트림 처리 및 실시간 결과 시각화 |

### 2. 모듈 구조 (`src/` 폴더)

| 파일 | 클래스 | 역할 (모듈화) |
| :----- | :----- | :----- |
| `detector.py` | `YoloDetector` | YOLOv8n을 사용하여 영상 프레임 내 사람 객체를 감지하고 바운딩 박스(bbox)를 출력합니다. |
| `pose_estimator.py` | `PoseEstimator` | MediaPipe Pose를 사용하여 인체 랜드마크를 추출하고 시각화합니다. |
| `fall_logic.py` | **`FallDetectorLogic`** | **낙상 판단의 핵심 로직 및 다단계 상태 추적**. |
| `main.py` | `main()` | 전체 파이프라인의 엔트리 포인트(Entry Point)로, 각 모듈을 통합하고 결과를 출력합니다. |

---

## 🧠 핵심 로직: 베이즈 추론 + 히스테리시스

알고리즘은 5가지 상태(`Standing`, `Sitting`, `Lying`, `Potential Fall`, `Fall Detected!`)를 유지하면서, 각 프레임의 **증거(feature)** 로부터 `P(Fall)`을 업데이트합니다. 의사결정은 확률 임계값 기반으로 이뤄지며, **히스테리시스**로 떨림을 완화합니다.

### 1. 사용 특징과 우도(LLR)

- **Velocity_Y**: 바운딩 박스 중심의 Y축 속도. 가우시안 근사 우도.
- **Aspect Ratio**: 바운딩 박스 가로/세로 비. 가우시안 근사 우도.
- **Stillness**: 프레임 간 Y 변화량이 임계값 미만인지. 베르누이 우도.

각 특징은 `LLR = log p(f|Fall) - log p(f|¬Fall)`로 변환되어 프레임마다 누적됩니다.

### 2. 핵심 파라미터(기본값)

- Prior: `P(Fall) ≈ 1%` → `DEFAULT_LOG_ODDS`
- Transition bias: `TRANSITION_BIAS ≈ log(1.02)` (연속성 미세 가중)
- 결정 임계값: `FALL_THRESHOLD = 0.9`, 복구 임계값: `RECOVER_THRESHOLD = 0.7`
- 정지판정: `STILLNESS_Y_THRESHOLD = 5` 픽셀
- 가우시안 근사 평균/분산: 속도(`VEL_*`), 비율(`AR_*`)은 데이터로 튜닝

### 3. 상태 결정 로직(요지)

- `p_fall = sigmoid(DEFAULT_LOG_ODDS + Σ LLR + TRANSITION_BIAS)`
- `p_fall ≥ 0.9` 이면 `Fall Detected!` 확정(유지)
- `p_fall < 0.7`이면 복구하여 `Lying/Standing` 등으로 전환
- 그 외에는 보조 상태(`Lying/Sitting/Potential Fall/Standing`)를 규칙으로 라벨링

---

## 📊 기대 효과

- 프레임 노이즈/누락에 강한 **시간적 일관성**
- 임계값 기반 규칙 대비 **오탐/미탐 균형 조절 용이** (튜닝으로 조정)
- 히스테리시스로 상태 **깜빡임 감소**

---

## 🛠️ 설치 및 실행 방법

### 1. 환경 설정

```bash
# 가상 환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\Activate       # Windows Powershell

# 필수 라이브러리 설치
pip install ultralytics mediapipe opencv-python numpy
```

### 2. 실행 (실시간 처리)
프로젝트 루트 디렉토리에서 main.py파일을 실행하여 실시간 낙상 감지 결과를 영상으로 확인할 수 있습니다. 

```bash
python src/main.py
```

### 3. 파라미터 튜닝 가이드

- 카메라 해상도/프레임레이트에 따라 `VEL_*`, `AR_*`, `STILL_P_*`, `FALL_THRESHOLD`, `RECOVER_THRESHOLD`, `TRANSITION_BIAS`를 조정하세요.
- 다인 장면에서는 추적기(예: DeepSORT, ByteTrack)로 `track_id`를 안정적으로 유지하면 posterior가 트랙별로 일관되게 누적됩니다.
