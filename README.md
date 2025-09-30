# 규칙 기반 낙상 감지 알고리즘 (Fall Detection System)

[![Language: Python](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 💡 프로젝트 개요 및 배경

본 프로젝트는 **YOLOv8** 객체 감지 모델과 **MediaPipe Pose**를 결합하여 노인 및 환자의 낙상 사고를 실시간으로 감지하는 고신뢰성 시스템입니다.

### 문제 해결: 원근 문제와 오탐지 방지

1. **원근 문제 해결**: 순수 MediaPipe 뼈대 데이터는 인물이 카메라와 멀어지거나 가까워질 때 영상상 크기 변화(원근 변화)에 취약합니다. 이를 해결하기 위해 **YOLO의 바운딩 박스(BBox) 중심점과 종횡비**를 메인 지표로 활용하여, 원근 변화에 강인한 낙상 지표를 확보했습니다.
2. **오탐지 방지**: 단순히 넘어지는 동작(False Positive)과 **실제 움직임 없는 사고**를 구분하기 위해, 시간 기반의 **다단계 상태 추론 로직 (Stillness Tracking)**을 핵심적으로 구현했습니다.

---

## ⚙️ 시스템 아키텍처 및 기술 스택

시스템은 기능별 모듈로 분리되어 유기적으로 연결된 파이프라인 구조를 갖습니다.

### 1. 주요 기술 스택

| 역할 | 모델/라이브러리 | 목적 |
| :----- | :----- | :----- |
| **객체 감지 (BBox)** | `ultralytics` (YOLOv8n) | 사람 객체 감지 및 추적 |
| **자세 추정 (Pose)** | `mediapipe` | 인체 랜드마크(관절 좌표) 추정 및 시각화 (보조 역할) |
| **핵심 로직** | Python (규칙 기반 FSM) | Y축 속도, 비율, **정지 시간**을 이용한 최종 사고 확정 |
| **실행 환경** | `opencv-python (cv2)` | 비디오 스트림 처리 및 실시간 결과 시각화 |

### 2. 모듈 구조 (`src/` 폴더)

| 파일 | 클래스 | 역할 (모듈화) |
| :----- | :----- | :----- |
| `detector.py` | `YoloDetector` | YOLOv8n을 사용하여 영상 프레임 내 사람 객체를 감지하고 바운딩 박스(bbox)를 출력합니다. |
| `pose_estimator.py` | `PoseEstimator` | MediaPipe Pose를 사용하여 인체 랜드마크를 추출하고 시각화합니다. |
| `fall_logic.py` | **`FallDetectorLogic`** | **낙상 판단의 핵심 로직 및 다단계 상태 추적**. |
| `main.py` | `main()` | 전체 파이프라인의 엔트리 포인트(Entry Point)로, 각 모듈을 통합하고 결과를 출력합니다. |

---

## 🧠 핵심 로직: Stillness Tracking 상세

알고리즘은 5가지 상태(`Standing`, `Sitting`, `Lying`, `Potential Fall`, `Fall Detected!`)를 정의하며, 오탐지 방지를 위해 **3초간의 정지 시간**을 필수 조건으로 사용합니다.

### 1. 주요 판단 지표 임계값

| 지표 | 값 | 역할 |
| :----- | :----- | :----- |
| Velocity_Y | $\mathbf{150} \text{ 픽셀/초}$ | 중심점의 급격한 하강 감지 (1차 경보) |
| $\Delta x / \Delta y$ | $\mathbf{1.0}$ | 바운딩 박스 종횡비; 누운 자세(`Lying`) 판별 |
| STILLNESS_TIME | $\mathbf{3}$ 초 | 사고 확정을 위한 최소 정지 시간 |
| STILLNESS_Y | $\mathbf{5}$ 픽셀 | 정지 상태로 간주할 Y축 중심점 변화 최대치 |

### 2. 다단계 상태 전환 메커니즘

| 상태 | 설명 | 전환 조건 |
| :----- | :----- | :----- |
| `Standing` | 정상 상태 | - |
| `Potential Fall` | 낙상 가능성 (1차 경보) | Velocity_Y 임계값 초과 시 **타이머 시작** |
| **`Fall Detected!`** | **최종 사고 확정** | **`Potential Fall` & 누운 자세 & 3초 이상 정지 상태 유지** |
| `Lying` | 단순히 누워있는 자세 | (낙상 속도 없이 비율만 충족) |

---

## 📊 테스트 결과 (Stillness Logic 검증 성공)

테스트 결과, 알고리즘은 **3초의 고정된 지연 시간**을 거쳐 낙상 사고를 성공적으로 확정했습니다. 이는 단순 누움 상태와 실제 긴급 상황을 명확히 구분했음을 의미합니다.

| 프레임 | 상태 | 설명 |
| :----- | :----- | :----- |
| **0 ~ 70** | `Standing` | 초기 정상 상태 |
| **70** | `Potential Fall` | 급격한 움직임 감지, **타이머 시작** |
| **70 ~ 130** | `Potential Fall` | **3초 (약 60프레임)** 동안 정지 상태 유지 확인 |
| **130 이후** | **`Fall Detected!`** | 정지 시간(3초) 초과 후, 최종 사고 확정 |

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
