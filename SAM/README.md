# SAM-MOT 객체 추적 프로젝트 설치 및 실행 가이드

## 1. 환경 설정

프로젝트를 실행하기 위해 필요한 패키지를 설치합니다.

```bash
# 가상 환경 생성 (선택 사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 필요한 패키지 설치
pip install torch torchvision
pip install opencv-python numpy hydra-core omegaconf
pip install requests tqdm matplotlib

# SAM2 모델 설치 (실제 구현 시 필요)
# pip install git+https://github.com/facebookresearch/segment-anything-2.git

# ByteTrack 설치 (실제 구현 시 필요)
# pip install git+https://github.com/FoundationVision/ByteTrack.git
```

## 2. 프로젝트 구조

```
sam_mot_tracking/
├── configs/                  # Hydra 설정 파일
│   ├── config.yaml           # 기본 설정 파일
│   ├── models/               # 모델 관련 설정
│   │   └── sam2.yaml
│   ├── trackers/             # 추적기 관련 설정
│   │   └── bytetrack.yaml
│   ├── data/                 # 데이터 관련 설정
│   │   └── default.yaml
│   └── visualization/        # 시각화 관련 설정
│       └── default.yaml
├── data/                     # 데이터 저장 디렉토리
│   ├── raw/                  # 원본 영상 데이터
│   └── processed/            # 전처리된 영상 데이터
├── src/                      # 소스 코드
│   ├── models/               # 세그멘테이션 모델
│   │   └── sam2_segmenter.py
│   ├── trackers/             # 추적 알고리즘
│   │   └── bytetrack_tracker.py
│   ├── data/                 # 데이터 처리
│   │   └── prepare_data.py
│   ├── utils/                # 유틸리티 함수
│   ├── visualization/        # 시각화 관련 코드
│   └── sam_mot_pipeline.py   # 통합 파이프라인
├── outputs/                  # 결과물 저장 디렉토리
├── run_pipeline.py           # 파이프라인 실행 스크립트
└── requirements_analysis.md  # 요구사항 분석 문서
```

## 3. 실행 방법

```bash
# 프로젝트 루트 디렉토리에서 실행
python run_pipeline.py
```

## 4. 설정 변경

Hydra 프레임워크를 사용하여 다양한 설정을 쉽게 변경할 수 있습니다.

```bash
# 추적할 객체 수 변경
python run_pipeline.py run.num_objects=10

# 다른 비디오 파일 사용
python run_pipeline.py run.video_path=/path/to/your/video.mp4

# 디스플레이 끄기
python run_pipeline.py run.display=false

# 여러 설정 동시 변경
python run_pipeline.py models=sam2 trackers=bytetrack run.num_objects=5 visualization.trajectory.show=true
```

## 5. 주요 기능

- SAM2 기반 픽셀 수준 세그멘테이션
- ByteTrack 기반 다중 객체 추적
- 세그멘테이션과 추적 결과 통합
- 객체 ID, 마스크, 바운딩 박스, 궤적 시각화
- 결과 영상 저장

## 6. 결과물

실행 결과는 `outputs/` 디렉토리에 저장됩니다. 각 실행마다 타임스탬프가 포함된 새 디렉토리가 생성됩니다.

- `outputs/YYYY-MM-DD_HH-MM-SS/results/sam_mot_results.mp4`: 추적 결과 영상
