# PyTorch 통합 실험 프레임워크 - 폴더 구조 설계

```
pytorch_integrated/
├── src/                      # 소스 코드 루트 폴더
│   ├── datasets/             # 데이터셋 모듈
│   │   ├── __init__.py
│   │   ├── image_datasets.py     # 이미지 데이터셋 구현
│   │   ├── text_datasets.py      # 텍스트 데이터셋 구현
│   │   └── multimodal_datasets.py # 멀티모달 데이터셋 구현
│   │
│   ├── models/               # 모델 모듈
│   │   ├── __init__.py
│   │   ├── autoencoder.py        # 기본 오토인코더 구현
│   │   ├── vae.py                # 변분 오토인코더 구현
│   │   ├── vq_vae.py             # Vector Quantized VAE 구현
│   │   ├── clip.py               # CLIP 모델 구현
│   │   └── flamingo.py           # Flamingo/Perceiver IO 구현
│   │
│   ├── utils/                # 유틸리티 모듈
│   │   ├── __init__.py
│   │   ├── training.py           # 학습 유틸리티
│   │   ├── inference.py          # 추론 유틸리티
│   │   ├── visualization.py      # 시각화 유틸리티
│   │   ├── embedding.py          # 임베딩 추출 및 시각화
│   │   └── metrics.py            # 평가 지표 계산
│   │
│   ├── webapp/               # 웹앱 모듈 (독립적으로 실행 가능)
│   │   ├── __init__.py
│   │   ├── main.py               # Flask 앱 진입점
│   │   ├── routes/               # Flask 라우트
│   │   │   ├── __init__.py
│   │   │   └── pytorch_examples.py # 예제 라우트
│   │   ├── static/               # 정적 파일
│   │   │   ├── css/              # CSS 스타일시트
│   │   │   │   └── main.css
│   │   │   └── js/               # JavaScript 파일
│   │   │       └── main.js
│   │   └── templates/            # HTML 템플릿
│   │       ├── base.html         # 기본 템플릿
│   │       ├── index.html        # 메인 페이지
│   │       ├── example_page.html # 예제 페이지
│   │       └── playground.html   # 코드 플레이그라운드
│   │
│   └── __init__.py           # 패키지 초기화
│
├── examples/                 # 예제 스크립트 폴더
│   ├── run_vae_mnist.py          # MNIST에 VAE 실행 예제
│   ├── run_vqvae_cifar10.py      # CIFAR-10에 VQ-VAE 실행 예제
│   ├── run_clip_flickr.py        # Flickr에 CLIP 실행 예제
│   └── run_model_comparison.py   # 모델 비교 예제
│
├── scripts/                  # 유틸리티 스크립트 폴더
│   ├── install_dependencies.sh   # 의존성 설치 스크립트
│   ├── run_webapp.sh             # 웹앱 실행 스크립트
│   └── run_tests.sh              # 테스트 실행 스크립트
│
├── tests/                    # 테스트 폴더
│   ├── __init__.py
│   ├── test_models.py            # 모델 테스트
│   ├── test_datasets.py          # 데이터셋 테스트
│   └── test_utils.py             # 유틸리티 테스트
│
├── configs/                  # 설정 파일 폴더
│   ├── default.yaml              # 기본 설정
│   ├── models/                   # 모델별 설정
│   │   ├── vae.yaml
│   │   ├── vq_vae.yaml
│   │   └── clip.yaml
│   └── datasets/                 # 데이터셋별 설정
│       ├── mnist.yaml
│       ├── cifar10.yaml
│       └── flickr.yaml
│
├── docs/                     # 문서 폴더
│   ├── user_guide.md             # 사용자 가이드
│   ├── model_guide.md            # 모델 가이드
│   ├── webapp_guide.md           # 웹앱 가이드
│   └── api_reference.md          # API 참조 문서
│
├── results/                  # 결과 저장 폴더 (자동 생성)
│   ├── models/                   # 저장된 모델 파일
│   ├── figures/                  # 생성된 그림
│   └── logs/                     # 로그 파일
│
├── requirements.txt          # 의존성 목록
├── setup.py                  # 설치 스크립트
├── README.md                 # 프로젝트 README
└── LICENSE                   # 라이선스 파일
```

## 주요 특징

1. **모듈화된 구조**: 모델, 데이터셋, 유틸리티가 명확히 분리되어 있어 코드 관리와 이해가 용이함
2. **독립적인 웹앱**: `src/webapp` 폴더에 웹앱 모듈이 독립적으로 구성되어 있어, 필요에 따라 사용 가능
3. **직접 실행 가능한 예제**: `examples/` 폴더에 파이썬 스크립트로 직접 실행 가능한 예제 제공
4. **설정 관리**: `configs/` 폴더에 YAML 형식의 설정 파일을 통해 실험 설정 관리
5. **결과 시각화**: `utils/visualization.py`를 통해 웹앱 없이도 결과 시각화 가능
6. **문서화**: `docs/` 폴더에 상세한 사용자 가이드와 API 참조 문서 제공

## 실행 방법

1. **파이썬 스크립트로 직접 실행**:
   ```bash
   python examples/run_vae_mnist.py
   ```

2. **웹앱 실행**:
   ```bash
   python -m src.webapp.main
   ```

3. **스크립트 사용**:
   ```bash
   ./scripts/run_webapp.sh
   ```

## 확장 방법

1. **새 모델 추가**: `src/models/` 폴더에 새 모델 파일 추가
2. **새 데이터셋 추가**: `src/datasets/` 폴더에 새 데이터셋 파일 추가
3. **새 예제 추가**: `examples/` 폴더에 새 예제 스크립트 추가
