# PyTorch 통합 실험 프레임워크

PyTorch를 활용한 딥러닝 모델 실험 및 시각화를 위한 통합 프레임워크입니다. 다양한 모델 아키텍처(AE, VAE, VQ-VAE, CLIP, Flamingo 등)와 데이터셋(이미지, 텍스트, 멀티모달)을 지원하며, 파이썬 스크립트 또는 웹 인터페이스를 통해 실험할 수 있습니다.

## 주요 기능

- **다양한 모델 아키텍처**: AutoEncoder, VAE, VQ-VAE, CLIP, Flamingo(Perceiver IO) 등
- **다양한 데이터셋 지원**: 이미지(MNIST, CIFAR-10), 텍스트(IMDB), 멀티모달(Flickr)
- **시각화 도구**: 임베딩 시각화(t-SNE, PCA, UMAP), 재구성 비교, 생성 샘플 등
- **실험 자동화**: 모델 학습, 평가, 비교 실험 자동화
- **웹 인터페이스**: 브라우저를 통한 실험 및 시각화 (선택적)

## 시작하기

### 요구사항

- Python 3.8 이상
- PyTorch 1.10 이상
- 기타 필요 패키지: numpy, matplotlib, scikit-learn, tqdm, flask 등

### 설치

```bash
# 저장소 클론 또는 다운로드
git clone https://github.com/username/pytorch_integrated.git
cd pytorch_integrated

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 파이썬 스크립트로 실행

```bash
# MNIST에 VAE 모델 학습 및 시각화
python examples/run_vae_mnist.py

# CIFAR-10에 VQ-VAE 모델 학습 및 시각화
python examples/run_vqvae_cifar10.py

# Flickr에 CLIP 모델 학습 및 시각화
python examples/run_clip_flickr.py

# 여러 모델 비교 실험
python examples/run_model_comparison.py
```

### 웹앱으로 실행

```bash
# Flask 웹앱 실행
python src/webapp/app.py

# 브라우저에서 http://localhost:5000 접속
```

## 프로젝트 구조

```
pytorch_integrated/
├── src/                      # 소스 코드
│   ├── models/               # 모델 구현
│   ├── datasets/             # 데이터셋 구현
│   ├── utils/                # 유틸리티 함수
│   └── webapp/               # 웹 애플리케이션 (선택적)
├── examples/                 # 예제 스크립트
├── configs/                  # 설정 파일 (선택적)
├── results/                  # 실험 결과 저장 디렉토리
├── data/                     # 데이터셋 저장 디렉토리
└── docs/                     # 문서
```

## 문서

자세한 사용법은 다음 문서를 참조하세요:

- [사용자 가이드](docs/user_guide.md): 상세한 사용법 및 예제
- [API 문서](docs/api_reference.md): 모듈 및 함수 레퍼런스
- [개발자 가이드](docs/developer_guide.md): 프로젝트 확장 및 기여 방법

## 예제 결과

### VAE 모델 - MNIST 데이터셋

VAE 모델을 MNIST 데이터셋에 학습한 결과입니다:

- 학습 곡선
- 원본 이미지와 재구성 이미지 비교
- 잠재 공간에서 생성된 샘플
- 2D 잠재 공간 시각화

### VQ-VAE 모델 - CIFAR-10 데이터셋

VQ-VAE 모델을 CIFAR-10 데이터셋에 학습한 결과입니다:

- 학습 곡선
- 원본 이미지와 재구성 이미지 비교
- 코드북 사용 빈도 분석
- 코드북에서 생성된 샘플

### CLIP 모델 - Flickr 데이터셋

CLIP 모델을 Flickr 데이터셋에 학습한 결과입니다:

- 학습 곡선
- 이미지 임베딩 시각화
- 텍스트 임베딩 시각화
- 이미지-텍스트 검색 예시

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 기여

기여는 언제나 환영합니다! 버그 리포트, 기능 요청, 풀 리퀘스트 등을 통해 프로젝트 개선에 참여해주세요.
