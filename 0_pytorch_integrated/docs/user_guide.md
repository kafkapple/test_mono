# PyTorch 통합 실험 프레임워크 - 사용자 가이드

이 문서는 PyTorch 통합 실험 프레임워크의 사용법을 설명합니다. 이 프레임워크는 다양한 딥러닝 모델(AE, VAE, VQ-VAE, CLIP, Flamingo 등)을 실험하고 시각화할 수 있는 환경을 제공합니다.

## 목차

1. [설치 및 요구사항](#설치-및-요구사항)
2. [프로젝트 구조](#프로젝트-구조)
3. [파이썬 스크립트로 실험하기](#파이썬-스크립트로-실험하기)
4. [웹앱으로 실험하기](#웹앱으로-실험하기)
5. [커스텀 모델 및 데이터셋 추가하기](#커스텀-모델-및-데이터셋-추가하기)
6. [결과 시각화 및 분석](#결과-시각화-및-분석)
7. [문제 해결](#문제-해결)

## 설치 및 요구사항

### 요구사항

- Python 3.8 이상
- PyTorch 1.10 이상
- CUDA 지원 GPU (선택 사항, 없어도 CPU로 실행 가능)

### 설치 방법

1. 저장소 클론 또는 다운로드:
   ```bash
   git clone https://github.com/username/pytorch_integrated.git
   cd pytorch_integrated
   ```

2. 필요한 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```

## 프로젝트 구조

프로젝트는 다음과 같은 구조로 구성되어 있습니다:

```
pytorch_integrated/
├── src/                      # 소스 코드
│   ├── models/               # 모델 구현
│   │   ├── autoencoder.py    # 기본 오토인코더
│   │   ├── vae.py            # 변분 오토인코더
│   │   ├── vq_vae.py         # Vector Quantized VAE
│   │   ├── clip.py           # CLIP 모델
│   │   └── flamingo.py       # Flamingo/Perceiver IO 모델
│   ├── datasets/             # 데이터셋 구현
│   │   ├── image_datasets.py # 이미지 데이터셋
│   │   ├── text_datasets.py  # 텍스트 데이터셋
│   │   └── multimodal_datasets.py # 멀티모달 데이터셋
│   ├── utils/                # 유틸리티 함수
│   │   ├── visualization.py  # 시각화 도구
│   │   └── training.py       # 학습 및 평가 도구
│   └── webapp/               # 웹 애플리케이션 (선택적)
│       ├── app.py            # Flask 앱
│       ├── static/           # 정적 파일
│       └── templates/        # HTML 템플릿
├── examples/                 # 예제 스크립트
│   ├── run_vae_mnist.py      # MNIST에 VAE 실험
│   ├── run_vqvae_cifar10.py  # CIFAR-10에 VQ-VAE 실험
│   ├── run_clip_flickr.py    # Flickr에 CLIP 실험
│   └── run_model_comparison.py # 모델 비교 실험
├── configs/                  # 설정 파일 (선택적)
├── results/                  # 실험 결과 저장 디렉토리
├── data/                     # 데이터셋 저장 디렉토리
└── README.md                 # 프로젝트 개요
```

## 파이썬 스크립트로 실험하기

### 기본 예제 실행

예제 스크립트는 `examples/` 디렉토리에 있으며, 다음과 같이 실행할 수 있습니다:

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

각 스크립트는 모델을 학습하고, 결과를 시각화하여 `results/` 디렉토리에 저장합니다.

### 커스텀 실험 작성

자신만의 실험 스크립트를 작성하려면 예제 스크립트를 참고하여 다음과 같이 구성할 수 있습니다:

```python
import os
import sys
import torch

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 프로젝트 모듈 임포트
from src.models.vae import VAE
from src.datasets.image_datasets import get_mnist_dataloaders
from src.utils.training import Trainer, set_seed, get_device

def main():
    # 재현성을 위한 시드 설정
    set_seed(42)
    
    # 장치 설정
    device = get_device()
    
    # 데이터 로더 생성
    train_loader, test_loader = get_mnist_dataloaders(batch_size=128)
    
    # 모델 생성
    model = VAE(input_dim=784, hidden_dim=256, latent_dim=20)
    model = model.to(device)
    
    # 학습 및 평가
    # ...

if __name__ == "__main__":
    main()
```

## 웹앱으로 실험하기

웹 인터페이스를 통해 실험하려면 다음 단계를 따르세요:

1. Flask 웹앱 실행:
   ```bash
   python src/webapp/app.py
   ```

2. 웹 브라우저에서 `http://localhost:5000` 접속

3. 웹 인터페이스를 통해 모델, 데이터셋 선택 및 실험 수행

웹앱은 다음과 같은 기능을 제공합니다:
- 다양한 모델 및 데이터셋 선택
- 모델 학습 및 평가
- 결과 시각화
- 파라미터 조정

## 커스텀 모델 및 데이터셋 추가하기

### 새 모델 추가

새로운 모델을 추가하려면 `src/models/` 디렉토리에 새 파일을 생성하고 모델 클래스를 구현하세요:

```python
import torch
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # 모델 레이어 정의
        # ...
    
    def forward(self, x):
        # 순전파 구현
        # ...
        return output
```

### 새 데이터셋 추가

새로운 데이터셋을 추가하려면 `src/datasets/` 디렉토리에 구현하세요:

```python
from torch.utils.data import Dataset, DataLoader

class MyCustomDataset(Dataset):
    def __init__(self, root, transform=None):
        # 데이터셋 초기화
        # ...
    
    def __len__(self):
        # 데이터셋 크기 반환
        # ...
    
    def __getitem__(self, idx):
        # 인덱스에 해당하는 샘플 반환
        # ...
```

## 결과 시각화 및 분석

실험 결과는 `results/` 디렉토리에 저장되며, 다음과 같은 시각화를 포함합니다:

- 학습 곡선 (손실, 정확도)
- 원본 이미지와 재구성 이미지 비교
- 잠재 공간 시각화 (t-SNE, PCA, UMAP)
- 생성된 샘플
- 임베딩 클러스터링

시각화 유틸리티는 `src/utils/visualization.py`에 구현되어 있으며, 다음과 같이 사용할 수 있습니다:

```python
from src.utils.visualization import plot_reconstruction, plot_embeddings, save_visualization

# 재구성 시각화
recon_img = plot_reconstruction(original, reconstruction)
save_visualization(recon_img, "path/to/save/reconstruction.png")

# 임베딩 시각화
emb_img = plot_embeddings(embeddings, labels, method='tsne')
save_visualization(emb_img, "path/to/save/embeddings.png")
```

## 문제 해결

### 일반적인 문제

1. **CUDA 메모리 부족**: 배치 크기를 줄이거나 더 작은 모델을 사용하세요.
2. **학습이 불안정함**: 학습률을 낮추거나 그래디언트 클리핑을 적용하세요.
3. **데이터셋 로드 오류**: 데이터 경로가 올바른지 확인하세요.

### 지원 및 문의

문제가 발생하거나 질문이 있으면 GitHub 이슈를 통해 문의하세요.
