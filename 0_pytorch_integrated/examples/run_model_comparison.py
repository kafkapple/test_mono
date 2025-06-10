"""
모델 비교 실험 스크립트 - 여러 모델 아키텍처를 비교하는 예제
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import time
import json

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 프로젝트 모듈 임포트
from src.models.autoencoder import AutoEncoder
from src.models.vae import VAE
from src.models.vq_vae import ConvVQVAE
from src.datasets.image_datasets import get_mnist_dataloaders
from src.utils.visualization import plot_reconstruction, plot_embeddings, save_visualization
from src.utils.training import Trainer, set_seed, get_device, count_parameters

def main():
    # 재현성을 위한 시드 설정
    set_seed(42)
    
    # 장치 설정
    device = get_device()
    print(f"Using device: {device}")
    
    # 결과 저장 디렉토리 생성
    results_dir = os.path.join("results", "model_comparison")
    os.makedirs(results_dir, exist_ok=True)
    
    # 데이터 로더 생성
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_dataloaders(batch_size=128)
    
    # 데이터 형태 확인
    print("\n데이터 로더 출력 형태 확인:")
    print("-" * 50)
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx == 0:  # 첫 번째 배치만 확인
            print(f"데이터 형태: {data.shape}")
            print(f"데이터 범위: [{data.min():.3f}, {data.max():.3f}]")
            print(f"데이터 평균: {data.mean():.3f}")
            print(f"데이터 표준편차: {data.std():.3f}")
            break
    print("-" * 50)
    
    # 모델 설정
    input_dim = 28 * 28  # MNIST 이미지 크기
    hidden_dim = 256
    latent_dim = 20
    
    # 모델 생성
    models = {
        'AutoEncoder': AutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim),
        'VAE': VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim),
        'ConvVQVAE': ConvVQVAE(in_channels=1, hidden_dim=hidden_dim//4, latent_dim=latent_dim, num_embeddings=512)
    }
    
    # 모델별 데이터 변환 함수 정의
    def transform_data(data, model_name):
        if model_name in ['AutoEncoder', 'VAE']:
            # 평탄화된 형태로 변환 (B, C*H*W)
            if len(data.shape) == 4:  # (B, C, H, W) 형태인 경우
                return data.view(data.size(0), -1)
            return data
        else:  # ConvVQVAE
            # 이미지 형태로 변환 (B, C, H, W)
            if len(data.shape) == 3:  # (B, C, H*W) 형태인 경우
                return data.view(data.size(0), data.size(1), 28, 28)
            elif len(data.shape) == 2:  # (B, C*H*W) 형태인 경우
                return data.view(data.size(0), 1, 28, 28)
            return data  # 이미 올바른 형태인 경우 그대로 반환
    
    # 모델 정보 출력
    print("\n모델 아키텍처 비교:")
    print("-" * 50)
    for name, model in models.items():
        num_params = count_parameters(model)
        print(f"{name}: {num_params:,} 파라미터")
    print("-" * 50)
    
    # 학습 설정
    epochs = 10
    results = {}
    
    # 각 모델 학습 및 평가
    for name, model in models.items():
        print(f"\n{name} 모델 학습 시작...")
        model = model.to(device)
        
        # 모델별 결과 디렉토리 생성
        model_dir = os.path.join(results_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 옵티마이저 설정
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # 트레이너 생성
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            optimizer=optimizer,
            device=device,
            save_dir=model_dir,
            transform_fn=lambda x: transform_data(x, name)  # 데이터 변환 함수 추가
        )
        
        # 데이터 변환 디버깅
        print(f"\n{name} 모델 데이터 변환 디버깅:")
        print("-" * 50)
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == 0:  # 첫 번째 배치만 확인
                print(f"변환 전 데이터 형태: {data.shape}")
                transformed_data = transform_data(data, name)
                print(f"변환 후 데이터 형태: {transformed_data.shape}")
                print(f"변환 후 데이터 범위: [{transformed_data.min():.3f}, {transformed_data.max():.3f}]")
                break
        print("-" * 50)
        
        # 학습 시간 측정
        start_time = time.time()
        
        # 모델 학습
        history = trainer.train(epochs=epochs, save_best=True)
        
        # 학습 시간 계산
        train_time = time.time() - start_time
        
        # 학습 곡선 시각화
        trainer.plot_history(save_path=os.path.join(model_dir, "training_curves.png"))
        
        # 테스트 데이터에서 샘플 가져오기
        dataiter = iter(test_loader)
        images, _ = next(dataiter)
        images = images.to(device)
        
        # 이미지 재구성
        model.eval()
        with torch.no_grad():
            if name == 'VAE':
                reconstructions, _, _ = model(images)
                # 평탄화된 이미지를 2D 형태로 변환 (B, 1, 28, 28)
                reconstructions = reconstructions.view(-1, 1, 28, 28)
                images = images.view(-1, 1, 28, 28)
            elif name == 'ConvVQVAE':
                reconstructions, _, _ = model(images)
                # ConvVQVAE는 이미 올바른 형태 (B, C, H, W)로 출력
            else:  # AutoEncoder
                reconstructions, _ = model(images)  # 튜플에서 첫 번째 요소만 사용
                # 평탄화된 이미지를 2D 형태로 변환 (B, 1, 28, 28)
                reconstructions = reconstructions.view(-1, 1, 28, 28)
                images = images.view(-1, 1, 28, 28)
        
        # 재구성 시각화
        recon_img = plot_reconstruction(
            original=images[:10],
            reconstruction=reconstructions[:10],
            n_samples=10,
            figsize=(12, 6)
        )
        save_visualization(recon_img, os.path.join(model_dir, "reconstructions.png"))
        
        # 임베딩 추출 및 시각화
        if name != 'ConvVQVAE':  # ConvVQVAE는 임베딩 추출 방식이 다름
            # 임베딩 추출
            embeddings = []
            labels = []
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):
                    if batch_idx >= 10:  # 10개 배치만 사용
                        break
                    
                    data = data.to(device)
                    
                    if name == 'VAE':
                        mu, _ = model.encode(data.view(-1, input_dim))
                        embedding = mu
                    else:
                        embedding = model.encode(data.view(-1, input_dim))
                    
                    embeddings.append(embedding.cpu().numpy())
                    labels.append(target.numpy())
            
            # 임베딩 결합
            embeddings = np.vstack(embeddings)
            labels = np.concatenate(labels)
            
            # 임베딩 시각화 (t-SNE)
            emb_img_tsne = plot_embeddings(
                embeddings=embeddings,
                labels=labels,
                method='tsne',
                figsize=(10, 8),
                title=f"{name} Embeddings (t-SNE)"
            )
            save_visualization(emb_img_tsne, os.path.join(model_dir, "embeddings_tsne.png"))
            
            # 임베딩 시각화 (PCA)
            emb_img_pca = plot_embeddings(
                embeddings=embeddings,
                labels=labels,
                method='pca',
                figsize=(10, 8),
                title=f"{name} Embeddings (PCA)"
            )
            save_visualization(emb_img_pca, os.path.join(model_dir, "embeddings_pca.png"))
        
        # 결과 저장
        results[name] = {
            'train_loss': history['train_loss'][-1],
            'val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'train_time': train_time,
            'parameters': count_parameters(model)
        }
    
    # 결과 비교 표 생성
    print("\n모델 성능 비교:")
    print("-" * 80)
    print(f"{'모델':<15} {'파라미터 수':<15} {'학습 시간(초)':<15} {'최종 학습 손실':<20} {'최종 검증 손실':<20}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<15} {result['parameters']:<15,} {result['train_time']:<15.2f} {result['train_loss']:<20.6f} {result['val_loss'] if result['val_loss'] else 'N/A':<20}")
    
    print("-" * 80)
    
    # 결과를 JSON으로 저장
    with open(os.path.join(results_dir, "comparison_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 손실 비교 그래프 생성
    plt.figure(figsize=(12, 6))
    
    for name, model in models.items():
        model_dir = os.path.join(results_dir, name)
        
        # 학습 기록 로드
        with open(os.path.join(model_dir, "training_history.json"), 'r') as f:
            history = json.load(f)
        
        # 학습 손실 그래프
        plt.plot(history['train_loss'], label=f"{name} (Train)")
        
        # 검증 손실 그래프 (있는 경우)
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], linestyle='--', label=f"{name} (Val)")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('모델 간 손실 비교')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, "loss_comparison.png"), dpi=300, bbox_inches='tight')
    
    print(f"\n모든 결과가 {results_dir}에 저장되었습니다.")
    print("완료!")

if __name__ == "__main__":
    main()
