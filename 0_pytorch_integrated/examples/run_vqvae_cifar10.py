"""
CIFAR-10에 VQ-VAE 모델을 학습하고 시각화하는 예제 스크립트
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 프로젝트 모듈 임포트
from src.models.vq_vae import ConvVQVAE, train_vqvae
from src.datasets.image_datasets import get_cifar10_dataloaders
from src.utils.visualization import plot_reconstruction, plot_generated_samples, save_visualization
from src.utils.training import Trainer, set_seed, get_device

def main():
    # 재현성을 위한 시드 설정
    set_seed(42)
    
    # 장치 설정
    device = get_device()
    print(f"Using device: {device}")
    
    # 결과 저장 디렉토리 생성
    results_dir = os.path.join("results", "vqvae_cifar10")
    os.makedirs(results_dir, exist_ok=True)
    
    # 데이터 로더 생성
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=128)
    
    # 모델 생성
    in_channels = 3  # CIFAR-10 이미지 채널 수
    hidden_dim = 128
    latent_dim = 64
    num_embeddings = 512
    
    print(f"Creating VQ-VAE model with {num_embeddings} codebook entries...")
    model = ConvVQVAE(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings
    )
    model = model.to(device)
    
    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 학습 설정
    epochs = 20
    
    # 트레이너 생성
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        device=device,
        save_dir=results_dir
    )
    
    # 모델 학습
    print(f"Training VQ-VAE for {epochs} epochs...")
    history = trainer.train(epochs=epochs, save_best=True)
    
    # 학습 곡선 시각화
    print("Plotting training curves...")
    trainer.plot_history(save_path=os.path.join(results_dir, "training_curves.png"))
    
    # 테스트 데이터에서 샘플 가져오기
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    images = images.to(device)
    
    # 이미지 재구성
    print("Generating reconstructions...")
    with torch.no_grad():
        reconstructions, _, _ = model(images)
    
    # 재구성 시각화
    recon_img = plot_reconstruction(
        original=images[:10],
        reconstruction=reconstructions[:10],
        n_samples=10,
        figsize=(12, 6)
    )
    save_visualization(recon_img, os.path.join(results_dir, "reconstructions.png"))
    
    # 코드북 인덱스 시각화
    print("Visualizing codebook usage...")
    
    # 테스트 데이터에서 코드북 인덱스 수집
    indices_list = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            if batch_idx >= 10:  # 10개 배치만 사용
                break
            data = data.to(device)
            _, _, indices = model(data)
            indices_list.append(indices.cpu().numpy())
            
    all_indices = np.concatenate(indices_list)
    
    # 코드북 사용 빈도 계산 및 시각화
    plt.figure(figsize=(10, 5))
    plt.hist(all_indices, bins=num_embeddings, alpha=0.7)
    plt.xlabel('Codebook Index')
    plt.ylabel('Frequency')
    plt.title('VQ-VAE Codebook Usage')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "codebook_usage.png"), dpi=300)
    plt.close()
    
    # 잠재 공간에서 샘플 생성 (코드북 인덱스 샘플링)
    print("Generating samples from codebook...")
    
    # 무작위 코드북 인덱스 생성
    num_samples = 25
    random_indices = torch.randint(0, num_embeddings, (num_samples, 1), device=device)
    
    # 인덱스에서 이미지 생성
    with torch.no_grad():
        # 임베딩 가져오기
        embeddings = model.vq.embedding(random_indices).squeeze(1)
        
        # 임베딩 형태 변환 (모델 구현에 따라 조정 필요)
        batch_size = embeddings.size(0)
        spatial_size = 8  # 32x32 이미지의 경우 4번 다운샘플링하면 8x8
        embeddings = embeddings.view(batch_size, latent_dim, 1, 1).expand(batch_size, latent_dim, spatial_size, spatial_size)
        
        # 디코딩
        samples = model.decoder(embeddings)
    
    # 생성된 샘플 시각화
    samples_img = plot_generated_samples(
        samples=samples,
        nrow=5,
        ncol=5,
        figsize=(10, 10),
        title="Generated Samples from Codebook"
    )
    save_visualization(samples_img, os.path.join(results_dir, "generated_samples.png"))
    
    print(f"All results saved to {results_dir}")
    print("Done!")

if __name__ == "__main__":
    main()
