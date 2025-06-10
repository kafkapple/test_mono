"""
MNIST에 VAE 모델을 학습하고 시각화하는 예제 스크립트
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
from src.models.vae import VAE
from src.datasets.image_datasets import get_mnist_dataloaders
from src.utils.visualization import plot_reconstruction, plot_latent_space_2d, plot_generated_samples, save_visualization
from src.utils.training import Trainer, set_seed, get_device
from src.utils.config import (
    get_model_config, 
    create_model_from_config, 
    get_optimizer_from_config, 
    get_scheduler_from_config,
    load_config
)

def main():
    # 재현성을 위한 시드 설정
    set_seed(42)
    
    # 장치 설정
    device = get_device()
    print(f"Using device: {device}")
    
    # 설정 파일 로드
    model_config = get_model_config('vae')
    train_config = load_config('configs/training/train.yaml')
    default_config = load_config('configs/experiment/default.yaml')
    
    # 설정 병합
    config = {
        'model': model_config,
        'training': train_config['training'],
        'experiment': default_config['experiment']
    }
    
    print("Loaded configurations:")
    print(f"Model type: {config['model']['type']}")
    print(f"Architecture: {config['model']['architecture']}")
    print(f"Training settings: {config['training']}")
    
    # 결과 저장 디렉토리 생성
    results_dir = config['model']['save']['dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # 데이터 로더 생성
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=config['training']['batch_size']
    )
    
    # 데이터 변환 함수 정의
    def flatten_batch(batch):
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            data, target = batch[0], batch[1]
            data = data.view(data.size(0), -1)  # 평탄화
            return (data, target)
        else:
            data = batch
            data = data.view(data.size(0), -1)  # 평탄화
            return data
    
    # 데이터 로더 래퍼 클래스 정의
    class FlatteningDataLoader:
        """데이터를 평탄화하는 DataLoader 래퍼"""
        def __init__(self, dataloader):
            self.dataloader = dataloader
            self.dataset = dataloader.dataset  # 원본 데이터셋 참조
            self.batch_size = dataloader.batch_size
            self.num_workers = dataloader.num_workers
            self.pin_memory = dataloader.pin_memory
            self.drop_last = dataloader.drop_last
            self.timeout = dataloader.timeout
            self.sampler = dataloader.sampler
            self.prefetch_factor = dataloader.prefetch_factor
            self.persistent_workers = dataloader.persistent_workers
            
        def __iter__(self):
            for batch in self.dataloader:
                yield flatten_batch(batch)
                
        def __len__(self):
            return len(self.dataloader)
            
        def __getattr__(self, name):
            """다른 속성은 원본 DataLoader에서 가져옴"""
            return getattr(self.dataloader, name)
    
    # 데이터 로더 래핑
    train_loader = FlatteningDataLoader(train_loader)
    test_loader = FlatteningDataLoader(test_loader)
    
    # 모델 생성
    print(f"Creating VAE model with latent dimension {config['model']['architecture']['latent_dim']}...")
    model = create_model_from_config(config['model'])
    model = model.to(device)
    
    # 옵티마이저 생성
    optimizer = get_optimizer_from_config(model, config)
    
    # 학습률 스케줄러 생성
    scheduler = get_scheduler_from_config(optimizer, config)
    
    # 트레이너 생성
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        device=device,
        save_dir=results_dir,
        log_interval=config['training']['logging']['log_interval']
    )
    
    # 모델 학습
    print(f"Training VAE for {config['training']['epochs']} epochs...")
    history = trainer.train(
        epochs=config['training']['epochs'],
        save_best=config['model']['save']['save_best'],
        early_stopping=config['training']['early_stopping']
    )
    
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
    
    # 잠재 공간에서 샘플 생성
    print("Generating samples from latent space...")
    with torch.no_grad():
        samples = model.sample(num_samples=25, device=device)
    
    # 생성된 샘플 시각화
    samples_img = plot_generated_samples(
        samples=samples.view(-1, 1, 28, 28),
        nrow=5,
        ncol=5,
        figsize=(10, 10),
        title="Generated Samples"
    )
    save_visualization(samples_img, os.path.join(results_dir, "generated_samples.png"))
    
    # 2D 잠재 공간 시각화 (잠재 차원이 2인 경우에만)
    if config['model']['architecture']['latent_dim'] == 2:
        print("Visualizing 2D latent space...")
        config_2d = config.copy()
        config_2d['model']['architecture']['latent_dim'] = 2
        model_2d = create_model_from_config(config_2d['model'])
        model_2d = model_2d.to(device)
        
        # 2D 모델 학습
        optimizer_2d = get_optimizer_from_config(model_2d, config_2d)
        trainer_2d = Trainer(
            model=model_2d,
            train_loader=train_loader,
            val_loader=test_loader,
            optimizer=optimizer_2d,
            device=device,
            save_dir=os.path.join(results_dir, "2d_model")
        )
        
        trainer_2d.train(
            epochs=config['training']['epochs'] // 2,  # 2D 모델은 더 적은 에포크로 학습
            save_best=config['model']['save']['save_best'],
            early_stopping=config['training']['early_stopping']
        )
        
        # 2D 잠재 공간 시각화
        latent_img = plot_latent_space_2d(
            model=model_2d,
            n_samples=20,
            figsize=(12, 12)
        )
        save_visualization(latent_img, os.path.join(results_dir, "latent_space_2d.png"))
    
    print(f"All results saved to {results_dir}")
    print("Done!")

if __name__ == "__main__":
    main()
