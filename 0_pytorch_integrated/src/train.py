"""
메인 학습 스크립트 - Hydra 설정 기반 모델 학습

이 스크립트는 configs/config.yaml 및 관련 설정 파일을 사용하여
다양한 모델과 데이터셋에 대한 학습을 실행합니다.
"""
import os
import sys
import logging
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 프로젝트 루트 경로 추가 (src 모듈 임포트 문제 해결)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 프로젝트 모듈 임포트
from src.models.autoencoder import AutoEncoder
from src.models.vae import VAE
from src.models.vq_vae import ConvVQVAE
from src.models.clip import CLIP
from src.models.flamingo import Flamingo
from src.datasets.image_datasets import get_mnist_dataloaders, get_cifar10_dataloaders
from src.datasets.text_datasets import get_imdb_dataloaders
from src.datasets.multimodal_datasets import FlickrDataset, get_multimodal_dataloader
from src.utils.visualization import plot_reconstruction, plot_embeddings, save_visualization
from src.utils.training import Trainer, set_seed, get_device

# 로깅 설정
log = logging.getLogger(__name__)

def get_model(cfg: DictConfig, device: torch.device):
    """설정에 따라 모델 인스턴스를 생성합니다."""
    model_type = cfg.model.type
    
    if model_type == "autoencoder":
        model = AutoEncoder(
            input_dim=784 if cfg.dataset.name == "mnist" else 3072,
            hidden_dim=cfg.model.hidden_dim,
            latent_dim=cfg.model.latent_dim
        )
    elif model_type == "vae":
        model = VAE(
            input_dim=784 if cfg.dataset.name == "mnist" else 3072,
            hidden_dim=cfg.model.hidden_dim,
            latent_dim=cfg.model.latent_dim
        )
    elif model_type == "vqvae":
        model = ConvVQVAE(
            in_channels=1 if cfg.dataset.name == "mnist" else 3,
            hidden_dim=cfg.model.hidden_dim,
            latent_dim=cfg.model.latent_dim,
            num_embeddings=cfg.model.get("num_embeddings", 512)
        )
    elif model_type == "clip":
        model = CLIP(
            vocab_size=cfg.model.get("vocab_size", 10000),
            embedding_dim=cfg.model.hidden_dim,
            output_dim=cfg.model.latent_dim
        )
    elif model_type == "flamingo":
        model = Flamingo(
            vocab_size=cfg.model.get("vocab_size", 10000),
            embedding_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.get("num_layers", 4),
            num_heads=cfg.model.get("num_heads", 8),
            output_dim=cfg.model.latent_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)

def get_dataloaders(cfg: DictConfig):
    """설정에 따라 데이터로더를 생성합니다."""
    dataset_name = cfg.dataset.name
    batch_size = cfg.training.batch_size
    
    if dataset_name == "mnist":
        return get_mnist_dataloaders(batch_size=batch_size, root=cfg.dataset.root)
    elif dataset_name == "cifar10":
        return get_cifar10_dataloaders(batch_size=batch_size, root=cfg.dataset.root)
    elif dataset_name == "imdb":
        return get_imdb_dataloaders(batch_size=batch_size, root=cfg.dataset.root)
    elif dataset_name == "flickr":
        # 더미 데이터셋 사용 (실제 데이터셋이 없는 경우)
        from src.datasets.multimodal_datasets import create_dummy_flickr_dataset
        dummy_dir = os.path.join(cfg.dataset.root, "dummy_flickr")
        dataset_dir, captions_file = create_dummy_flickr_dataset(dummy_dir, num_samples=500)
        
        from torchvision import transforms
        from torchtext.data.utils import get_tokenizer
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        tokenizer = get_tokenizer("basic_english")
        
        dataset = FlickrDataset(
            root_dir=dataset_dir,
            captions_file=captions_file,
            transform=transform,
            tokenizer=tokenizer,
            max_length=77
        )
        
        # 데이터셋 분할
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # 데이터 로더 생성
        train_loader = get_multimodal_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = get_multimodal_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_optimizer(cfg: DictConfig, model: nn.Module):
    """설정에 따라 옵티마이저를 생성합니다."""
    optimizer_name = cfg.training.optimizer.lower()
    lr = cfg.training.learning_rate
    weight_decay = cfg.training.weight_decay
    
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """메인 학습 함수"""
    # 설정 출력
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # 재현성을 위한 시드 설정
    set_seed(cfg.experiment.seed)
    
    # 장치 설정
    device = get_device() if cfg.run.device == "auto" else torch.device(cfg.run.device)
    log.info(f"Using device: {device}")
    
    # 결과 저장 디렉토리 생성
    os.makedirs(cfg.logging.save_dir, exist_ok=True)
    
    # 데이터 로더 생성
    log.info(f"Loading {cfg.dataset.name} dataset...")
    train_loader, val_loader = get_dataloaders(cfg)
    
    # 모델 생성
    log.info(f"Creating {cfg.model.type} model...")
    model = get_model(cfg, device)
    
    # 옵티마이저 설정
    optimizer = get_optimizer(cfg, model)
    
    # 트레이너 생성
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_dir=cfg.logging.save_dir
    )
    
    # 모델 학습
    if cfg.run.mode == "train":
        log.info(f"Training {cfg.model.type} for {cfg.training.epochs} epochs...")
        history = trainer.train(
            epochs=cfg.training.epochs,
            save_best=cfg.logging.save_best_only
        )
        
        # 학습 곡선 시각화
        trainer.plot_history(save_path=os.path.join(cfg.logging.save_dir, "training_curves.png"))
        
        # 테스트 데이터에서 샘플 가져오기
        dataiter = iter(val_loader)
        batch = next(dataiter)
        
        # 이미지 데이터셋인 경우 재구성 시각화
        if cfg.dataset.name in ["mnist", "cifar10"]:
            images, _ = batch
            images = images.to(device)
            
            # 이미지 재구성
            model.eval()
            with torch.no_grad():
                if cfg.model.type == 'vae':
                    reconstructions, _, _ = model(images)
                elif cfg.model.type == 'vqvae':
                    reconstructions, _, _ = model(images)
                else:
                    reconstructions = model(images)
            
            # 재구성 시각화
            recon_img = plot_reconstruction(
                original=images[:10],
                reconstruction=reconstructions[:10],
                n_samples=10,
                figsize=(12, 6)
            )
            save_visualization(recon_img, os.path.join(cfg.logging.save_dir, "reconstructions.png"))
        
        log.info(f"Training completed. Results saved to {cfg.logging.save_dir}")
    
    # 모델 테스트
    elif cfg.run.mode == "test":
        log.info("Testing model...")
        # 모델 로드
        model_path = os.path.join(cfg.logging.save_dir, "best_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            log.info(f"Loaded model from {model_path}")
        
        # 테스트 평가
        test_loss = trainer.evaluate(val_loader)
        log.info(f"Test loss: {test_loss:.6f}")
    
    # 임베딩 시각화
    elif cfg.run.mode == "visualize":
        log.info("Visualizing embeddings...")
        # 모델 로드
        model_path = os.path.join(cfg.logging.save_dir, "best_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            log.info(f"Loaded model from {model_path}")
        
        # 임베딩 추출 및 시각화
        from src.utils.visualization import extract_embeddings
        
        embeddings = extract_embeddings(model, val_loader, device=device)
        
        if isinstance(embeddings, tuple) and len(embeddings) == 2:
            embeddings, labels = embeddings
        else:
            labels = None
        
        # t-SNE 시각화
        tsne_img = plot_embeddings(
            embeddings=embeddings,
            labels=labels,
            method='tsne',
            figsize=(10, 8),
            title=f"{cfg.model.type} Embeddings (t-SNE)"
        )
        save_visualization(tsne_img, os.path.join(cfg.logging.save_dir, "embeddings_tsne.png"))
        
        # PCA 시각화
        pca_img = plot_embeddings(
            embeddings=embeddings,
            labels=labels,
            method='pca',
            figsize=(10, 8),
            title=f"{cfg.model.type} Embeddings (PCA)"
        )
        save_visualization(pca_img, os.path.join(cfg.logging.save_dir, "embeddings_pca.png"))
        
        log.info(f"Visualization completed. Results saved to {cfg.logging.save_dir}")
    
    else:
        raise ValueError(f"Unknown run mode: {cfg.run.mode}")

if __name__ == "__main__":
    main()
