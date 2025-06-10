"""
이미지 데이터셋 모듈
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from src.utils.config import get_dataset_config, create_transform_from_config, create_dataloader_from_config

class MNISTDataset(Dataset):
    """
    MNIST 데이터셋 래퍼
    """
    def __init__(self, root='./data', train=True, transform=None, download=True):
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]) if transform is None else transform,
            download=download
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 데이터셋 래퍼
    """
    def __init__(self, root='./data', train=True, transform=None, download=True):
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]) if transform is None else transform,
            download=download
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class CustomImageDataset(Dataset):
    """
    사용자 정의 이미지 데이터셋
    지정된 디렉토리에서 이미지 로드
    """
    def __init__(self, root_dir, transform=None, extensions=('.jpg', '.jpeg', '.png')):
        self.root_dir = root_dir
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.extensions = extensions
        
        # 이미지 파일 목록 수집
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(self.extensions):
                    self.image_paths.append(os.path.join(root, file))
        
        self.image_paths.sort()  # 일관된 순서 보장
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_path  # 이미지와 경로 반환

class ImageFolderDataset(Dataset):
    """
    ImageFolder 형식 데이터셋 래퍼
    클래스별로 하위 폴더에 이미지가 저장된 구조
    """
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(
            root=root_dir,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]) if transform is None else transform
        )
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def get_image_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    이미지 데이터셋에 대한 DataLoader 생성
    
    Args:
        dataset: 데이터셋 인스턴스
        batch_size: 배치 크기
        shuffle: 데이터 셔플 여부
        num_workers: 데이터 로딩 워커 수
        
    Returns:
        DataLoader 인스턴스
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

def get_mnist_dataloaders(batch_size=None, num_workers=None, pin_memory=None):
    """
    MNIST 데이터셋의 데이터 로더 생성
    
    Args:
        batch_size: 배치 크기 (None이면 설정 파일의 값 사용)
        num_workers: 데이터 로딩 워커 수 (None이면 설정 파일의 값 사용)
        pin_memory: pin_memory 설정 (None이면 설정 파일의 값 사용)
        
    Returns:
        train_loader, test_loader: 학습 및 테스트 데이터 로더
    """
    # 설정 파일 로드
    config = get_dataset_config('mnist')
    dataset_config = config['dataset']  # dataset 키 아래의 설정 사용
    
    # 변환 생성
    train_transform = create_transform_from_config(dataset_config['transform']['train'])
    test_transform = create_transform_from_config(dataset_config['transform']['test'])
    
    # 데이터 로더 설정
    train_loader_config = dataset_config['dataloader']['train'].copy()
    test_loader_config = dataset_config['dataloader']['test'].copy()
    
    # 인자로 전달된 값이 있으면 설정 파일의 값을 덮어씀
    if batch_size is not None:
        train_loader_config['batch_size'] = batch_size
        test_loader_config['batch_size'] = batch_size
    if num_workers is not None:
        train_loader_config['num_workers'] = num_workers
        test_loader_config['num_workers'] = num_workers
    if pin_memory is not None:
        train_loader_config['pin_memory'] = pin_memory
        test_loader_config['pin_memory'] = pin_memory
    
    # 데이터셋 생성
    train_dataset = datasets.MNIST(
        root=dataset_config['paths']['root'],
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.MNIST(
        root=dataset_config['paths']['root'],
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, **train_loader_config)
    test_loader = DataLoader(test_dataset, **test_loader_config)
    
    return train_loader, test_loader

def get_cifar10_dataloaders(batch_size=None, num_workers=None, pin_memory=None):
    """
    CIFAR-10 데이터셋의 데이터 로더 생성
    
    Args:
        batch_size: 배치 크기 (None이면 설정 파일의 값 사용)
        num_workers: 데이터 로딩 워커 수 (None이면 설정 파일의 값 사용)
        pin_memory: pin_memory 설정 (None이면 설정 파일의 값 사용)
        
    Returns:
        train_loader, test_loader: 학습 및 테스트 데이터 로더
    """
    # 설정 파일 로드
    config = get_dataset_config('cifar10')
    dataset_config = config['dataset']  # dataset 키 아래의 설정 사용
    
    # 변환 생성
    train_transform = create_transform_from_config(dataset_config['transform']['train'])
    test_transform = create_transform_from_config(dataset_config['transform']['test'])
    
    # 데이터 로더 설정
    train_loader_config = dataset_config['dataloader']['train'].copy()
    test_loader_config = dataset_config['dataloader']['test'].copy()
    
    # 인자로 전달된 값이 있으면 설정 파일의 값을 덮어씀
    if batch_size is not None:
        train_loader_config['batch_size'] = batch_size
        test_loader_config['batch_size'] = batch_size
    if num_workers is not None:
        train_loader_config['num_workers'] = num_workers
        test_loader_config['num_workers'] = num_workers
    if pin_memory is not None:
        train_loader_config['pin_memory'] = pin_memory
        test_loader_config['pin_memory'] = pin_memory
    
    # 데이터셋 생성
    train_dataset = datasets.CIFAR10(
        root=dataset_config['paths']['root'],
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=dataset_config['paths']['root'],
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, **train_loader_config)
    test_loader = DataLoader(test_dataset, **test_loader_config)
    
    return train_loader, test_loader

def visualize_image_batch(dataloader, num_images=16):
    """
    데이터로더에서 이미지 배치 시각화
    
    Args:
        dataloader: 이미지 데이터로더
        num_images: 시각화할 이미지 수
        
    Returns:
        시각화된 이미지 그리드 (numpy 배열)
    """
    import matplotlib.pyplot as plt
    
    # 배치 가져오기
    images, labels = next(iter(dataloader))
    
    # 이미지 수 제한
    images = images[:num_images]
    labels = labels[:num_images] if labels is not None else None
    
    # 그리드 크기 계산
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    # 이미지 그리드 생성
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # 이미지 정규화 해제
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = (img * 0.5 + 0.5).clip(0, 1)  # [-1, 1] -> [0, 1]
            
            ax.imshow(img)
            if labels is not None:
                ax.set_title(f"Label: {labels[i].item()}")
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    # 그림을 numpy 배열로 변환
    fig.canvas.draw()
    grid_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    grid_image = grid_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return grid_image
