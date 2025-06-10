"""
데이터 변환을 위한 유틸리티 함수들
"""
import torch
import torch.nn.functional as F

def flatten_batch(batch):
    """
    배치 데이터를 평탄화하는 함수
    
    Args:
        batch: (B, C, H, W) 형태의 이미지 배치 텐서
        
    Returns:
        (B, C*H*W) 형태의 평탄화된 텐서
    """
    if isinstance(batch, (tuple, list)):
        # 배치가 (이미지, 레이블) 형태인 경우
        images, labels = batch
        return images.view(images.size(0), -1), labels
    else:
        # 배치가 이미지만 있는 경우
        return batch.view(batch.size(0), -1)

def unflatten_batch(batch, shape):
    """
    평탄화된 배치 데이터를 원래 형태로 복원하는 함수
    
    Args:
        batch: (B, C*H*W) 형태의 평탄화된 텐서
        shape: 원래 텐서의 형태 (C, H, W)
        
    Returns:
        (B, C, H, W) 형태의 텐서
    """
    if isinstance(batch, (tuple, list)):
        # 배치가 (이미지, 레이블) 형태인 경우
        images, labels = batch
        return images.view(images.size(0), *shape), labels
    else:
        # 배치가 이미지만 있는 경우
        return batch.view(batch.size(0), *shape)

def normalize_batch(batch, mean, std):
    """
    배치 데이터를 정규화하는 함수
    
    Args:
        batch: (B, C, H, W) 형태의 이미지 배치 텐서
        mean: 채널별 평균값
        std: 채널별 표준편차
        
    Returns:
        정규화된 텐서
    """
    if isinstance(batch, (tuple, list)):
        # 배치가 (이미지, 레이블) 형태인 경우
        images, labels = batch
        return F.normalize(images, mean=mean, std=std), labels
    else:
        # 배치가 이미지만 있는 경우
        return F.normalize(batch, mean=mean, std=std)

def denormalize_batch(batch, mean, std):
    """
    정규화된 배치 데이터를 원래 범위로 복원하는 함수
    
    Args:
        batch: 정규화된 (B, C, H, W) 형태의 이미지 배치 텐서
        mean: 채널별 평균값
        std: 채널별 표준편차
        
    Returns:
        원래 범위의 텐서
    """
    if isinstance(batch, (tuple, list)):
        # 배치가 (이미지, 레이블) 형태인 경우
        images, labels = batch
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(images.device)
        std = torch.tensor(std).view(1, -1, 1, 1).to(images.device)
        return images * std + mean, labels
    else:
        # 배치가 이미지만 있는 경우
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(batch.device)
        std = torch.tensor(std).view(1, -1, 1, 1).to(batch.device)
        return batch * std + mean 