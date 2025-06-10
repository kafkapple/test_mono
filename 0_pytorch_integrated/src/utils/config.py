"""
설정 파일 관리 유틸리티
"""
import os
import yaml
from typing import Dict, Any, List, Tuple
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from src.utils.transforms import flatten_batch

def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML 설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    print(f"Attempting to load config from: {os.path.abspath(config_path)}")  # 디버깅 로그
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print(f"Successfully loaded config with keys: {list(config.keys())}")  # 디버깅 로그
            return config
    except Exception as e:
        print(f"Error loading config file: {str(e)}")  # 디버깅 로그
        raise

def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    데이터셋 설정 로드
    
    Args:
        dataset_name: 데이터셋 이름 (예: 'mnist', 'cifar10')
        
    Returns:
        데이터셋 설정 딕셔너리
    """
    config_path = os.path.join('configs', 'datasets', f'{dataset_name}.yaml')
    print(f"Loading dataset config from: {config_path}")  # 디버깅 로그
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    config = load_config(config_path)
    print(f"Loaded config keys: {list(config.keys())}")  # 디버깅 로그
    
    # 설정 파일의 구조 확인
    if 'dataset' not in config:
        raise KeyError(f"설정 파일에 'dataset' 키가 없습니다: {config_path}")
    
    return config  # 전체 설정 반환

def create_transform_from_config(transform_config: List[Dict[str, Any]]) -> transforms.Compose:
    """
    설정으로부터 데이터 변환 생성
    
    Args:
        transform_config: 변환 설정 리스트
        
    Returns:
        transforms.Compose 객체
    """
    transform_list = []
    
    for transform in transform_config:
        name = transform['name']
        params = {k: v for k, v in transform.items() if k != 'name'}
        
        if name == 'Lambda':
            # Lambda 변환의 경우 func 파라미터를 함수 객체로 변환
            func_name = params.pop('func')
            if func_name == 'flatten_batch':
                func = flatten_batch
            else:
                raise ValueError(f"알 수 없는 Lambda 함수: {func_name}")
            transform_list.append(transforms.Lambda(func))
        else:
            # 다른 변환의 경우 torchvision.transforms에서 클래스 가져오기
            transform_class = getattr(transforms, name)
            transform_list.append(transform_class(**params))
    
    return transforms.Compose(transform_list)

def create_dataloader_from_config(dataset, dataloader_config: Dict[str, Any]) -> DataLoader:
    """
    설정으로부터 데이터 로더 생성
    
    Args:
        dataset: 데이터셋 객체
        dataloader_config: 데이터 로더 설정
        
    Returns:
        DataLoader 객체
    """
    return DataLoader(
        dataset,
        batch_size=dataloader_config['batch_size'],
        shuffle=dataloader_config.get('shuffle', False),
        num_workers=dataloader_config.get('num_workers', 0),
        pin_memory=dataloader_config.get('pin_memory', False),
        drop_last=dataloader_config.get('drop_last', False)
    )

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    모델 설정 로드
    
    Args:
        model_name: 모델 이름 (예: 'vae', 'conv_vae')
        
    Returns:
        모델 설정 딕셔너리
    """
    config_path = os.path.join('configs', 'models', f'{model_name}.yaml')
    config = load_config(config_path)
    return config['model']

def create_model_from_config(config: Dict[str, Any]):
    """
    설정으로부터 모델 생성
    
    Args:
        config: 모델 설정 딕셔너리
        
    Returns:
        생성된 모델 인스턴스
    """
    from src.models.vae import VAE, ConvVAE
    
    model_type = config['type']
    arch_config = config['architecture']
    
    if model_type == 'vae':
        return VAE(
            input_dim=arch_config['input_dim'],
            hidden_dim=arch_config['hidden_dim'],
            latent_dim=arch_config['latent_dim']
        )
    elif model_type == 'conv_vae':
        conv_config = arch_config['conv_vae']
        return ConvVAE(
            in_channels=conv_config['in_channels'],
            latent_dim=arch_config['latent_dim']
        )
    else:
        raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")

def get_optimizer_from_config(model, config: Dict[str, Any]):
    """
    설정으로부터 옵티마이저 생성
    
    Args:
        model: 최적화할 모델
        config: 옵티마이저 설정
        
    Returns:
        생성된 옵티마이저 인스턴스
    """
    import torch.optim as optim
    
    opt_config = config['training']['optimizer']
    opt_name = opt_config['name'].lower()
    lr = float(config['training']['learning_rate'])  # 문자열을 float로 변환
    
    if opt_name == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=opt_config['weight_decay']
        )
    elif opt_name == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=opt_config['weight_decay']
        )
    else:
        raise ValueError(f"지원하지 않는 옵티마이저입니다: {opt_name}")

def get_scheduler_from_config(optimizer, config: Dict[str, Any]):
    """
    설정으로부터 학습률 스케줄러 생성
    
    Args:
        optimizer: 최적화할 옵티마이저
        config: 스케줄러 설정
        
    Returns:
        생성된 스케줄러 인스턴스 또는 None
    """
    import torch.optim.lr_scheduler as lr_scheduler
    
    scheduler_config = config['training']['scheduler']
    scheduler_name = scheduler_config['name'].lower()
    
    if scheduler_name == 'none':
        return None
    elif scheduler_name == 'step':
        return lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config['step_size'],
            gamma=scheduler_config['gamma']
        )
    elif scheduler_name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    else:
        raise ValueError(f"지원하지 않는 스케줄러입니다: {scheduler_name}") 