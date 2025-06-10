"""
멀티모달 데이터셋 모듈
이미지-텍스트 쌍 데이터셋 구현
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import csv
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class FlickrDataset(Dataset):
    """
    Flickr8k/Flickr30k 데이터셋
    이미지와 캡션 쌍으로 구성
    """
    def __init__(self, root_dir, captions_file, tokenizer=None, vocab=None, transform=None, max_length=100):
        self.root_dir = root_dir
        self.max_length = max_length
        
        # 이미지 변환
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # 캡션 파일 로드
        self.image_captions = []
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            # 파일 형식 확인 (CSV 또는 JSON)
            if captions_file.endswith('.json'):
                data = json.load(f)
                for item in data:
                    image_path = os.path.join(root_dir, item['image'])
                    caption = item['caption']
                    self.image_captions.append((image_path, caption))
            else:  # CSV 형식 가정
                reader = csv.reader(f)
                next(reader, None)  # 헤더 건너뛰기
                for row in reader:
                    image_path = os.path.join(root_dir, row[0])
                    caption = row[1]
                    self.image_captions.append((image_path, caption))
        
        # 토크나이저 설정
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer("basic_english")
        
        # 어휘 구축 (제공되지 않은 경우)
        if vocab is None:
            def yield_tokens():
                for _, caption in self.image_captions:
                    yield self.tokenizer(caption)
            
            self.vocab = build_vocab_from_iterator(
                yield_tokens(),
                specials=["<unk>", "<pad>", "<bos>", "<eos>"],
                min_freq=2
            )
            self.vocab.set_default_index(self.vocab["<unk>"])
        else:
            self.vocab = vocab
        
        # 특수 토큰 인덱스
        self.pad_idx = self.vocab["<pad>"]
        self.bos_idx = self.vocab["<bos>"]
        self.eos_idx = self.vocab["<eos>"]
        
    def __len__(self):
        return len(self.image_captions)
    
    def __getitem__(self, idx):
        image_path, caption = self.image_captions[idx]
        
        # 이미지 로드 및 변환
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 캡션 토큰화 및 인덱싱
        tokens = self.tokenizer(caption)
        token_ids = [self.bos_idx] + [self.vocab[token] for token in tokens] + [self.eos_idx]
        
        # 최대 길이로 자르거나 패딩
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.pad_idx] * (self.max_length - len(token_ids))
        
        # 텐서 변환
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        return image, token_ids, caption

class COCODataset(Dataset):
    """
    MS COCO 데이터셋
    이미지와 캡션 쌍으로 구성
    """
    def __init__(self, root_dir, annotations_file, tokenizer=None, vocab=None, transform=None, max_length=100):
        self.root_dir = root_dir
        self.max_length = max_length
        
        # 이미지 변환
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # 어노테이션 파일 로드
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # 이미지 ID와 파일명 매핑
        self.image_id_to_filename = {}
        for image in annotations['images']:
            self.image_id_to_filename[image['id']] = image['file_name']
        
        # 이미지-캡션 쌍 구성
        self.image_captions = []
        for annotation in annotations['annotations']:
            image_id = annotation['image_id']
            caption = annotation['caption']
            if image_id in self.image_id_to_filename:
                image_path = os.path.join(root_dir, self.image_id_to_filename[image_id])
                self.image_captions.append((image_path, caption))
        
        # 토크나이저 설정
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer("basic_english")
        
        # 어휘 구축 (제공되지 않은 경우)
        if vocab is None:
            def yield_tokens():
                for _, caption in self.image_captions:
                    yield self.tokenizer(caption)
            
            self.vocab = build_vocab_from_iterator(
                yield_tokens(),
                specials=["<unk>", "<pad>", "<bos>", "<eos>"],
                min_freq=5
            )
            self.vocab.set_default_index(self.vocab["<unk>"])
        else:
            self.vocab = vocab
        
        # 특수 토큰 인덱스
        self.pad_idx = self.vocab["<pad>"]
        self.bos_idx = self.vocab["<bos>"]
        self.eos_idx = self.vocab["<eos>"]
        
    def __len__(self):
        return len(self.image_captions)
    
    def __getitem__(self, idx):
        image_path, caption = self.image_captions[idx]
        
        # 이미지 로드 및 변환
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 캡션 토큰화 및 인덱싱
        tokens = self.tokenizer(caption)
        token_ids = [self.bos_idx] + [self.vocab[token] for token in tokens] + [self.eos_idx]
        
        # 최대 길이로 자르거나 패딩
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.pad_idx] * (self.max_length - len(token_ids))
        
        # 텐서 변환
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        return image, token_ids, caption

class CustomMultimodalDataset(Dataset):
    """
    사용자 정의 멀티모달 데이터셋
    이미지-텍스트 쌍으로 구성
    """
    def __init__(self, root_dir, annotations_file, tokenizer=None, vocab=None, transform=None, max_length=100):
        self.root_dir = root_dir
        self.max_length = max_length
        
        # 이미지 변환
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # 어노테이션 파일 로드
        self.image_captions = []
        
        with open(annotations_file, 'r', encoding='utf-8') as f:
            # 파일 형식 확인 (CSV 또는 JSON)
            if annotations_file.endswith('.json'):
                data = json.load(f)
                for item in data:
                    image_path = os.path.join(root_dir, item['image'])
                    caption = item['caption']
                    self.image_captions.append((image_path, caption))
            else:  # CSV 형식 가정
                reader = csv.reader(f)
                next(reader, None)  # 헤더 건너뛰기
                for row in reader:
                    image_path = os.path.join(root_dir, row[0])
                    caption = row[1]
                    self.image_captions.append((image_path, caption))
        
        # 토크나이저 설정
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer("basic_english")
        
        # 어휘 구축 (제공되지 않은 경우)
        if vocab is None:
            def yield_tokens():
                for _, caption in self.image_captions:
                    yield self.tokenizer(caption)
            
            self.vocab = build_vocab_from_iterator(
                yield_tokens(),
                specials=["<unk>", "<pad>", "<bos>", "<eos>"]
            )
            self.vocab.set_default_index(self.vocab["<unk>"])
        else:
            self.vocab = vocab
        
        # 특수 토큰 인덱스
        self.pad_idx = self.vocab["<pad>"]
        self.bos_idx = self.vocab["<bos>"]
        self.eos_idx = self.vocab["<eos>"]
        
    def __len__(self):
        return len(self.image_captions)
    
    def __getitem__(self, idx):
        image_path, caption = self.image_captions[idx]
        
        # 이미지 로드 및 변환
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 캡션 토큰화 및 인덱싱
        tokens = self.tokenizer(caption)
        token_ids = [self.bos_idx] + [self.vocab[token] for token in tokens] + [self.eos_idx]
        
        # 최대 길이로 자르거나 패딩
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.pad_idx] * (self.max_length - len(token_ids))
        
        # 텐서 변환
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        return image, token_ids, caption

def get_multimodal_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    멀티모달 데이터셋에 대한 DataLoader 생성
    
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

def create_dummy_flickr_dataset(output_dir, num_samples=100):
    """
    테스트용 더미 Flickr 데이터셋 생성
    
    Args:
        output_dir: 출력 디렉토리
        num_samples: 생성할 샘플 수
        
    Returns:
        생성된 데이터셋 경로
    """
    import numpy as np
    from PIL import Image
    
    # 디렉토리 생성
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # 더미 이미지 및 캡션 생성
    captions = []
    
    for i in range(num_samples):
        # 더미 이미지 생성 (컬러 노이즈)
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # 이미지 저장
        img_filename = f'image_{i:05d}.jpg'
        img_path = os.path.join(images_dir, img_filename)
        img.save(img_path)
        
        # 더미 캡션 생성
        caption_templates = [
            "A photo of {}",
            "An image showing {}",
            "This picture displays {}",
            "A {} in the scene",
            "The image contains {}"
        ]
        
        objects = [
            "a cat", "a dog", "a person", "a car", "a tree",
            "a building", "a flower", "a bird", "a mountain", "a lake"
        ]
        
        template = np.random.choice(caption_templates)
        obj = np.random.choice(objects)
        caption = template.format(obj)
        
        captions.append({
            'image': f'images/{img_filename}',
            'caption': caption
        })
    
    # 캡션 파일 저장
    captions_file = os.path.join(output_dir, 'captions.json')
    with open(captions_file, 'w', encoding='utf-8') as f:
        json.dump(captions, f, indent=2)
    
    return output_dir, captions_file

def visualize_multimodal_batch(dataloader, num_samples=5):
    """
    데이터로더에서 멀티모달 배치 시각화
    
    Args:
        dataloader: 멀티모달 데이터로더
        num_samples: 시각화할 샘플 수
        
    Returns:
        시각화된 이미지-텍스트 쌍 (numpy 배열)
    """
    import matplotlib.pyplot as plt
    
    # 배치 가져오기
    images, token_ids, captions = next(iter(dataloader))
    
    # 샘플 수 제한
    images = images[:num_samples]
    captions = captions[:num_samples]
    
    # 그리드 생성
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i, (img, caption) in enumerate(zip(images, captions)):
        # 이미지 정규화 해제
        img = img.detach().cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = img.clip(0, 1)
        
        # 이미지 표시
        axes[i].imshow(img)
        axes[i].set_title(f"Caption: {caption}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 그림을 numpy 배열로 변환
    fig.canvas.draw()
    grid_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    grid_image = grid_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return grid_image
