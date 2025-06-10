"""
텍스트 데이터셋 모듈
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import IMDB, AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
from collections import Counter
import re
import nltk
from nltk.tokenize import word_tokenize

# NLTK 데이터 다운로드 (필요시)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextDataset(Dataset):
    """
    기본 텍스트 데이터셋
    """
    def __init__(self, texts, labels=None, tokenizer=None, vocab=None, max_length=100):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        # 토크나이저 설정
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer("basic_english")
        
        # 어휘 구축 (제공되지 않은 경우)
        if vocab is None:
            def yield_tokens():
                for text in texts:
                    yield self.tokenizer(text)
            
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
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 토큰화 및 인덱싱
        tokens = self.tokenizer(text)
        token_ids = [self.bos_idx] + [self.vocab[token] for token in tokens] + [self.eos_idx]
        
        # 최대 길이로 자르거나 패딩
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.pad_idx] * (self.max_length - len(token_ids))
        
        # 텐서 변환
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # 레이블 반환 (있는 경우)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return token_ids, label
        
        return token_ids

class IMDBDataset(Dataset):
    """
    IMDB 영화 리뷰 데이터셋 래퍼
    """
    def __init__(self, root='./data', split='train', tokenizer=None, vocab=None, max_length=256):
        # 데이터 로드
        self.dataset = list(IMDB(root=root, split=split))
        
        # 텍스트와 레이블 분리
        self.texts = [item[1] for item in self.dataset]
        self.labels = [int(item[0]) for item in self.dataset]
        
        # 토크나이저 설정
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer("basic_english")
        self.max_length = max_length
        
        # 어휘 구축 (제공되지 않은 경우)
        if vocab is None:
            def yield_tokens():
                for text in self.texts:
                    yield self.tokenizer(text)
            
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
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 토큰화 및 인덱싱
        tokens = self.tokenizer(text)
        token_ids = [self.bos_idx] + [self.vocab[token] for token in tokens] + [self.eos_idx]
        
        # 최대 길이로 자르거나 패딩
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.pad_idx] * (self.max_length - len(token_ids))
        
        # 텐서 변환
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        
        return token_ids, label

class AGNewsDataset(Dataset):
    """
    AG News 데이터셋 래퍼
    """
    def __init__(self, root='./data', split='train', tokenizer=None, vocab=None, max_length=128):
        # 데이터 로드
        self.dataset = list(AG_NEWS(root=root, split=split))
        
        # 텍스트와 레이블 분리
        self.texts = [item[1] for item in self.dataset]
        self.labels = [int(item[0]) - 1 for item in self.dataset]  # 0-based 인덱싱
        
        # 토크나이저 설정
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer("basic_english")
        self.max_length = max_length
        
        # 어휘 구축 (제공되지 않은 경우)
        if vocab is None:
            def yield_tokens():
                for text in self.texts:
                    yield self.tokenizer(text)
            
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
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 토큰화 및 인덱싱
        tokens = self.tokenizer(text)
        token_ids = [self.bos_idx] + [self.vocab[token] for token in tokens] + [self.eos_idx]
        
        # 최대 길이로 자르거나 패딩
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.pad_idx] * (self.max_length - len(token_ids))
        
        # 텐서 변환
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        
        return token_ids, label

class CustomTextDataset(Dataset):
    """
    사용자 정의 텍스트 데이터셋
    텍스트 파일에서 데이터 로드
    """
    def __init__(self, file_path, tokenizer=None, vocab=None, max_length=100, has_labels=False, label_first=True):
        self.texts = []
        self.labels = [] if has_labels else None
        self.max_length = max_length
        
        # 파일에서 텍스트 로드
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if has_labels:
                    # 레이블과 텍스트 분리
                    if label_first:
                        parts = line.split('\t', 1)
                        if len(parts) == 2:
                            label, text = parts
                            self.labels.append(int(label))
                            self.texts.append(text)
                    else:
                        parts = line.split('\t', 1)
                        if len(parts) == 2:
                            text, label = parts
                            self.labels.append(int(label))
                            self.texts.append(text)
                else:
                    self.texts.append(line)
        
        # 토크나이저 설정
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer("basic_english")
        
        # 어휘 구축 (제공되지 않은 경우)
        if vocab is None:
            def yield_tokens():
                for text in self.texts:
                    yield self.tokenizer(text)
            
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
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 토큰화 및 인덱싱
        tokens = self.tokenizer(text)
        token_ids = [self.bos_idx] + [self.vocab[token] for token in tokens] + [self.eos_idx]
        
        # 최대 길이로 자르거나 패딩
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.pad_idx] * (self.max_length - len(token_ids))
        
        # 텐서 변환
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # 레이블 반환 (있는 경우)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return token_ids, label
        
        return token_ids

def get_text_dataloader(dataset, batch_size=32, shuffle=True):
    """
    텍스트 데이터셋에 대한 DataLoader 생성
    
    Args:
        dataset: 데이터셋 인스턴스
        batch_size: 배치 크기
        shuffle: 데이터 셔플 여부
        
    Returns:
        DataLoader 인스턴스
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

def get_imdb_dataloaders(batch_size=32, root='./data', max_length=256):
    """
    IMDB 데이터셋에 대한 학습 및 테스트 DataLoader 생성
    
    Args:
        batch_size: 배치 크기
        root: 데이터 저장 경로
        max_length: 최대 시퀀스 길이
        
    Returns:
        학습 및 테스트 DataLoader, 어휘
    """
    # 학습 데이터셋
    train_dataset = IMDBDataset(
        root=root,
        split='train',
        max_length=max_length
    )
    
    # 테스트 데이터셋 (학습 데이터셋의 어휘 공유)
    test_dataset = IMDBDataset(
        root=root,
        split='test',
        vocab=train_dataset.vocab,
        max_length=max_length
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader, train_dataset.vocab

def visualize_text_batch(dataloader, vocab, num_samples=5):
    """
    데이터로더에서 텍스트 배치 시각화
    
    Args:
        dataloader: 텍스트 데이터로더
        vocab: 어휘 객체
        num_samples: 시각화할 샘플 수
        
    Returns:
        시각화된 텍스트 샘플 목록
    """
    # 배치 가져오기
    batch = next(iter(dataloader))
    
    if isinstance(batch, tuple) and len(batch) == 2:
        tokens, labels = batch
        has_labels = True
    else:
        tokens = batch
        has_labels = False
    
    # 샘플 수 제한
    tokens = tokens[:num_samples]
    if has_labels:
        labels = labels[:num_samples]
    
    # 인덱스-토큰 매핑 생성
    idx_to_token = {idx: token for token, idx in vocab.get_stoi().items()}
    
    # 시각화 결과
    results = []
    
    for i in range(len(tokens)):
        # 패딩 및 특수 토큰 제거
        token_ids = tokens[i].tolist()
        
        # <bos>와 <eos> 사이의 토큰만 선택
        if vocab["<bos>"] in token_ids and vocab["<eos>"] in token_ids:
            start_idx = token_ids.index(vocab["<bos>"]) + 1
            end_idx = token_ids.index(vocab["<eos>"])
            token_ids = token_ids[start_idx:end_idx]
        else:
            # 패딩 토큰 제거
            token_ids = [t for t in token_ids if t != vocab["<pad>"]]
        
        # 토큰을 텍스트로 변환
        text = " ".join([idx_to_token[idx] for idx in token_ids])
        
        if has_labels:
            label = labels[i].item()
            results.append(f"Sample {i+1} (Label: {label}):\n{text}\n")
        else:
            results.append(f"Sample {i+1}:\n{text}\n")
    
    return results
