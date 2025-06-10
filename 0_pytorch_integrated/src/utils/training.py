"""
학습 및 평가 유틸리티 모듈
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import torch.nn.functional as F

# 모델 클래스 임포트
from src.models.vae import VAE

class Trainer:
    """
    모델 학습 및 평가를 위한 범용 트레이너 클래스
    """
    def __init__(self, model, train_loader, val_loader=None, criterion=None, optimizer=None, 
                 lr=1e-3, device='cpu', save_dir='./results/models', log_interval=10, transform_fn=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 손실 함수가 제공되지 않은 경우 기본값 설정
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        
        # 옵티마이저가 제공되지 않은 경우 기본값 설정
        self.optimizer = optimizer if optimizer is not None else optim.Adam(model.parameters(), lr=lr)
        
        self.device = device
        self.save_dir = save_dir
        self.log_interval = log_interval  # 로깅 간격 설정
        self.current_epoch = 0  # 현재 에포크 추적
        self.transform_fn = transform_fn  # 데이터 변환 함수 추가
        
        # 결과 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 모델을 장치로 이동
        self.model = self.model.to(self.device)
        
        # 학습 기록
        self.history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'val_loss': [],
            'val_recon_loss': [],
            'val_kl_loss': []
        }
        
    def train_epoch(self):
        """한 에포크 동안 모델 학습"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            
            # 데이터 변환 적용
            if self.transform_fn is not None:
                data = self.transform_fn(data)
            
            # 순전파
            self.optimizer.zero_grad()
            
            # 모델 타입에 따른 처리
            if isinstance(self.model, VAE):  # VAE 모델인 경우
                recon_batch, mu, logvar = self.model(data)
                loss_dict = self.model.loss_function(recon_batch, data, mu, logvar)
                loss = loss_dict['total_loss']
                recon_loss = loss_dict['recon_loss']
                kl_loss = loss_dict['kl_loss']
            elif 'VQVAE' in self.model.__class__.__name__:  # VQVAE 모델인 경우
                recon_batch, vq_loss, _ = self.model(data)
                loss_dict = self.model.loss_function(recon_batch, data, vq_loss)
                loss = loss_dict['total_loss']
                recon_loss = loss_dict['recon_loss']
                kl_loss = loss_dict['vq_loss']  # VQ 손실을 KL 손실로 표시
            elif 'AutoEncoder' in self.model.__class__.__name__:  # AutoEncoder 모델인 경우
                recon_batch, z = self.model(data)
                loss_dict = self.model.loss_function(recon_batch, data, z)
                loss = loss_dict['total_loss']
                recon_loss = loss_dict['recon_loss']
                kl_loss = torch.tensor(0.0, device=self.device)  # AutoEncoder는 KL 손실이 없음
            else:  # 다른 모델의 경우
                output = self.model(data)
                loss_dict = self.model.loss_function(output, target)
                loss = loss_dict['loss']
                recon_loss = loss  # 재구성 손실을 총 손실로 사용
                kl_loss = torch.tensor(0.0, device=self.device)  # KL 손실 없음
            
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            # 손실 누적
            batch_size = len(data)
            total_loss += loss.item() * batch_size
            total_recon_loss += recon_loss.item() * batch_size
            total_kl_loss += kl_loss.item() * batch_size
            total_samples += batch_size
            
            # 로깅
            if batch_idx % self.log_interval == 0:
                print(f'Train Epoch: {self.current_epoch + 1} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.6f}\t'
                      f'Recon: {recon_loss.item():.6f}\t'
                      f'KL/VQ: {kl_loss.item():.6f}')
        
        # 평균 손실 계산
        avg_loss = total_loss / total_samples
        avg_recon_loss = total_recon_loss / total_samples
        avg_kl_loss = total_kl_loss / total_samples
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def validate(self):
        """검증 데이터셋에서 모델 평가"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                
                # 데이터 변환 적용
                if self.transform_fn is not None:
                    data = self.transform_fn(data)
                
                # 모델 타입에 따른 처리
                if isinstance(self.model, VAE):  # VAE 모델인 경우
                    recon_batch, mu, logvar = self.model(data)
                    loss_dict = self.model.loss_function(recon_batch, data, mu, logvar)
                    loss = loss_dict['total_loss']
                    recon_loss = loss_dict['recon_loss']
                    kl_loss = loss_dict['kl_loss']
                elif 'VQVAE' in self.model.__class__.__name__:  # VQVAE 모델인 경우
                    recon_batch, vq_loss, _ = self.model(data)
                    loss_dict = self.model.loss_function(recon_batch, data, vq_loss)
                    loss = loss_dict['total_loss']
                    recon_loss = loss_dict['recon_loss']
                    kl_loss = loss_dict['vq_loss']  # VQ 손실을 KL 손실로 표시
                elif 'AutoEncoder' in self.model.__class__.__name__:  # AutoEncoder 모델인 경우
                    recon_batch, z = self.model(data)
                    loss_dict = self.model.loss_function(recon_batch, data, z)
                    loss = loss_dict['total_loss']
                    recon_loss = loss_dict['recon_loss']
                    kl_loss = torch.tensor(0.0, device=self.device)  # AutoEncoder는 KL 손실이 없음
                else:  # 다른 모델의 경우
                    output = self.model(data)
                    loss_dict = self.model.loss_function(output, target)
                    loss = loss_dict['loss']
                    recon_loss = loss  # 재구성 손실을 총 손실로 사용
                    kl_loss = torch.tensor(0.0, device=self.device)  # KL 손실 없음
                
                # 손실 누적
                batch_size = len(data)
                total_loss += loss.item() * batch_size
                total_recon_loss += recon_loss.item() * batch_size
                total_kl_loss += kl_loss.item() * batch_size
                total_samples += batch_size
        
        # 평균 손실 계산
        avg_loss = total_loss / total_samples
        avg_recon_loss = total_recon_loss / total_samples
        avg_kl_loss = total_kl_loss / total_samples
        
        print(f'Validation Loss: {avg_loss:.6f}, Recon Loss: {avg_recon_loss:.6f}, KL/VQ Loss: {avg_kl_loss:.6f}')
        
        return {
            'val_loss': avg_loss,
            'val_recon_loss': avg_recon_loss,
            'val_kl_loss': avg_kl_loss
        }
    
    def train(self, epochs, save_best=True, early_stopping=None):
        """
        모델 학습
        
        Args:
            epochs: 학습 에폭 수
            save_best: 최상의 모델 저장 여부
            early_stopping: 조기 종료 에폭 수 (None이면 사용 안 함)
            
        Returns:
            학습 기록
        """
        best_val_loss = float('inf')
        no_improve_epochs = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch  # 현재 에포크 업데이트
            start_time = time.time()
            
            # 학습
            train_metrics = self.train_epoch()
            train_loss = train_metrics['loss']
            train_recon_loss = train_metrics['recon_loss']
            train_kl_loss = train_metrics['kl_loss']
            
            # 검증
            val_metrics = self.validate()
            val_loss = val_metrics['val_loss']
            val_recon_loss = val_metrics['val_recon_loss']
            val_kl_loss = val_metrics['val_kl_loss']
            
            # 학습 시간 계산
            epoch_time = time.time() - start_time
            
            # 히스토리 업데이트
            self.history['train_loss'].append(train_loss)
            self.history['train_recon_loss'].append(train_recon_loss)
            self.history['train_kl_loss'].append(train_kl_loss)
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_recon_loss'].append(val_recon_loss)
                self.history['val_kl_loss'].append(val_kl_loss)
            
            # 결과 출력
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - train_loss: {train_loss:.4f}, train_recon_loss: {train_recon_loss:.4f}, train_kl_loss: {train_kl_loss:.4f}")
            if val_loss is not None:
                print(f" - val_loss: {val_loss:.4f}, val_recon_loss: {val_recon_loss:.4f}, val_kl_loss: {val_kl_loss:.4f}")
            
            # 최상의 모델 저장
            if save_best and val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(os.path.join(self.save_dir, 'best_model.pth'))
                no_improve_epochs = 0
            elif save_best and val_loss is not None:
                no_improve_epochs += 1
            
            # 매 에포크마다 모델 저장
            self.save_model(os.path.join(self.save_dir, f'model_epoch_{epoch+1}.pth'))
            
            # 조기 종료
            if early_stopping is not None and no_improve_epochs >= early_stopping:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # 최종 모델 저장
        self.save_model(os.path.join(self.save_dir, 'final_model.pth'))
        
        # 학습 기록 저장
        self.save_history()
        
        return self.history
    
    def _vae_loss(self, recon_x, x, mu, logvar, beta=1.0):
        """VAE 손실 함수"""
        # 재구성 손실
        recon_loss = self.criterion(recon_x, x)
        
        # KL 발산
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 총 손실
        return recon_loss + beta * kl_loss
    
    def _clip_loss(self, logits_per_image, logits_per_text):
        """CLIP 손실 함수"""
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # 이미지->텍스트 방향 손실
        loss_i = nn.functional.cross_entropy(logits_per_image, labels)
        
        # 텍스트->이미지 방향 손실
        loss_t = nn.functional.cross_entropy(logits_per_text, labels)
        
        # 총 손실
        return (loss_i + loss_t) / 2
    
    def save_model(self, path):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def save_history(self):
        """학습 기록 저장"""
        # 기록을 JSON으로 변환 가능하게 처리
        history_json = {k: [float(v) for v in vals] for k, vals in self.history.items() if vals}
        
        # 저장
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(history_json, f)
    
    def plot_history(self, figsize=(15, 5), save_path=None):
        """학습 기록 시각화"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 손실 그래프
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 재구성 손실 그래프
        axes[1].plot(self.history['train_recon_loss'], label='Train Recon Loss')
        if self.history['val_recon_loss']:
            axes[1].plot(self.history['val_recon_loss'], label='Validation Recon Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Reconstruction Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # KL 발산 그래프
        axes[2].plot(self.history['train_kl_loss'], label='Train KL Loss')
        if self.history['val_kl_loss']:
            axes[2].plot(self.history['val_kl_loss'], label='Validation KL Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('KL Divergence')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장 (요청된 경우)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class Evaluator:
    """
    모델 평가를 위한 범용 평가기 클래스
    """
    def __init__(self, model, test_loader, criterion=None, device='cpu', save_dir='./results'):
        self.model = model
        self.test_loader = test_loader
        
        # 손실 함수가 제공되지 않은 경우 기본값 설정
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        
        self.device = device
        self.save_dir = save_dir
        
        # 결과 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 모델을 장치로 이동
        self.model = self.model.to(self.device)
    
    def evaluate(self, save_results=True):
        """
        모델 평가
        
        Args:
            save_results: 결과 저장 여부
            
        Returns:
            평가 결과 딕셔너리
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                # 배치 데이터 처리
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    data, target = batch[0], batch[1]
                    has_target = True
                else:
                    data = batch
                    has_target = False
                
                # 데이터를 장치로 이동
                data = data.to(self.device)
                if has_target:
                    target = target.to(self.device)
                
                # 순전파
                if 'VAE' in self.model.__class__.__name__:
                    # VAE 모델 처리
                    recon_batch, mu, logvar = self.model(data)
                    loss = self._vae_loss(recon_batch, data, mu, logvar)
                    output = recon_batch
                elif 'VQVAE' in self.model.__class__.__name__:
                    # VQ-VAE 모델 처리
                    recon_batch, vq_loss, _ = self.model(data)
                    recon_loss = self.criterion(recon_batch, data)
                    loss = recon_loss + vq_loss
                    output = recon_batch
                elif 'CLIP' in self.model.__class__.__name__ and isinstance(batch, tuple) and len(batch) >= 2:
                    # CLIP 모델 처리
                    logits_per_image, logits_per_text = self.model(data, target)
                    loss = self._clip_loss(logits_per_image, logits_per_text)
                    output = logits_per_image
                else:
                    # 일반 모델 처리
                    output = self.model(data)
                    if has_target:
                        loss = self.criterion(output, target)
                    else:
                        # 자기지도 학습 (예: 오토인코더)
                        loss = self.criterion(output, data)
                
                # 손실 누적
                total_loss += loss.item()
                
                # 출력 및 타겟 수집
                all_outputs.append(output.cpu())
                if has_target:
                    all_targets.append(target.cpu())
                else:
                    all_targets.append(data.cpu())
                
                # 정확도 계산 (분류 문제인 경우)
                if has_target and len(target.size()) == 1:  # 분류 문제
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
        
        # 평균 손실 및 정확도 계산
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total if total > 0 else 0
        
        # 결과 딕셔너리
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'outputs': torch.cat(all_outputs),
            'targets': torch.cat(all_targets)
        }
        
        # 결과 출력
        print(f"Test Loss: {avg_loss:.4f}")
        if accuracy > 0:
            print(f"Test Accuracy: {accuracy:.2f}%")
        
        # 결과 저장
        if save_results:
            self.save_results(results)
        
        return results
    
    def _vae_loss(self, recon_x, x, mu, logvar, beta=1.0):
        """VAE 손실 함수"""
        # 재구성 손실
        recon_loss = self.criterion(recon_x, x)
        
        # KL 발산
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 총 손실
        return recon_loss + beta * kl_loss
    
    def _clip_loss(self, logits_per_image, logits_per_text):
        """CLIP 손실 함수"""
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # 이미지->텍스트 방향 손실
        loss_i = nn.functional.cross_entropy(logits_per_image, labels)
        
        # 텍스트->이미지 방향 손실
        loss_t = nn.functional.cross_entropy(logits_per_text, labels)
        
        # 총 손실
        return (loss_i + loss_t) / 2
    
    def save_results(self, results):
        """평가 결과 저장"""
        # 결과 파일 경로
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.save_dir, f'evaluation_results_{timestamp}.json')
        
        # 저장할 결과 준비
        save_results = {
            'loss': float(results['loss']),
            'accuracy': float(results['accuracy']) if 'accuracy' in results else None,
            'timestamp': timestamp
        }
        
        # 결과 저장
        with open(results_path, 'w') as f:
            json.dump(save_results, f)
        
        print(f"Evaluation results saved to {results_path}")
        
        return results_path

def get_device():
    """사용 가능한 장치 반환"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    """재현성을 위한 시드 설정"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
