"""
기본 오토인코더(AE) 모델 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    오토인코더의 인코더 부분
    입력 데이터를 잠재 공간으로 인코딩
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Decoder(nn.Module):
    """
    오토인코더의 디코더 부분
    잠재 공간의 벡터를 원본 데이터 공간으로 디코딩
    """
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class AutoEncoder(nn.Module):
    """
    기본 오토인코더 모델
    인코더와 디코더로 구성
    """
    def __init__(self, input_dim, hidden_dim=128, latent_dim=20):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def encode(self, x):
        """입력 데이터를 잠재 공간으로 인코딩"""
        return self.encoder(x)
    
    def decode(self, z):
        """잠재 공간의 벡터를 원본 데이터 공간으로 디코딩"""
        return self.decoder(z)
        
    def loss_function(self, recon_x, x, z=None):
        """
        오토인코더 손실 함수 계산
        
        Args:
            recon_x: 재구성된 입력
            x: 원본 입력
            z: 잠재 벡터 (선택적)
            
        Returns:
            손실 구성요소를 포함한 딕셔너리
        """
        # 재구성 손실 (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # 총 손실 (오토인코더는 재구성 손실만 사용)
        total_loss = recon_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'loss': total_loss  # Trainer 클래스와의 호환성을 위해 추가
        }

class ConvAutoEncoder(nn.Module):
    """
    합성곱 오토인코더 모델
    이미지 데이터 처리에 적합
    """
    def __init__(self, in_channels=1, latent_dim=8):
        super(ConvAutoEncoder, self).__init__()
        
        # 인코더
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # 잠재 공간 변환
        self.fc_encoder = nn.Linear(64 * 4 * 4, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, 64 * 4 * 4)
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 인코딩
        x = self.encoder(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        z = self.fc_encoder(x)
        
        # 디코딩
        x = self.fc_decoder(z)
        x = x.view(batch_size, 64, 4, 4)
        x_recon = self.decoder(x)
        
        return x_recon, z
    
    def encode(self, x):
        """입력 이미지를 잠재 공간으로 인코딩"""
        x = self.encoder(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        z = self.fc_encoder(x)
        return z
    
    def decode(self, z):
        """잠재 공간의 벡터를 이미지로 디코딩"""
        batch_size = z.size(0)
        x = self.fc_decoder(z)
        x = x.view(batch_size, 64, 4, 4)
        x_recon = self.decoder(x)
        return x_recon
        
    def loss_function(self, recon_x, x, z=None):
        """
        오토인코더 손실 함수 계산
        
        Args:
            recon_x: 재구성된 입력
            x: 원본 입력
            z: 잠재 벡터 (선택적)
            
        Returns:
            손실 구성요소를 포함한 딕셔너리
        """
        # 재구성 손실 (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # 총 손실 (오토인코더는 재구성 손실만 사용)
        total_loss = recon_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'loss': total_loss  # Trainer 클래스와의 호환성을 위해 추가
        }

def train_autoencoder(model, dataloader, epochs=10, lr=1e-3, device='cpu'):
    """
    오토인코더 모델 학습 함수
    
    Args:
        model: 학습할 오토인코더 모델
        dataloader: 학습 데이터 로더
        epochs: 학습 에폭 수
        lr: 학습률
        device: 학습 장치 (CPU 또는 CUDA)
        
    Returns:
        학습된 모델과 학습 손실 기록
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            
            # 데이터 형태 변환 (필요한 경우)
            if len(data.shape) == 4 and isinstance(model, AutoEncoder):
                data = data.view(data.size(0), -1)
            
            # 순전파
            optimizer.zero_grad()
            recon_batch, _ = model(data)
            loss = criterion(recon_batch, data)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_loss = train_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
    
    return model, losses

def generate_samples(model, num_samples=10, latent_dim=20, device='cpu'):
    """
    오토인코더 모델을 사용하여 샘플 생성
    
    Args:
        model: 학습된 오토인코더 모델
        num_samples: 생성할 샘플 수
        latent_dim: 잠재 공간 차원
        device: 실행 장치 (CPU 또는 CUDA)
        
    Returns:
        생성된 샘플
    """
    model = model.to(device)
    model.eval()
    
    # 잠재 공간에서 랜덤 샘플링
    z = torch.randn(num_samples, latent_dim).to(device)
    
    # 샘플 생성
    with torch.no_grad():
        samples = model.decode(z)
    
    return samples
