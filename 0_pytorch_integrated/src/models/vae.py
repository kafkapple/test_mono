"""
변분 오토인코더(VAE) 모델 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class VAEEncoder(nn.Module):
    """
    VAE의 인코더 부분
    입력 데이터를 잠재 공간의 평균과 분산으로 인코딩
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class VAEDecoder(nn.Module):
    """
    VAE의 디코더 부분
    잠재 공간의 벡터를 원본 데이터 공간으로 디코딩
    """
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))
        return x_recon

class VAE(nn.Module):
    """
    변분 오토인코더 모델
    인코더와 디코더로 구성
    """
    def __init__(self, input_dim, hidden_dim=128, latent_dim=20):
        super(VAE, self).__init__()
        
        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 잠재 공간의 평균과 로그 분산
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 픽셀값을 [0,1] 범위로
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        """
        VAE 손실 함수 계산
        
        Args:
            recon_x: 재구성된 입력
            x: 원본 입력
            mu: 잠재 공간의 평균
            logvar: 잠재 공간의 로그 분산
            beta: KL 발산 항의 가중치 (기본값: 1.0)
            
        Returns:
            손실 구성요소를 포함한 딕셔너리
        """
        # 재구성 손실 (Binary Cross Entropy)
        # 입력값을 0~1 범위로 조정
        x = torch.sigmoid(x)  # 원본 입력을 0~1 범위로 조정
        recon_x = torch.sigmoid(recon_x)  # 재구성 출력을 0~1 범위로 조정
        recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL 발산
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 총 손실
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def sample(self, num_samples, device='cpu'):
        """잠재 공간에서 샘플링하여 새로운 데이터 생성"""
        z = torch.randn(num_samples, self.fc_mu.out_features).to(device)
        samples = self.decode(z)
        return samples

class ConvVAE(nn.Module):
    """
    합성곱 변분 오토인코더 모델
    이미지 데이터 처리에 적합
    """
    def __init__(self, in_channels=1, latent_dim=20):
        super(ConvVAE, self).__init__()
        
        # 인코더
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 잠재 공간 크기 계산 (28x28 이미지 기준)
        self.flatten_size = 128 * 4 * 4
        
        # 잠재 공간 매핑
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flatten_size)
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        """
        재매개화 트릭: 평균과 분산으로부터 잠재 벡터 샘플링
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
        
    def forward(self, x):
        # 인코딩
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # 재매개화
        z = self.reparameterize(mu, logvar)
        
        # 디코딩
        h = self.fc_decoder(z)
        x_recon = self.decoder(h)
        
        return x_recon, mu, logvar
    
    def encode(self, x):
        """입력 이미지를 잠재 공간으로 인코딩"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z):
        """잠재 공간의 벡터를 이미지로 디코딩"""
        h = self.fc_decoder(z)
        x_recon = self.decoder(h)
        return x_recon
    
    def sample(self, num_samples, device='cpu'):
        """잠재 공간에서 샘플링하여 새로운 이미지 생성"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE 손실 함수: 재구성 손실 + KL 발산
    
    Args:
        recon_x: 재구성된 입력
        x: 원본 입력
        mu: 잠재 공간의 평균
        logvar: 잠재 공간의 로그 분산
        beta: KL 발산 가중치 (beta-VAE)
        
    Returns:
        총 손실, 재구성 손실, KL 발산
    """
    # 재구성 손실 (이진 교차 엔트로피)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL 발산
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 총 손실
    loss = BCE + beta * KLD
    
    return loss, BCE, KLD

def train_vae(model, dataloader, epochs=10, lr=1e-3, beta=1.0, device='cpu'):
    """
    VAE 모델 학습 함수
    
    Args:
        model: 학습할 VAE 모델
        dataloader: 학습 데이터 로더
        epochs: 학습 에폭 수
        lr: 학습률
        beta: KL 발산 가중치 (beta-VAE)
        device: 학습 장치 (CPU 또는 CUDA)
        
    Returns:
        학습된 모델과 학습 손실 기록
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = {'total': [], 'recon': [], 'kl': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        recon_loss = 0
        kl_loss = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            
            # 데이터 형태 변환 (필요한 경우)
            if len(data.shape) == 4 and isinstance(model, VAE):
                data = data.view(data.size(0), -1)
            
            # 순전파
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = vae_loss_function(recon_batch, data, mu, logvar, beta)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss += bce.item()
            kl_loss += kld.item()
            
        avg_loss = train_loss / len(dataloader.dataset)
        avg_recon = recon_loss / len(dataloader.dataset)
        avg_kl = kl_loss / len(dataloader.dataset)
        
        losses['total'].append(avg_loss)
        losses['recon'].append(avg_recon)
        losses['kl'].append(avg_kl)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Recon: {avg_recon:.6f}, KL: {avg_kl:.6f}')
    
    return model, losses

def interpolate_latent_space(model, x1, x2, steps=10, device='cpu'):
    """
    두 입력 사이의 잠재 공간 보간
    
    Args:
        model: 학습된 VAE 모델
        x1: 첫 번째 입력 데이터
        x2: 두 번째 입력 데이터
        steps: 보간 단계 수
        device: 실행 장치 (CPU 또는 CUDA)
        
    Returns:
        보간된 이미지 시퀀스
    """
    model = model.to(device)
    model.eval()
    
    x1 = x1.to(device)
    x2 = x2.to(device)
    
    # 데이터 형태 변환 (필요한 경우)
    if len(x1.shape) == 4 and isinstance(model, VAE):
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
    
    # 잠재 공간으로 인코딩
    with torch.no_grad():
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)
    
    # 잠재 공간에서 보간
    interpolations = []
    for alpha in torch.linspace(0, 1, steps):
        z = mu1 * (1 - alpha) + mu2 * alpha
        with torch.no_grad():
            recon = model.decode(z)
        interpolations.append(recon)
    
    return interpolations
