"""
Vector Quantized VAE (VQ-VAE) 모델 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    벡터 양자화 모듈
    연속적인 잠재 벡터를 이산적인 코드북 엔트리로 양자화
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 코드북 초기화
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
    def forward(self, inputs):
        # 입력 형태 변환
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # 각 잠재 벡터와 코드북 엔트리 간의 거리 계산
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # 가장 가까운 코드북 엔트리 찾기
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 양자화된 벡터 가져오기
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # 손실 계산
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices

class VQVAE(nn.Module):
    """
    Vector Quantized VAE 모델
    인코더, 벡터 양자화, 디코더로 구성
    """
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, num_embeddings=512, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        
        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 벡터 양자화
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z)
        x_recon = self.decoder(z_q)
        
        return x_recon, vq_loss, indices
    
    def encode(self, x):
        """입력 데이터를 잠재 공간으로 인코딩"""
        z = self.encoder(x)
        z_q, _, indices = self.vq(z)
        return z_q, indices
    
    def decode(self, z_q):
        """양자화된 잠재 벡터를 원본 데이터 공간으로 디코딩"""
        return self.decoder(z_q)
    
    def decode_indices(self, indices):
        """코드북 인덱스를 원본 데이터 공간으로 디코딩"""
        z_q = self.vq.embedding(indices).squeeze(1)
        return self.decoder(z_q)

class ConvVQVAE(nn.Module):
    """
    합성곱 Vector Quantized VAE 모델
    이미지 데이터 처리에 적합
    """
    def __init__(self, in_channels=1, hidden_dim=128, latent_dim=64, num_embeddings=512, commitment_cost=0.25):
        super(ConvVQVAE, self).__init__()
        
        # 인코더
        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 7x7 -> 4x4
            nn.Conv2d(64, latent_dim, kernel_size=4, stride=2, padding=1)
        )
        
        # 벡터 양자화
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        
        # 디코더
        self.decoder = nn.Sequential(
            # 4x4 -> 7x7
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 7x7 -> 14x14
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=0, output_padding=1),
            nn.Sigmoid()
        )
        
        # 크기 계산을 위한 더미 입력
        self.register_buffer('dummy_input', torch.zeros(1, in_channels, 28, 28))
        with torch.no_grad():
            # 인코더 크기 확인
            dummy_output = self.encoder(self.dummy_input)
            self.encoder_output_size = dummy_output.shape[2:]
            print(f"\n인코더 크기 변화:")
            print(f"입력 크기: {self.dummy_input.shape[2:]}")
            print(f"인코더 출력 크기: {self.encoder_output_size}")
            
            # 디코더 크기 확인
            dummy_decoder_input = torch.zeros(1, latent_dim, *self.encoder_output_size)
            print(f"\n디코더 크기 변화:")
            print(f"디코더 입력 크기: {dummy_decoder_input.shape[2:]}")
            
            # 각 레이어의 출력 크기 확인
            x = dummy_decoder_input
            for i, layer in enumerate(self.decoder):
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                    x = layer(x)
                    print(f"디코더 레이어 {i} 출력 크기: {x.shape[2:]}")
                    if isinstance(layer, nn.ConvTranspose2d):
                        print(f"  - kernel_size={layer.kernel_size}, stride={layer.stride}, padding={layer.padding}, output_padding={getattr(layer, 'output_padding', 0)}")
                        # 크기 계산 공식 출력
                        input_size = x.shape[2]
                        output_size = (input_size - 1) * layer.stride[0] - 2 * layer.padding[0] + (layer.kernel_size[0] - 1) + 1
                        if hasattr(layer, 'output_padding'):
                            output_size += layer.output_padding[0]
                        print(f"  - 크기 계산: ({input_size} - 1) * {layer.stride[0]} - 2 * {layer.padding[0]} + ({layer.kernel_size[0]} - 1) + 1 = {output_size}")
            
            # 최종 출력 크기 확인
            dummy_decoder_output = self.decoder(dummy_decoder_input)
            print(f"\n최종 출력 크기: {dummy_decoder_output.shape[2:]}")
            
            # 크기가 일치하지 않으면 경고
            if dummy_decoder_output.shape[2:] != (28, 28):
                print(f"\n경고: 디코더 출력 크기가 입력 크기와 일치하지 않습니다!")
                print(f"예상 크기: (28, 28), 실제 크기: {dummy_decoder_output.shape[2:]}")
                print("\n디코더 레이어 크기 계산:")
                for i, layer in enumerate(self.decoder):
                    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                        print(f"레이어 {i}: kernel_size={layer.kernel_size}, stride={layer.stride}, padding={layer.padding}, output_padding={getattr(layer, 'output_padding', 0)}")
        
    def forward(self, x):
        z = self.encoder(x)
        
        # 벡터 양자화를 위한 형태 변환
        z_shape = z.shape
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, z_shape[1])
        
        # 벡터 양자화
        z_q_flattened, vq_loss, indices = self.vq(z_flattened)
        
        # 원래 형태로 복원
        z_q = z_q_flattened.view(z_shape[0], z_shape[2], z_shape[3], z_shape[1]).permute(0, 3, 1, 2).contiguous()
        
        # 디코딩
        x_recon = self.decoder(z_q)
        
        return x_recon, vq_loss, indices
    
    def encode(self, x):
        """입력 이미지를 잠재 공간으로 인코딩"""
        z = self.encoder(x)
        
        # 벡터 양자화를 위한 형태 변환
        z_shape = z.shape
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, z_shape[1])
        
        # 벡터 양자화
        z_q_flattened, _, indices = self.vq(z_flattened)
        
        # 원래 형태로 복원
        z_q = z_q_flattened.view(z_shape[0], z_shape[2], z_shape[3], z_shape[1]).permute(0, 3, 1, 2).contiguous()
        
        return z_q, indices
    
    def decode(self, z_q):
        """양자화된 잠재 벡터를 이미지로 디코딩"""
        return self.decoder(z_q)
        
    def loss_function(self, recon_x, x, vq_loss, indices=None):
        """
        VQVAE 손실 함수 계산
        
        Args:
            recon_x: 재구성된 입력
            x: 원본 입력
            vq_loss: 벡터 양자화 손실
            indices: 코드북 인덱스 (선택적)
            
        Returns:
            손실 구성요소를 포함한 딕셔너리
        """
        # 재구성 손실 (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # 총 손실
        total_loss = recon_loss + vq_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss
        }

def train_vqvae(model, dataloader, epochs=10, lr=1e-3, device='cpu'):
    """
    VQ-VAE 모델 학습 함수
    
    Args:
        model: 학습할 VQ-VAE 모델
        dataloader: 학습 데이터 로더
        epochs: 학습 에폭 수
        lr: 학습률
        device: 학습 장치 (CPU 또는 CUDA)
        
    Returns:
        학습된 모델과 학습 손실 기록
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = {'total': [], 'recon': [], 'vq': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        recon_loss = 0
        vq_loss_total = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            
            # 데이터 형태 변환 (필요한 경우)
            if len(data.shape) == 4 and isinstance(model, VQVAE):
                data = data.view(data.size(0), -1)
            
            # 순전파
            optimizer.zero_grad()
            recon_batch, vq_loss, _ = model(data)
            
            # 재구성 손실
            recon_error = F.mse_loss(recon_batch, data)
            
            # 총 손실
            loss = recon_error + vq_loss
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss += recon_error.item()
            vq_loss_total += vq_loss.item()
            
        avg_loss = train_loss / len(dataloader)
        avg_recon = recon_loss / len(dataloader)
        avg_vq = vq_loss_total / len(dataloader)
        
        losses['total'].append(avg_loss)
        losses['recon'].append(avg_recon)
        losses['vq'].append(avg_vq)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Recon: {avg_recon:.6f}, VQ: {avg_vq:.6f}')
    
    return model, losses

def generate_from_codes(model, indices, device='cpu'):
    """
    코드북 인덱스로부터 이미지 생성
    
    Args:
        model: 학습된 VQ-VAE 모델
        indices: 코드북 인덱스
        device: 실행 장치 (CPU 또는 CUDA)
        
    Returns:
        생성된 이미지
    """
    model = model.to(device)
    model.eval()
    
    indices = indices.to(device)
    
    with torch.no_grad():
        if hasattr(model, 'decode_indices'):
            samples = model.decode_indices(indices)
        else:
            # ConvVQVAE의 경우 별도 처리 필요
            z_q = model.vq.embedding(indices)
            # 형태 변환 필요 (구현에 따라 다름)
            samples = model.decoder(z_q)
    
    return samples
