"""
Flamingo/Perceiver IO 모델 구현
멀티모달 이미지-텍스트 모델
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceiverAttention(nn.Module):
    """
    Perceiver IO의 크로스 어텐션 모듈
    """
    def __init__(self, dim, num_heads=8, head_dim=64, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        inner_dim = num_heads * head_dim
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context=None):
        """
        Args:
            x: 쿼리 텐서 [batch, n, dim]
            context: 키/값 텐서 [batch, m, dim], None이면 self-attention
        """
        context = x if context is None else context
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # 헤드 분할
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 어텐션 계산
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # 값과 가중합
        out = torch.matmul(attn, v)
        
        # 헤드 결합
        out = out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], -1)
        
        # 출력 프로젝션
        return self.to_out(out)

class PerceiverBlock(nn.Module):
    """
    Perceiver IO의 기본 블록
    크로스 어텐션 + 셀프 어텐션 + FFN
    """
    def __init__(self, dim, num_heads=8, head_dim=64, mlp_dim=2048, dropout=0.0):
        super().__init__()
        self.cross_attn = PerceiverAttention(dim, num_heads, head_dim, dropout)
        self.self_attn = PerceiverAttention(dim, num_heads, head_dim, dropout)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context=None):
        # 크로스 어텐션
        x = x + self.cross_attn(self.norm1(x), context)
        
        # 셀프 어텐션
        x = x + self.self_attn(self.norm2(x))
        
        # FFN
        x = x + self.mlp(self.norm3(x))
        
        return x

class PerceiverIO(nn.Module):
    """
    Perceiver IO 모델
    다양한 모달리티의 입력을 처리하고 쿼리에 따라 출력 생성
    """
    def __init__(self, input_dim, latent_dim=512, num_latents=256, num_blocks=6, num_heads=8, head_dim=64, mlp_dim=2048, dropout=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        
        # 입력 프로젝션
        self.input_proj = nn.Linear(input_dim, latent_dim)
        
        # 잠재 어레이 (학습 가능한 쿼리)
        self.latent_array = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        
        # Perceiver 블록
        self.blocks = nn.ModuleList([
            PerceiverBlock(latent_dim, num_heads, head_dim, mlp_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # 출력 프로젝션
        self.output_proj = nn.Linear(latent_dim, input_dim)
        
    def forward(self, x, output_query=None):
        """
        Args:
            x: 입력 데이터 [batch, seq_len, input_dim]
            output_query: 출력 쿼리 [batch, query_len, latent_dim], None이면 잠재 어레이 반환
        """
        batch_size = x.shape[0]
        
        # 입력 프로젝션
        x = self.input_proj(x)
        
        # 잠재 어레이 확장
        latent = self.latent_array.expand(batch_size, -1, -1)
        
        # Perceiver 블록 통과
        for block in self.blocks:
            latent = block(latent, x)
        
        # 출력 쿼리가 있으면 디코딩
        if output_query is not None:
            # 출력 쿼리와 잠재 어레이 간의 크로스 어텐션
            output = self.output_proj(output_query @ latent.transpose(1, 2))
            return output
        
        return latent

class FlamingoModel(nn.Module):
    """
    Flamingo 모델 (Perceiver IO 기반)
    이미지와 텍스트를 처리하는 멀티모달 모델
    """
    def __init__(self, vocab_size, image_size=224, patch_size=16, latent_dim=512, num_latents=256, num_blocks=6, num_heads=8, head_dim=64, mlp_dim=2048, dropout=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 이미지 인코더 (간소화된 ViT)
        self.image_encoder = ImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            dim=latent_dim
        )
        
        # 텍스트 인코더
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            dim=latent_dim
        )
        
        # Perceiver IO
        self.perceiver = PerceiverIO(
            input_dim=latent_dim,
            latent_dim=latent_dim,
            num_latents=num_latents,
            num_blocks=num_blocks,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        
        # 텍스트 디코더 (언어 모델링용)
        self.text_decoder = TextDecoder(
            vocab_size=vocab_size,
            dim=latent_dim
        )
        
    def forward(self, images, text_tokens, text_targets=None):
        """
        Args:
            images: 이미지 텐서 [batch, channels, height, width]
            text_tokens: 텍스트 토큰 [batch, seq_len]
            text_targets: 텍스트 타겟 토큰 (언어 모델링용) [batch, seq_len]
        """
        batch_size = images.shape[0]
        
        # 이미지 인코딩
        image_features = self.image_encoder(images)
        
        # 텍스트 인코딩
        text_features = self.text_encoder(text_tokens)
        
        # 멀티모달 특성 결합
        multimodal_features = torch.cat([image_features, text_features], dim=1)
        
        # Perceiver IO 처리
        latent = self.perceiver(multimodal_features)
        
        # 텍스트 디코딩 (언어 모델링)
        if text_targets is not None:
            # 텍스트 디코더 쿼리 생성
            decoder_query = self.text_encoder(text_targets)
            
            # 디코딩
            logits = self.text_decoder(decoder_query, latent)
            
            return logits
        
        return latent
    
    def generate_text(self, images, prompt_tokens, max_length=50, temperature=1.0, device='cpu'):
        """
        이미지와 프롬프트를 기반으로 텍스트 생성
        
        Args:
            images: 이미지 텐서 [batch, channels, height, width]
            prompt_tokens: 프롬프트 토큰 [batch, prompt_len]
            max_length: 생성할 최대 토큰 수
            temperature: 샘플링 온도
            device: 실행 장치
            
        Returns:
            생성된 텍스트 토큰
        """
        self.eval()
        batch_size = images.shape[0]
        
        # 이미지 인코딩
        with torch.no_grad():
            image_features = self.image_encoder(images)
        
        # 초기 토큰은 프롬프트
        generated = prompt_tokens.clone()
        
        # 자기회귀적 생성
        for _ in range(max_length):
            # 현재까지 생성된 토큰으로 특성 추출
            with torch.no_grad():
                text_features = self.text_encoder(generated)
                
                # 멀티모달 특성 결합
                multimodal_features = torch.cat([image_features, text_features], dim=1)
                
                # Perceiver IO 처리
                latent = self.perceiver(multimodal_features)
                
                # 다음 토큰 예측
                decoder_query = self.text_encoder(generated)
                logits = self.text_decoder(decoder_query, latent)
                
                # 마지막 위치의 로짓만 사용
                next_token_logits = logits[:, -1, :] / temperature
                
                # 샘플링
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 생성된 토큰 추가
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated

class ImageEncoder(nn.Module):
    """
    간소화된 Vision Transformer (ViT) 이미지 인코더
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=3, dim=512, depth=6, heads=8, mlp_dim=2048, dropout=0.0):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        
        self.patch_size = patch_size
        
        # 패치 임베딩
        self.patch_embedding = nn.Linear(patch_dim, dim)
        
        # 위치 임베딩
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # 클래스 토큰
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 레이어 정규화
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # 이미지를 패치로 분할
        x = x.reshape(batch_size, channels, height // self.patch_size, self.patch_size, width // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, -1, channels * self.patch_size ** 2)
        
        # 패치 임베딩
        x = self.patch_embedding(x)
        
        # 클래스 토큰 추가
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 위치 임베딩 추가
        x = x + self.pos_embedding
        
        # 트랜스포머 인코더
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, dim] -> [seq_len, batch_size, dim]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, dim] -> [batch_size, seq_len, dim]
        
        # 레이어 정규화
        x = self.norm(x)
        
        return x

class TextEncoder(nn.Module):
    """
    텍스트 인코더
    """
    def __init__(self, vocab_size, dim=512, max_length=77):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, max_length, dim))
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 토큰 임베딩
        token_emb = self.token_embedding(x)
        
        # 위치 임베딩 추가
        position_emb = self.position_embedding[:, :seq_len, :]
        
        # 임베딩 결합
        x = token_emb + position_emb
        
        return x

class TextDecoder(nn.Module):
    """
    텍스트 디코더
    """
    def __init__(self, vocab_size, dim=512):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        
    def forward(self, x, context):
        # 컨텍스트와 크로스 어텐션 (간소화)
        x = x @ context.transpose(1, 2)
        
        # 정규화 및 예측
        x = self.norm(x)
        x = self.head(x)
        
        return x

def train_flamingo(model, dataloader, epochs=10, lr=1e-4, device='cpu'):
    """
    Flamingo 모델 학습 함수
    
    Args:
        model: 학습할 Flamingo 모델
        dataloader: (이미지, 텍스트, 타겟) 데이터 로더
        epochs: 학습 에폭 수
        lr: 학습률
        device: 학습 장치 (CPU 또는 CUDA)
        
    Returns:
        학습된 모델과 학습 손실 기록
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (images, text_tokens, text_targets) in enumerate(dataloader):
            images = images.to(device)
            text_tokens = text_tokens.to(device)
            text_targets = text_targets.to(device)
            
            # 순전파
            optimizer.zero_grad()
            logits = model(images, text_tokens, text_targets)
            
            # 손실 계산 (시퀀스 차원 제외)
            loss = criterion(logits.view(-1, logits.size(-1)), text_targets.view(-1))
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_loss = train_loss / len(dataloader)
        losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
    
    return model, losses

def generate_caption(model, image, prompt_tokens, tokenizer, max_length=50, temperature=1.0, device='cpu'):
    """
    이미지에 대한 캡션 생성
    
    Args:
        model: 학습된 Flamingo 모델
        image: 이미지 텐서 [1, channels, height, width]
        prompt_tokens: 프롬프트 토큰 [1, prompt_len]
        tokenizer: 토크나이저 (디코딩용)
        max_length: 생성할 최대 토큰 수
        temperature: 샘플링 온도
        device: 실행 장치
        
    Returns:
        생성된 캡션 텍스트
    """
    model = model.to(device)
    model.eval()
    
    image = image.to(device)
    prompt_tokens = prompt_tokens.to(device)
    
    # 텍스트 생성
    generated_tokens = model.generate_text(
        images=image,
        prompt_tokens=prompt_tokens,
        max_length=max_length,
        temperature=temperature,
        device=device
    )
    
    # 토큰을 텍스트로 디코딩
    generated_text = tokenizer.decode(generated_tokens[0].cpu().numpy())
    
    return generated_text
