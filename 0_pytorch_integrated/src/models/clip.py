"""
CLIP (Contrastive Language-Image Pre-training) 모델 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    """
    CLIP의 텍스트 인코더
    텍스트를 임베딩 공간으로 인코딩
    """
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=512, output_dim=512, max_length=77):
        super(TextEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(max_length, embedding_dim))
        
        # 간소화된 트랜스포머 인코더
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=6
        )
        
        # 최종 프로젝션
        self.ln_final = nn.LayerNorm(embedding_dim)
        self.projection = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, text_tokens):
        # 토큰 임베딩 + 위치 임베딩
        x = self.token_embedding(text_tokens) + self.positional_embedding[:text_tokens.size(1)]
        
        # 트랜스포머 인코더
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, dim] -> [seq_len, batch_size, dim]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, dim] -> [batch_size, seq_len, dim]
        
        # 최종 레이어 정규화 및 프로젝션
        x = self.ln_final(x)
        
        # [CLS] 토큰 (첫 번째 토큰)의 임베딩 사용
        x = x[:, 0]
        x = self.projection(x)
        
        return x

class ImageEncoder(nn.Module):
    """
    CLIP의 이미지 인코더
    이미지를 임베딩 공간으로 인코딩
    """
    def __init__(self, input_channels=3, output_dim=512):
        super(ImageEncoder, self).__init__()
        
        # 간소화된 비전 트랜스포머 (ViT)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 간소화된 ResNet 블록
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(512, output_dim)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # 첫 번째 블록은 다운샘플링 가능
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # 나머지 블록
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.projection(x)
        
        return x

class CLIP(nn.Module):
    """
    CLIP (Contrastive Language-Image Pre-training) 모델
    텍스트와 이미지를 공통 임베딩 공간으로 매핑
    """
    def __init__(self, vocab_size, embedding_dim=512, output_dim=512, temperature=0.07):
        super(CLIP, self).__init__()
        
        self.text_encoder = TextEncoder(vocab_size, embedding_dim, output_dim=output_dim)
        self.image_encoder = ImageEncoder(output_dim=output_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        
    def forward(self, images, text_tokens):
        # 이미지 및 텍스트 인코딩
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(text_tokens)
        
        # 특성 정규화
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        # 로짓 스케일링
        logit_scale = self.logit_scale.exp()
        
        # 이미지-텍스트 유사도 행렬
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    
    def encode_image(self, images):
        """이미지를 임베딩 공간으로 인코딩"""
        image_features = self.image_encoder(images)
        image_features = F.normalize(image_features, dim=1)
        return image_features
    
    def encode_text(self, text_tokens):
        """텍스트를 임베딩 공간으로 인코딩"""
        text_features = self.text_encoder(text_tokens)
        text_features = F.normalize(text_features, dim=1)
        return text_features

def clip_loss(logits_per_image, logits_per_text):
    """
    CLIP 대조 손실 함수
    
    Args:
        logits_per_image: 이미지에 대한 텍스트 유사도 로짓
        logits_per_text: 텍스트에 대한 이미지 유사도 로짓
        
    Returns:
        총 대조 손실
    """
    batch_size = logits_per_image.shape[0]
    
    # 대각선 요소가 정답 (각 이미지는 해당 텍스트와 매칭)
    labels = torch.arange(batch_size, device=logits_per_image.device)
    
    # 이미지->텍스트 방향 손실
    loss_i = F.cross_entropy(logits_per_image, labels)
    
    # 텍스트->이미지 방향 손실
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    # 총 손실
    total_loss = (loss_i + loss_t) / 2
    
    return total_loss

def train_clip(model, dataloader, epochs=10, lr=1e-4, device='cpu'):
    """
    CLIP 모델 학습 함수
    
    Args:
        model: 학습할 CLIP 모델
        dataloader: (이미지, 텍스트) 쌍의 데이터 로더
        epochs: 학습 에폭 수
        lr: 학습률
        device: 학습 장치 (CPU 또는 CUDA)
        
    Returns:
        학습된 모델과 학습 손실 기록
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (images, text_tokens) in enumerate(dataloader):
            images = images.to(device)
            text_tokens = text_tokens.to(device)
            
            # 순전파
            optimizer.zero_grad()
            logits_per_image, logits_per_text = model(images, text_tokens)
            
            # 손실 계산
            loss = clip_loss(logits_per_image, logits_per_text)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_loss = train_loss / len(dataloader)
        losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
    
    return model, losses

def image_text_similarity(model, images, texts, device='cpu'):
    """
    이미지와 텍스트 간의 유사도 계산
    
    Args:
        model: 학습된 CLIP 모델
        images: 이미지 텐서 [batch_size, channels, height, width]
        texts: 텍스트 토큰 텐서 [batch_size, seq_len]
        device: 실행 장치 (CPU 또는 CUDA)
        
    Returns:
        이미지-텍스트 유사도 행렬
    """
    model = model.to(device)
    model.eval()
    
    images = images.to(device)
    texts = texts.to(device)
    
    with torch.no_grad():
        # 이미지 및 텍스트 인코딩
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        
        # 로짓 스케일링
        logit_scale = model.logit_scale.exp()
        
        # 이미지-텍스트 유사도 행렬
        similarity = logit_scale * image_features @ text_features.t()
    
    return similarity

def retrieve_images_from_text(model, text_query, image_dataset, top_k=5, device='cpu'):
    """
    텍스트 쿼리로 이미지 검색
    
    Args:
        model: 학습된 CLIP 모델
        text_query: 텍스트 쿼리 토큰 [1, seq_len]
        image_dataset: 이미지 데이터셋
        top_k: 반환할 상위 결과 수
        device: 실행 장치 (CPU 또는 CUDA)
        
    Returns:
        상위 k개 이미지 및 유사도 점수
    """
    model = model.to(device)
    model.eval()
    
    text_query = text_query.to(device)
    
    with torch.no_grad():
        # 텍스트 쿼리 인코딩
        text_features = model.encode_text(text_query)
        
        # 모든 이미지에 대한 유사도 계산
        similarities = []
        images = []
        
        for idx, image in enumerate(image_dataset):
            if isinstance(image, tuple):
                image = image[0]  # (image, label) 형태인 경우
            
            image = image.unsqueeze(0).to(device)  # [1, channels, height, width]
            image_features = model.encode_image(image)
            
            # 유사도 계산
            similarity = (100.0 * image_features @ text_features.t()).item()
            similarities.append((idx, similarity))
            images.append(image)
        
        # 유사도 기준 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k개 결과 반환
        top_indices = [idx for idx, _ in similarities[:top_k]]
        top_scores = [score for _, score in similarities[:top_k]]
        top_images = [images[idx] for idx in top_indices]
    
    return top_images, top_scores, top_indices
