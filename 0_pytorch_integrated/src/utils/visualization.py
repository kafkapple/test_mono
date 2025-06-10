"""
시각화 및 임베딩 유틸리티 모듈
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import seaborn as sns
import os
from PIL import Image
import io

def extract_embeddings(model, dataloader, device='cpu'):
    """
    모델에서 임베딩 추출
    
    Args:
        model: 임베딩을 추출할 모델
        dataloader: 데이터 로더
        device: 실행 장치 (CPU 또는 CUDA)
        
    Returns:
        임베딩 텐서와 레이블 (있는 경우)
    """
    model = model.to(device)
    model.eval()
    
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 배치 데이터 처리
            if isinstance(batch, tuple) and len(batch) >= 2:
                data = batch[0].to(device)
                label = batch[1]
                has_labels = True
            else:
                data = batch.to(device)
                has_labels = False
            
            # 모델에 따라 임베딩 추출 방식 결정
            if hasattr(model, 'encode'):
                # VAE, AE 등의 인코더 사용
                if isinstance(model, nn.Module) and hasattr(model, 'encode'):
                    if 'VAE' in model.__class__.__name__:
                        # VAE의 경우 평균 벡터 사용
                        mu, _ = model.encode(data)
                        embedding = mu
                    else:
                        # 일반 AE의 경우 인코더 출력 사용
                        embedding = model.encode(data)
                else:
                    # 기본 추출 방식
                    embedding = model(data)
            elif hasattr(model, 'encoder'):
                # 인코더 속성이 있는 경우
                embedding = model.encoder(data)
            else:
                # 기본 추출 방식
                embedding = model(data)
            
            # 임베딩 수집
            embeddings.append(embedding.cpu().numpy())
            
            # 레이블 수집 (있는 경우)
            if has_labels:
                labels.append(label.cpu().numpy())
    
    # 임베딩 결합
    embeddings = np.vstack(embeddings)
    
    # 레이블 결합 (있는 경우)
    if labels:
        labels = np.concatenate(labels)
        return embeddings, labels
    
    return embeddings

def reduce_dimensions(embeddings, method='tsne', n_components=2, **kwargs):
    """
    임베딩 차원 축소
    
    Args:
        embeddings: 임베딩 배열 [n_samples, n_features]
        method: 차원 축소 방법 ('tsne', 'pca', 'umap')
        n_components: 축소할 차원 수
        **kwargs: 차원 축소 알고리즘에 전달할 추가 인자
        
    Returns:
        축소된 임베딩 [n_samples, n_components]
    """
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, **kwargs)
    elif method == 'pca':
        reducer = PCA(n_components=n_components, **kwargs)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, **kwargs)
    else:
        raise ValueError(f"지원되지 않는 차원 축소 방법: {method}")
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

def cluster_embeddings(embeddings, method='kmeans', n_clusters=5, **kwargs):
    """
    임베딩 클러스터링
    
    Args:
        embeddings: 임베딩 배열 [n_samples, n_features]
        method: 클러스터링 방법 ('kmeans', 'dbscan')
        n_clusters: 클러스터 수 (KMeans에만 적용)
        **kwargs: 클러스터링 알고리즘에 전달할 추가 인자
        
    Returns:
        클러스터 레이블 [n_samples]
    """
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, **kwargs)
    elif method == 'dbscan':
        clusterer = DBSCAN(**kwargs)
    else:
        raise ValueError(f"지원되지 않는 클러스터링 방법: {method}")
    
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # 실루엣 점수 계산 (클러스터가 2개 이상이고 모든 샘플이 클러스터에 할당된 경우)
    if len(np.unique(cluster_labels)) >= 2 and -1 not in cluster_labels:
        silhouette = silhouette_score(embeddings, cluster_labels)
        print(f"Silhouette Score: {silhouette:.4f}")
    
    return cluster_labels

def plot_embeddings(embeddings, labels=None, method='tsne', n_components=2, figsize=(10, 8), save_path=None, title=None, **kwargs):
    """
    임베딩 시각화
    
    Args:
        embeddings: 임베딩 배열 [n_samples, n_features]
        labels: 레이블 또는 클러스터 [n_samples] (선택 사항)
        method: 차원 축소 방법 ('tsne', 'pca', 'umap')
        n_components: 축소할 차원 수 (2 또는 3)
        figsize: 그림 크기
        save_path: 저장 경로 (선택 사항)
        title: 그림 제목 (선택 사항)
        **kwargs: 차원 축소 알고리즘에 전달할 추가 인자
        
    Returns:
        시각화 이미지 (numpy 배열)
    """
    # 차원 축소
    reduced_embeddings = reduce_dimensions(embeddings, method, n_components, **kwargs)
    
    # 그림 생성
    fig = plt.figure(figsize=figsize)
    
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            scatter = ax.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                reduced_embeddings[:, 2],
                c=labels,
                cmap='tab10',
                alpha=0.8
            )
            legend = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend)
        else:
            ax.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                reduced_embeddings[:, 2],
                alpha=0.8
            )
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
    else:
        ax = fig.add_subplot(111)
        
        if labels is not None:
            scatter = ax.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                c=labels,
                cmap='tab10',
                alpha=0.8
            )
            legend = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend)
        else:
            ax.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                alpha=0.8
            )
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
    
    # 제목 설정
    if title:
        plt.title(title)
    else:
        plt.title(f'{method.upper()} Visualization')
    
    plt.tight_layout()
    
    # 저장 (요청된 경우)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 그림을 numpy 배열로 변환
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # ARGB 형식
    img = img[:, :, 1:]  # ARGB에서 RGB만 사용
    
    plt.close(fig)
    
    return img

def plot_reconstruction(original, reconstruction, n_samples=10, figsize=(12, 6)):
    """
    원본 이미지와 재구성된 이미지를 시각화
    
    Args:
        original: 원본 이미지 텐서 (B, C*H*W) 또는 (B, C, H, W)
        reconstruction: 재구성된 이미지 텐서 (B, C*H*W) 또는 (B, C, H, W)
        n_samples: 시각화할 샘플 수
        figsize: 그래프 크기
    """
    # 입력 이미지 형태 확인 및 변환
    if len(original.shape) == 2:  # 평탄화된 이미지 (B, C*H*W)
        # 이미지 크기 추정 (MNIST의 경우 784 = 28*28*1)
        img_size = int(np.sqrt(original.shape[1]))
        batch_size = original.shape[0]
        # 이미지 형태로 변환 (B, 1, H, W)
        original = original.view(batch_size, 1, img_size, img_size)
        reconstruction = reconstruction.view(batch_size, 1, img_size, img_size)
    elif len(original.shape) == 3:  # (B, H, W) 형태
        # 채널 차원 추가 (B, 1, H, W)
        original = original.unsqueeze(1)
        reconstruction = reconstruction.unsqueeze(1)
    
    # 시각화할 샘플 수 조정
    n_samples = min(n_samples, original.shape[0])
    
    # 그래프 생성
    fig, axes = plt.subplots(2, n_samples, figsize=figsize)
    
    # 원본 및 재구성 이미지 시각화
    for i in range(n_samples):
        # 원본 이미지
        orig_img = original[i].detach().cpu().numpy()
        if len(orig_img.shape) == 3:  # (C, H, W)
            orig_img = np.transpose(orig_img, (1, 2, 0))  # (H, W, C)
        axes[0, i].imshow(orig_img.squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # 재구성 이미지
        recon_img = reconstruction[i].detach().cpu().numpy()
        if len(recon_img.shape) == 3:  # (C, H, W)
            recon_img = np.transpose(recon_img, (1, 2, 0))  # (H, W, C)
        axes[1, i].imshow(recon_img.squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstruction')
    
    plt.tight_layout()
    return fig

def plot_generated_samples(samples, nrow=5, ncol=5, figsize=(10, 10), title=None):
    """
    생성된 샘플 시각화
    
    Args:
        samples: 생성된 샘플 배치 [batch_size, channels, height, width]
        nrow: 행 수
        ncol: 열 수
        figsize: 그림 크기
        title: 그림 제목 (선택 사항)
        
    Returns:
        matplotlib Figure 객체
    """
    # 샘플 수 제한
    n_samples = min(nrow * ncol, samples.shape[0])
    samples = samples[:n_samples]
    
    # 그림 생성
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            # 이미지 변환
            img = samples[i].detach().cpu().numpy()
            if len(img.shape) == 3:  # (C, H, W)
                img = np.transpose(img, (1, 2, 0))  # (H, W, C)
            ax.imshow(img.squeeze(), cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
    
    # 제목 설정
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    return fig

def plot_latent_space_2d(model, n_samples=20, figsize=(10, 10), save_path=None):
    """
    2D 잠재 공간 시각화 (VAE 또는 AE 모델용)
    
    Args:
        model: VAE 또는 AE 모델
        n_samples: 각 차원당 샘플 수
        figsize: 그림 크기
        save_path: 저장 경로 (선택 사항)
        
    Returns:
        시각화 이미지 (numpy 배열)
    """
    # 모델을 평가 모드로 설정
    model.eval()
    
    # 2D 그리드 생성
    x = np.linspace(-3, 3, n_samples)
    y = np.linspace(-3, 3, n_samples)
    xv, yv = np.meshgrid(x, y)
    
    # 그림 생성
    fig, axes = plt.subplots(n_samples, n_samples, figsize=figsize)
    
    with torch.no_grad():
        for i, yi in enumerate(y):
            for j, xi in enumerate(x):
                # 잠재 벡터 생성
                z = torch.zeros(1, 2)
                z[0, 0] = xi
                z[0, 1] = yi
                
                # 이미지 디코딩
                decoded = model.decode(z)
                
                # 이미지 변환
                img = decoded[0].detach().cpu().numpy().transpose(1, 2, 0)
                
                # 채널 수에 따라 처리
                if img.shape[2] == 1:
                    img = img.squeeze(2)
                    axes[i, j].imshow(img, cmap='gray')
                else:
                    # 이미지 정규화 해제 (가정: [-1, 1] 범위)
                    img = (img + 1) / 2
                    axes[i, j].imshow(img.clip(0, 1))
                
                axes[i, j].axis('off')
    
    plt.suptitle('Latent Space Visualization', fontsize=16)
    plt.tight_layout()
    
    # 저장 (요청된 경우)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 그림을 numpy 배열로 변환
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # ARGB 형식
    img = img[:, :, 1:]  # ARGB에서 RGB만 사용
    
    plt.close(fig)
    
    return img

def plot_training_curves(losses, figsize=(10, 6), save_path=None):
    """
    학습 곡선 시각화
    
    Args:
        losses: 손실 딕셔너리 또는 리스트
        figsize: 그림 크기
        save_path: 저장 경로 (선택 사항)
        
    Returns:
        시각화 이미지 (numpy 배열)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 손실 유형에 따라 처리
    if isinstance(losses, dict):
        for name, values in losses.items():
            ax.plot(values, label=name)
    else:
        ax.plot(losses, label='Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장 (요청된 경우)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 그림을 numpy 배열로 변환
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # ARGB 형식
    img = img[:, :, 1:]  # ARGB에서 RGB만 사용
    
    plt.close(fig)
    
    return img

def save_visualization(img, save_path):
    """
    시각화 이미지 저장
    
    Args:
        img: numpy 배열 이미지 또는 matplotlib Figure 객체
        save_path: 저장 경로
        
    Returns:
        저장된 파일 경로
    """
    # 디렉토리 생성 (필요한 경우)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # Figure 객체인 경우
    if isinstance(img, plt.Figure):
        img.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(img)
    else:
        # numpy 배열인 경우
        Image.fromarray(img).save(save_path)
    
    return save_path

def get_visualization_as_bytes(img, format='png'):
    """
    시각화 이미지를 바이트로 변환
    
    Args:
        img: numpy 배열 이미지
        format: 이미지 형식 ('png', 'jpg', 등)
        
    Returns:
        이미지 바이트
    """
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format=format)
    return buf.getvalue()
