"""
Flickr 데이터셋에 CLIP 모델을 학습하고 시각화하는 예제 스크립트
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchtext.data.utils import get_tokenizer

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 프로젝트 모듈 임포트
from src.models.clip import CLIP, train_clip, image_text_similarity
from src.datasets.multimodal_datasets import FlickrDataset, get_multimodal_dataloader, create_dummy_flickr_dataset
from src.utils.visualization import plot_embeddings, save_visualization
from src.utils.training import Trainer, set_seed, get_device

def main():
    # 재현성을 위한 시드 설정
    set_seed(42)
    
    # 장치 설정
    device = get_device()
    print(f"Using device: {device}")
    
    # 결과 저장 디렉토리 생성
    results_dir = os.path.join("results", "clip_flickr")
    os.makedirs(results_dir, exist_ok=True)
    
    # 더미 Flickr 데이터셋 생성 (실제 데이터셋이 없는 경우)
    print("Creating dummy Flickr dataset...")
    dummy_dir = os.path.join("data", "dummy_flickr")
    dataset_dir, captions_file = create_dummy_flickr_dataset(dummy_dir, num_samples=500)
    
    # 이미지 변환 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 토크나이저 설정
    tokenizer = get_tokenizer("basic_english")
    
    # 데이터셋 생성
    print("Loading Flickr dataset...")
    dataset = FlickrDataset(
        root_dir=dataset_dir,
        captions_file=captions_file,
        transform=transform,
        tokenizer=tokenizer,
        max_length=77
    )
    
    # 데이터셋 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 데이터 로더 생성
    train_loader = get_multimodal_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_loader = get_multimodal_dataloader(val_dataset, batch_size=32, shuffle=False)
    
    # 모델 생성
    vocab_size = len(dataset.vocab)
    embedding_dim = 512
    output_dim = 512
    
    print(f"Creating CLIP model with embedding dimension {output_dim}...")
    model = CLIP(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        output_dim=output_dim
    )
    model = model.to(device)
    
    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 학습 설정
    epochs = 10
    
    # 모델 학습
    print(f"Training CLIP for {epochs} epochs...")
    model, losses = train_clip(
        model=model,
        dataloader=train_loader,
        epochs=epochs,
        lr=1e-4,
        device=device
    )
    
    # 학습 곡선 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CLIP Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, "training_loss.png"), dpi=300)
    plt.close()
    
    # 모델 저장
    torch.save(model.state_dict(), os.path.join(results_dir, "clip_model.pth"))
    
    # 검증 데이터에서 임베딩 추출
    print("Extracting embeddings for visualization...")
    image_embeddings = []
    text_embeddings = []
    
    model.eval()
    with torch.no_grad():
        for images, tokens, _ in val_loader:
            images = images.to(device)
            tokens = tokens.to(device)
            
            # 이미지 및 텍스트 임베딩 추출
            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(tokens)
            
            image_embeddings.append(img_emb.cpu().numpy())
            text_embeddings.append(txt_emb.cpu().numpy())
    
    # 임베딩 결합
    image_embeddings = np.vstack(image_embeddings)
    text_embeddings = np.vstack(text_embeddings)
    
    # 이미지 임베딩 시각화
    print("Visualizing image embeddings...")
    img_emb_vis = plot_embeddings(
        embeddings=image_embeddings,
        method='tsne',
        figsize=(10, 8),
        title="Image Embeddings (t-SNE)"
    )
    save_visualization(img_emb_vis, os.path.join(results_dir, "image_embeddings_tsne.png"))
    
    # 텍스트 임베딩 시각화
    print("Visualizing text embeddings...")
    txt_emb_vis = plot_embeddings(
        embeddings=text_embeddings,
        method='tsne',
        figsize=(10, 8),
        title="Text Embeddings (t-SNE)"
    )
    save_visualization(txt_emb_vis, os.path.join(results_dir, "text_embeddings_tsne.png"))
    
    # 이미지-텍스트 검색 예시
    print("Testing image-text retrieval...")
    
    # 검증 데이터에서 샘플 가져오기
    dataiter = iter(val_loader)
    images, tokens, captions = next(dataiter)
    
    # 첫 번째 이미지와 모든 텍스트 간의 유사도 계산
    query_image = images[0:1].to(device)
    all_tokens = tokens.to(device)
    
    with torch.no_grad():
        similarity = image_text_similarity(model, query_image, all_tokens, device)
    
    # 상위 5개 결과 출력
    top_k = 5
    values, indices = similarity[0].topk(top_k)
    
    # 결과 시각화
    plt.figure(figsize=(12, 4))
    plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())
    plt.title("Query Image")
    plt.axis('off')
    plt.savefig(os.path.join(results_dir, "query_image.png"), dpi=300)
    plt.close()
    
    # 검색 결과 저장
    with open(os.path.join(results_dir, "retrieval_results.txt"), 'w') as f:
        f.write("Query Image\n\n")
        f.write("Top 5 matching captions:\n")
        for i, idx in enumerate(indices):
            f.write(f"{i+1}. {captions[idx]} (Score: {values[idx]:.2f})\n")
    
    print(f"All results saved to {results_dir}")
    print("Done!")

if __name__ == "__main__":
    main()
