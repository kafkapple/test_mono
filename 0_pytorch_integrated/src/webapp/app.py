"""
웹앱 모듈 - Flask 기반 웹 인터페이스
"""
import os
import sys
import json
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
import io
import base64
from PIL import Image

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 프로젝트 모듈 임포트
from src.models.autoencoder import AutoEncoder
from src.models.vae import VAE
from src.models.vq_vae import ConvVQVAE
from src.models.clip import CLIP
from src.models.flamingo import Flamingo
from src.datasets.image_datasets import get_mnist_dataloaders, get_cifar10_dataloaders
from src.datasets.text_datasets import get_imdb_dataloaders
from src.datasets.multimodal_datasets import FlickrDataset, get_multimodal_dataloader
from src.utils.visualization import plot_reconstruction, plot_generated_samples, plot_embeddings
from src.utils.training import get_device

# Flask 앱 생성
app = Flask(__name__)

# 전역 변수
MODELS = {}
DATALOADERS = {}
DEVICE = get_device()

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/models')
def models():
    """모델 목록 페이지"""
    model_info = [
        {
            'id': 'autoencoder',
            'name': 'AutoEncoder',
            'description': '기본 오토인코더 모델'
        },
        {
            'id': 'vae',
            'name': 'VAE',
            'description': '변분 오토인코더 모델'
        },
        {
            'id': 'vqvae',
            'name': 'VQ-VAE',
            'description': 'Vector Quantized VAE 모델'
        },
        {
            'id': 'clip',
            'name': 'CLIP',
            'description': 'Contrastive Language-Image Pre-training 모델'
        },
        {
            'id': 'flamingo',
            'name': 'Flamingo',
            'description': 'Perceiver IO 기반 멀티모달 모델'
        }
    ]
    return render_template('models.html', models=model_info)

@app.route('/datasets')
def datasets():
    """데이터셋 목록 페이지"""
    dataset_info = [
        {
            'id': 'mnist',
            'name': 'MNIST',
            'type': 'image',
            'description': '손글씨 숫자 이미지 데이터셋'
        },
        {
            'id': 'cifar10',
            'name': 'CIFAR-10',
            'type': 'image',
            'description': '10개 클래스 컬러 이미지 데이터셋'
        },
        {
            'id': 'imdb',
            'name': 'IMDB',
            'type': 'text',
            'description': '영화 리뷰 감성 분석 데이터셋'
        },
        {
            'id': 'flickr',
            'name': 'Flickr',
            'type': 'multimodal',
            'description': '이미지-캡션 쌍 데이터셋'
        }
    ]
    return render_template('datasets.html', datasets=dataset_info)

@app.route('/experiments')
def experiments():
    """실험 페이지"""
    return render_template('experiments.html')

@app.route('/api/load_model', methods=['POST'])
def load_model():
    """모델 로드 API"""
    data = request.json
    model_type = data.get('model_type')
    params = data.get('params', {})
    
    try:
        if model_type == 'autoencoder':
            model = AutoEncoder(
                input_dim=params.get('input_dim', 784),
                hidden_dim=params.get('hidden_dim', 256),
                latent_dim=params.get('latent_dim', 64)
            )
        elif model_type == 'vae':
            model = VAE(
                input_dim=params.get('input_dim', 784),
                hidden_dim=params.get('hidden_dim', 256),
                latent_dim=params.get('latent_dim', 20)
            )
        elif model_type == 'vqvae':
            model = ConvVQVAE(
                in_channels=params.get('in_channels', 3),
                hidden_dim=params.get('hidden_dim', 128),
                latent_dim=params.get('latent_dim', 64),
                num_embeddings=params.get('num_embeddings', 512)
            )
        elif model_type == 'clip':
            model = CLIP(
                vocab_size=params.get('vocab_size', 10000),
                embedding_dim=params.get('embedding_dim', 512),
                output_dim=params.get('output_dim', 512)
            )
        elif model_type == 'flamingo':
            model = Flamingo(
                vocab_size=params.get('vocab_size', 10000),
                embedding_dim=params.get('embedding_dim', 512),
                num_layers=params.get('num_layers', 4),
                num_heads=params.get('num_heads', 8),
                output_dim=params.get('output_dim', 512)
            )
        else:
            return jsonify({'success': False, 'error': f'Unknown model type: {model_type}'})
        
        # 모델을 장치로 이동
        model = model.to(DEVICE)
        
        # 모델 저장
        model_id = f"{model_type}_{len(MODELS)}"
        MODELS[model_id] = model
        
        return jsonify({
            'success': True,
            'model_id': model_id,
            'message': f'{model_type} model loaded successfully'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/load_dataset', methods=['POST'])
def load_dataset():
    """데이터셋 로드 API"""
    data = request.json
    dataset_type = data.get('dataset_type')
    params = data.get('params', {})
    
    try:
        if dataset_type == 'mnist':
            train_loader, test_loader = get_mnist_dataloaders(
                batch_size=params.get('batch_size', 32),
                root=params.get('root', './data')
            )
            dataloader = test_loader  # 시각화용으로 테스트 데이터 사용
        elif dataset_type == 'cifar10':
            train_loader, test_loader = get_cifar10_dataloaders(
                batch_size=params.get('batch_size', 32),
                root=params.get('root', './data')
            )
            dataloader = test_loader  # 시각화용으로 테스트 데이터 사용
        elif dataset_type == 'imdb':
            train_loader, test_loader, vocab = get_imdb_dataloaders(
                batch_size=params.get('batch_size', 32),
                root=params.get('root', './data'),
                max_length=params.get('max_length', 256)
            )
            dataloader = test_loader  # 시각화용으로 테스트 데이터 사용
        elif dataset_type == 'flickr':
            # 더미 데이터셋 사용 (실제 데이터셋이 없는 경우)
            from src.datasets.multimodal_datasets import create_dummy_flickr_dataset
            dummy_dir = os.path.join("data", "dummy_flickr")
            dataset_dir, captions_file = create_dummy_flickr_dataset(dummy_dir, num_samples=100)
            
            from torchvision import transforms
            from torchtext.data.utils import get_tokenizer
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            
            tokenizer = get_tokenizer("basic_english")
            
            dataset = FlickrDataset(
                root_dir=dataset_dir,
                captions_file=captions_file,
                transform=transform,
                tokenizer=tokenizer,
                max_length=params.get('max_length', 77)
            )
            
            dataloader = get_multimodal_dataloader(
                dataset,
                batch_size=params.get('batch_size', 32),
                shuffle=False
            )
        else:
            return jsonify({'success': False, 'error': f'Unknown dataset type: {dataset_type}'})
        
        # 데이터로더 저장
        dataloader_id = f"{dataset_type}_{len(DATALOADERS)}"
        DATALOADERS[dataloader_id] = dataloader
        
        return jsonify({
            'success': True,
            'dataloader_id': dataloader_id,
            'message': f'{dataset_type} dataset loaded successfully'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/visualize_dataset', methods=['POST'])
def visualize_dataset():
    """데이터셋 시각화 API"""
    data = request.json
    dataloader_id = data.get('dataloader_id')
    
    if dataloader_id not in DATALOADERS:
        return jsonify({'success': False, 'error': f'Dataloader not found: {dataloader_id}'})
    
    try:
        dataloader = DATALOADERS[dataloader_id]
        
        # 데이터셋 유형 확인
        if 'mnist' in dataloader_id or 'cifar' in dataloader_id:
            # 이미지 데이터셋
            from src.datasets.image_datasets import visualize_image_batch
            vis_img = visualize_image_batch(dataloader, num_images=16)
            
            # 이미지를 base64로 인코딩
            img_pil = Image.fromarray(vis_img)
            img_buffer = io.BytesIO()
            img_pil.save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_str}',
                'message': 'Dataset visualization generated'
            })
        
        elif 'imdb' in dataloader_id:
            # 텍스트 데이터셋
            from src.datasets.text_datasets import visualize_text_batch
            vocab = None  # 실제 구현에서는 vocab 객체 필요
            text_samples = visualize_text_batch(dataloader, vocab, num_samples=5)
            
            return jsonify({
                'success': True,
                'text_samples': text_samples,
                'message': 'Text dataset visualization generated'
            })
        
        elif 'flickr' in dataloader_id:
            # 멀티모달 데이터셋
            from src.datasets.multimodal_datasets import visualize_multimodal_batch
            vis_img = visualize_multimodal_batch(dataloader, num_samples=5)
            
            # 이미지를 base64로 인코딩
            img_pil = Image.fromarray(vis_img)
            img_buffer = io.BytesIO()
            img_pil.save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_str}',
                'message': 'Multimodal dataset visualization generated'
            })
        
        else:
            return jsonify({'success': False, 'error': 'Unsupported dataset type'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/run_experiment', methods=['POST'])
def run_experiment():
    """실험 실행 API"""
    data = request.json
    model_id = data.get('model_id')
    dataloader_id = data.get('dataloader_id')
    experiment_type = data.get('experiment_type')
    
    if model_id not in MODELS:
        return jsonify({'success': False, 'error': f'Model not found: {model_id}'})
    
    if dataloader_id not in DATALOADERS:
        return jsonify({'success': False, 'error': f'Dataloader not found: {dataloader_id}'})
    
    try:
        model = MODELS[model_id]
        dataloader = DATALOADERS[dataloader_id]
        
        if experiment_type == 'reconstruction':
            # 재구성 실험
            model.eval()
            with torch.no_grad():
                # 배치 가져오기
                batch = next(iter(dataloader))
                
                if isinstance(batch, tuple) and len(batch) >= 2:
                    data = batch[0].to(DEVICE)
                else:
                    data = batch.to(DEVICE)
                
                # 모델 유형에 따라 재구성
                if 'VAE' in model.__class__.__name__:
                    reconstructions, _, _ = model(data)
                elif 'VQVAE' in model.__class__.__name__:
                    reconstructions, _, _ = model(data)
                else:
                    reconstructions = model(data)
                
                # 재구성 시각화
                vis_img = plot_reconstruction(
                    original=data[:10],
                    reconstruction=reconstructions[:10],
                    n_samples=10,
                    figsize=(12, 6)
                )
                
                # 이미지를 base64로 인코딩
                img_pil = Image.fromarray(vis_img)
                img_buffer = io.BytesIO()
                img_pil.save(img_buffer, format='PNG')
                img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                
                return jsonify({
                    'success': True,
                    'image': f'data:image/png;base64,{img_str}',
                    'message': 'Reconstruction experiment completed'
                })
        
        elif experiment_type == 'generation':
            # 생성 실험
            model.eval()
            with torch.no_grad():
                # 모델 유형에 따라 샘플 생성
                if 'VAE' in model.__class__.__name__:
                    samples = model.sample(num_samples=25, device=DEVICE)
                    if len(samples.shape) == 2:
                        # 선형 VAE의 경우 이미지 형태로 변환
                        if 'mnist' in dataloader_id:
                            samples = samples.view(-1, 1, 28, 28)
                        else:
                            samples = samples.view(-1, 3, 32, 32)
                elif 'VQVAE' in model.__class__.__name__:
                    # VQ-VAE의 경우 코드북에서 샘플링
                    num_samples = 25
                    latent_dim = model.latent_dim
                    num_embeddings = model.vq.num_embeddings
                    
                    # 무작위 코드북 인덱스 생성
                    random_indices = torch.randint(0, num_embeddings, (num_samples, 1), device=DEVICE)
                    
                    # 인덱스에서 임베딩 가져오기
                    embeddings = model.vq.embedding(random_indices).squeeze(1)
                    
                    # 임베딩 형태 변환
                    batch_size = embeddings.size(0)
                    spatial_size = 8  # 32x32 이미지의 경우 4번 다운샘플링하면 8x8
                    embeddings = embeddings.view(batch_size, latent_dim, 1, 1).expand(batch_size, latent_dim, spatial_size, spatial_size)
                    
                    # 디코딩
                    samples = model.decoder(embeddings)
                else:
                    return jsonify({'success': False, 'error': 'Model does not support generation'})
                
                # 생성된 샘플 시각화
                vis_img = plot_generated_samples(
                    samples=samples,
                    nrow=5,
                    ncol=5,
                    figsize=(10, 10),
                    title="Generated Samples"
                )
                
                # 이미지를 base64로 인코딩
                img_pil = Image.fromarray(vis_img)
                img_buffer = io.BytesIO()
                img_pil.save(img_buffer, format='PNG')
                img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                
                return jsonify({
                    'success': True,
                    'image': f'data:image/png;base64,{img_str}',
                    'message': 'Generation experiment completed'
                })
        
        elif experiment_type == 'embedding':
            # 임베딩 시각화 실험
            from src.utils.visualization import extract_embeddings
            
            # 임베딩 추출
            embeddings = extract_embeddings(model, dataloader, device=DEVICE)
            
            if isinstance(embeddings, tuple) and len(embeddings) == 2:
                embeddings, labels = embeddings
            else:
                labels = None
            
            # 임베딩 시각화
            vis_img = plot_embeddings(
                embeddings=embeddings,
                labels=labels,
                method='tsne',
                figsize=(10, 8),
                title="Embeddings (t-SNE)"
            )
            
            # 이미지를 base64로 인코딩
            img_pil = Image.fromarray(vis_img)
            img_buffer = io.BytesIO()
            img_pil.save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_str}',
                'message': 'Embedding visualization completed'
            })
        
        else:
            return jsonify({'success': False, 'error': f'Unknown experiment type: {experiment_type}'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/run_script', methods=['POST'])
def run_script():
    """파이썬 스크립트 실행 API"""
    data = request.json
    script_type = data.get('script_type')
    
    try:
        if script_type == 'vae_mnist':
            # VAE MNIST 스크립트 실행
            import subprocess
            result = subprocess.run(
                ['python', 'examples/run_vae_mnist.py'],
                capture_output=True,
                text=True
            )
            
            return jsonify({
                'success': True,
                'output': result.stdout,
                'error': result.stderr,
                'message': 'VAE MNIST script executed'
            })
        
        elif script_type == 'vqvae_cifar10':
            # VQ-VAE CIFAR-10 스크립트 실행
            import subprocess
            result = subprocess.run(
                ['python', 'examples/run_vqvae_cifar10.py'],
                capture_output=True,
                text=True
            )
            
            return jsonify({
                'success': True,
                'output': result.stdout,
                'error': result.stderr,
                'message': 'VQ-VAE CIFAR-10 script executed'
            })
        
        elif script_type == 'clip_flickr':
            # CLIP Flickr 스크립트 실행
            import subprocess
            result = subprocess.run(
                ['python', 'examples/run_clip_flickr.py'],
                capture_output=True,
                text=True
            )
            
            return jsonify({
                'success': True,
                'output': result.stdout,
                'error': result.stderr,
                'message': 'CLIP Flickr script executed'
            })
        
        else:
            return jsonify({'success': False, 'error': f'Unknown script type: {script_type}'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def main():
    """웹앱 실행"""
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
