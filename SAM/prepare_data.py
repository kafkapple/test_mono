import os
import cv2
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

log = logging.getLogger(__name__)

def download_sample_video(url, output_path):
    """
    샘플 비디오를 다운로드합니다.
    
    Args:
        url (str): 다운로드할 비디오 URL
        output_path (str): 저장할 경로
    """
    import requests
    
    log.info(f"Downloading sample video from {url} to {output_path}")
    
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 파일이 이미 존재하는지 확인
    if os.path.exists(output_path):
        log.info(f"File already exists at {output_path}")
        return
    
    # 비디오 다운로드
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # 파일 저장
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    log.info(f"Successfully downloaded video to {output_path}")

def preprocess_video(input_path, output_path, config):
    """
    비디오를 전처리합니다.
    
    Args:
        input_path (str): 입력 비디오 경로
        output_path (str): 출력 비디오 경로
        config (DictConfig): 전처리 설정
    """
    log.info(f"Preprocessing video from {input_path} to {output_path}")
    
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(input_path)
    
    # 비디오 속성 확인
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    log.info(f"Original video: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # 리사이즈 설정 적용
    if config.preprocessing.resize:
        width = config.preprocessing.width
        height = config.preprocessing.height
    
    # FPS 설정 적용
    if config.preprocessing.fps:
        fps = config.preprocessing.fps
    
    # 최대 프레임 수 설정 적용
    max_frames = total_frames
    if config.preprocessing.max_frames:
        max_frames = min(config.preprocessing.max_frames, total_frames)
    
    # 비디오 작성자 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 프레임 처리
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 리사이즈 적용
        if config.preprocessing.resize:
            frame = cv2.resize(frame, (width, height))
        
        # 데이터 증강 적용
        if config.augmentation.enabled:
            # 밝기 조정
            if config.augmentation.brightness:
                brightness = 1.0 + (2 * config.augmentation.brightness * (0.5 - 0.5))
                frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)
            
            # 대비 조정
            if config.augmentation.contrast:
                contrast = 1.0 + (2 * config.augmentation.contrast * (0.5 - 0.5))
                frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)
        
        # 프레임 저장
        out.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            log.info(f"Processed {frame_count}/{max_frames} frames")
    
    # 자원 해제
    cap.release()
    out.release()
    
    log.info(f"Successfully preprocessed video to {output_path}")
    log.info(f"Processed video: {width}x{height}, {fps} fps, {frame_count} frames")

@hydra.main(config_path="../configs", config_name="config")
def main(config: DictConfig):
    """
    샘플 비디오 데이터를 준비하고 전처리합니다.
    
    Args:
        config (DictConfig): Hydra 설정
    """
    log.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # 샘플 비디오 URL (Pexels에서 무료 비디오 사용)
    sample_video_url = "https://www.pexels.com/download/video/1721294/?fps=25.0&h=720&w=1280"
    
    # 원본 비디오 경로
    raw_video_path = os.path.join(config.hydra.runtime.cwd, "data/raw/sample_video.mp4")
    
    # 전처리된 비디오 경로
    processed_video_path = os.path.join(config.save.processed_dir, "sample_video_processed.mp4")
    
    # 샘플 비디오 다운로드
    download_sample_video(sample_video_url, raw_video_path)
    
    # 비디오 전처리
    preprocess_video(raw_video_path, processed_video_path, config)
    
    log.info("Sample video preparation completed")

if __name__ == "__main__":
    main()
