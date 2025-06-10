import os
import cv2
import torch
import numpy as np
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# SAM2 모델 임포트를 위한 자리 표시자
# 실제 구현 시 SAM2 모델 라이브러리 임포트 필요
# from segment_anything_2 import sam_model_registry, SamPredictor

log = logging.getLogger(__name__)

class SAM2Segmenter:
    """SAM2 모델을 사용한 세그멘테이션 클래스"""
    
    def __init__(self, config: DictConfig):
        """
        SAM2 세그멘테이션 모델 초기화
        
        Args:
            config (DictConfig): 모델 설정
        """
        self.config = config
        self.device = torch.device(config.general.device if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")
        
        # 모델 체크포인트 경로
        checkpoint_path = os.path.join(config.hydra.runtime.cwd, "models", config.models.checkpoint)
        
        # 모델 초기화 (실제 구현 시 SAM2 모델 로드 코드로 대체)
        log.info(f"Initializing SAM2 model: {config.models.model_type} from {checkpoint_path}")
        # self.sam = sam_model_registry[config.models.model_type](checkpoint=checkpoint_path)
        # self.sam.to(self.device)
        # self.predictor = SamPredictor(self.sam)
        
        # 메모리 모듈 초기화 (비디오 처리용)
        self.memory_frames = []
        self.memory_masks = []
        self.memory_size = config.models.video.memory_size
        
        log.info("SAM2 model initialized successfully")
    
    def generate_prompts(self, image: np.ndarray) -> Dict[str, Any]:
        """
        이미지에 대한 프롬프트 생성
        
        Args:
            image (np.ndarray): 입력 이미지
            
        Returns:
            Dict[str, Any]: 프롬프트 정보
        """
        prompt_type = self.config.models.prompts.type
        prompts = {}
        
        if prompt_type == "auto":
            # 자동 프롬프트 생성 (그리드 또는 중앙 포인트)
            h, w = image.shape[:2]
            if self.config.models.prompts.auto_prompt_mode == "center":
                # 이미지 중앙에 포인트 생성
                prompts["point_coords"] = np.array([[w//2, h//2]])
                prompts["point_labels"] = np.array([1])
            else:  # grid
                # 그리드 포인트 생성
                points_per_side = self.config.models.parameters.points_per_side
                points_x = np.linspace(0, w, points_per_side + 2)[1:-1]
                points_y = np.linspace(0, h, points_per_side + 2)[1:-1]
                points = []
                for x in points_x:
                    for y in points_y:
                        points.append([x, y])
                prompts["point_coords"] = np.array(points)
                prompts["point_labels"] = np.ones(len(points))
        elif prompt_type == "point":
            # 수동 포인트 프롬프트 사용
            prompts["point_coords"] = np.array(self.config.models.prompts.point_coords)
            prompts["point_labels"] = np.array(self.config.models.prompts.point_labels)
        elif prompt_type == "box":
            # 수동 박스 프롬프트 사용
            prompts["box"] = np.array(self.config.models.prompts.box)
        elif prompt_type == "text":
            # 텍스트 프롬프트 사용 (Grounding SAM 등에서 지원)
            prompts["text"] = self.config.models.prompts.text
        
        return prompts
    
    def segment_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        단일 이미지에 대한 세그멘테이션 수행
        
        Args:
            image (np.ndarray): 입력 이미지 (BGR 형식)
            
        Returns:
            List[Dict[str, Any]]: 세그멘테이션 결과 리스트
        """
        # RGB로 변환 (SAM2는 RGB 입력 예상)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 프롬프트 생성
        prompts = self.generate_prompts(rgb_image)
        
        # 실제 구현 시 SAM2 모델 예측 코드로 대체
        # self.predictor.set_image(rgb_image)
        # masks, scores, logits = self.predictor.predict(**prompts)
        
        # 임시 더미 결과 생성 (실제 구현 시 제거)
        h, w = image.shape[:2]
        dummy_masks = [
            np.zeros((h, w), dtype=bool),
            np.zeros((h, w), dtype=bool)
        ]
        # 첫 번째 마스크에 원 그리기
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 4
        for y in range(h):
            for x in range(w):
                if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                    dummy_masks[0][y, x] = True
        
        # 두 번째 마스크에 사각형 그리기
        x1, y1 = w // 4, h // 4
        x2, y2 = 3 * w // 4, 3 * h // 4
        dummy_masks[1][y1:y2, x1:x2] = True
        
        dummy_scores = [0.95, 0.85]
        
        # 결과 형식화
        results = []
        for i, (mask, score) in enumerate(zip(dummy_masks, dummy_scores)):
            # 마스크에서 바운딩 박스 계산
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                x1, x2 = np.min(x_indices), np.max(x_indices)
                y1, y2 = np.min(y_indices), np.max(y_indices)
                bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h] 형식
            else:
                bbox = [0, 0, 0, 0]
            
            # 마스크 면적 계산
            area = np.sum(mask)
            
            # 최소 마스크 영역 필터링
            if area >= self.config.models.parameters.min_mask_region_area:
                results.append({
                    "id": i,
                    "mask": mask,
                    "bbox": bbox,
                    "area": area,
                    "score": score
                })
        
        return results
    
    def update_memory(self, frame: np.ndarray, masks: List[Dict[str, Any]]):
        """
        비디오 처리를 위한 메모리 모듈 업데이트
        
        Args:
            frame (np.ndarray): 현재 프레임
            masks (List[Dict[str, Any]]): 현재 프레임의 마스크 결과
        """
        # 메모리에 현재 프레임과 마스크 추가
        self.memory_frames.append(frame.copy())
        self.memory_masks.append(masks)
        
        # 메모리 크기 제한
        if len(self.memory_frames) > self.memory_size:
            self.memory_frames.pop(0)
            self.memory_masks.pop(0)
    
    def segment_video_frame(self, frame: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
        """
        비디오 프레임에 대한 세그멘테이션 수행
        
        Args:
            frame (np.ndarray): 입력 비디오 프레임
            frame_idx (int): 프레임 인덱스
            
        Returns:
            List[Dict[str, Any]]: 세그멘테이션 결과 리스트
        """
        # 단일 이미지 세그멘테이션 수행
        masks = self.segment_image(frame)
        
        # 메모리 모듈 업데이트 (비디오 처리용)
        if frame_idx % self.config.models.video.update_frequency == 0:
            self.update_memory(frame, masks)
        
        return masks

@hydra.main(config_path="../configs", config_name="config")
def main(config: DictConfig):
    """
    SAM2 세그멘테이션 파이프라인 테스트
    
    Args:
        config (DictConfig): Hydra 설정
    """
    log.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # 비디오 경로
    video_path = config.data.video_path
    if not os.path.exists(video_path):
        processed_path = os.path.join(config.data.save.processed_dir, "sample_video_processed.mp4")
        if os.path.exists(processed_path):
            video_path = processed_path
            log.info(f"Using processed video: {video_path}")
        else:
            log.error(f"Video not found at {video_path} or {processed_path}")
            return
    
    # 출력 디렉토리 생성
    output_dir = os.path.join(config.general.output_dir, "segmentation")
    os.makedirs(output_dir, exist_ok=True)
    
    # SAM2 세그멘터 초기화
    segmenter = SAM2Segmenter(config)
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 속성 확인
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    log.info(f"Processing video: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # 결과 비디오 작성자 객체 생성
    output_path = os.path.join(output_dir, "segmentation_results.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 프레임 처리
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 세그멘테이션 수행
        masks = segmenter.segment_video_frame(frame, frame_idx)
        
        # 결과 시각화
        vis_frame = frame.copy()
        for mask_data in masks:
            # 마스크 시각화
            mask = mask_data["mask"]
            color = np.random.randint(0, 255, size=3).tolist()
            
            # 마스크 오버레이
            mask_overlay = vis_frame.copy()
            mask_overlay[mask] = [c * 0.5 + m * 0.5 for c, m in zip(mask_overlay[mask], color)]
            
            # 알파 블렌딩
            alpha = 0.5
            vis_frame = cv2.addWeighted(mask_overlay, alpha, vis_frame, 1 - alpha, 0)
            
            # 바운딩 박스 그리기
            x, y, w, h = mask_data["bbox"]
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # 점수 표시
            score_text = f"{mask_data['score']:.2f}"
            cv2.putText(vis_frame, score_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 프레임 저장
        out.write(vis_frame)
        
        # 진행 상황 로깅
        if frame_idx % 10 == 0:
            log.info(f"Processed frame {frame_idx}/{total_frames}")
        
        frame_idx += 1
    
    # 자원 해제
    cap.release()
    out.release()
    
    log.info(f"Segmentation completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()
