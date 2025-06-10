import os
import cv2
import torch
import numpy as np
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# ByteTrack 관련 임포트를 위한 자리 표시자
# 실제 구현 시 ByteTrack 라이브러리 임포트 필요
# from bytetrack.tracker import BYTETracker

log = logging.getLogger(__name__)

class ByteTracker:
    """ByteTrack 알고리즘을 사용한 객체 추적 클래스"""
    
    def __init__(self, config: DictConfig):
        """
        ByteTrack 추적 모델 초기화
        
        Args:
            config (DictConfig): 모델 설정
        """
        self.config = config
        self.device = torch.device(config.general.device if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")
        
        # 추적 파라미터
        self.track_thresh = config.trackers.parameters.track_thresh
        self.track_buffer = config.trackers.parameters.track_buffer
        self.match_thresh = config.trackers.parameters.match_thresh
        self.min_box_area = config.trackers.parameters.min_box_area
        self.mot20 = config.trackers.parameters.mot20
        
        # 세그멘테이션 마스크 연동 설정
        self.use_mask_iou = config.trackers.segmentation.use_mask_iou
        self.mask_iou_weight = config.trackers.segmentation.mask_iou_weight
        self.mask_area_thresh = config.trackers.segmentation.mask_area_thresh
        
        # 추적기 초기화 (실제 구현 시 ByteTrack 모델 로드 코드로 대체)
        log.info(f"Initializing ByteTrack tracker: version {config.trackers.version}")
        # self.tracker = BYTETracker(
        #     track_thresh=self.track_thresh,
        #     track_buffer=self.track_buffer,
        #     match_thresh=self.match_thresh,
        #     min_box_area=self.min_box_area,
        #     mot20=self.mot20
        # )
        
        # 추적 결과 저장용 변수
        self.tracks = []
        self.frame_id = 0
        
        log.info("ByteTrack tracker initialized successfully")
    
    def calculate_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        두 마스크 간의 IoU(Intersection over Union) 계산
        
        Args:
            mask1 (np.ndarray): 첫 번째 마스크 (boolean 배열)
            mask2 (np.ndarray): 두 번째 마스크 (boolean 배열)
            
        Returns:
            float: IoU 값 (0~1)
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def update(self, segmentation_results: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        세그멘테이션 결과를 기반으로 객체 추적 수행
        
        Args:
            segmentation_results (List[Dict[str, Any]]): 세그멘테이션 결과 리스트
            frame (np.ndarray): 현재 프레임
            
        Returns:
            List[Dict[str, Any]]: 추적 결과가 포함된 세그멘테이션 결과 리스트
        """
        self.frame_id += 1
        
        # 세그멘테이션 결과에서 바운딩 박스 및 점수 추출
        bboxes = []
        scores = []
        masks = []
        
        for result in segmentation_results:
            bbox = result["bbox"]  # [x, y, w, h] 형식
            score = result["score"]
            mask = result["mask"]
            
            # 최소 마스크 영역 필터링
            if np.sum(mask) >= self.mask_area_thresh:
                bboxes.append(bbox)
                scores.append(score)
                masks.append(mask)
        
        # 바운딩 박스를 [x1, y1, x2, y2, score] 형식으로 변환
        if bboxes:
            dets = np.array(bboxes)
            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 0] + dets[:, 2]
            y2 = dets[:, 1] + dets[:, 3]
            
            track_dets = np.stack((x1, y1, x2, y2, np.array(scores)), axis=1)
        else:
            track_dets = np.empty((0, 5))
        
        # 실제 구현 시 ByteTrack 업데이트 코드로 대체
        # online_targets = self.tracker.update(track_dets, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]])
        
        # 임시 더미 추적 결과 생성 (실제 구현 시 제거)
        online_targets = []
        for i, (bbox, score, mask) in enumerate(zip(bboxes, scores, masks)):
            # 이전 트랙과 매칭 시도
            matched = False
            track_id = -1
            
            if self.tracks:
                best_iou = 0
                best_track_idx = -1
                
                for j, track in enumerate(self.tracks):
                    # 바운딩 박스 IoU 계산
                    x1, y1, w1, h1 = bbox
                    x2, y2, w2, h2 = track["bbox"]
                    
                    # 바운딩 박스를 [x1, y1, x2, y2] 형식으로 변환
                    box1 = [x1, y1, x1 + w1, y1 + h1]
                    box2 = [x2, y2, x2 + w2, y2 + h2]
                    
                    # IoU 계산
                    xx1 = max(box1[0], box2[0])
                    yy1 = max(box1[1], box2[1])
                    xx2 = min(box1[2], box2[2])
                    yy2 = min(box1[3], box2[3])
                    
                    w = max(0, xx2 - xx1)
                    h = max(0, yy2 - yy1)
                    
                    inter = w * h
                    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    
                    iou = inter / (area1 + area2 - inter + 1e-6)
                    
                    # 마스크 IoU 계산 (옵션)
                    mask_iou = 0
                    if self.use_mask_iou:
                        mask_iou = self.calculate_mask_iou(mask, track["mask"])
                        
                        # 바운딩 박스 IoU와 마스크 IoU 결합
                        iou = (1 - self.mask_iou_weight) * iou + self.mask_iou_weight * mask_iou
                    
                    # 최고 IoU 트랙 찾기
                    if iou > best_iou and iou >= self.match_thresh:
                        best_iou = iou
                        best_track_idx = j
                
                # 매칭된 트랙이 있으면 ID 할당
                if best_track_idx >= 0:
                    matched = True
                    track_id = self.tracks[best_track_idx]["track_id"]
            
            # 새 트랙 생성 또는 기존 트랙 업데이트
            if not matched:
                track_id = len(self.tracks) if not self.tracks else max([t["track_id"] for t in self.tracks]) + 1
            
            # 추적 결과 저장
            online_targets.append({
                "track_id": track_id,
                "bbox": bbox,
                "score": score,
                "mask": mask,
                "matched": matched
            })
        
        # 추적 결과 저장
        self.tracks = online_targets
        
        # 세그멘테이션 결과에 추적 ID 할당
        tracked_results = []
        for i, result in enumerate(segmentation_results):
            if i < len(online_targets):
                result["track_id"] = online_targets[i]["track_id"]
                result["matched"] = online_targets[i]["matched"]
            else:
                result["track_id"] = -1
                result["matched"] = False
            
            tracked_results.append(result)
        
        return tracked_results

@hydra.main(config_path="../configs", config_name="config")
def main(config: DictConfig):
    """
    ByteTrack 추적 파이프라인 테스트
    
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
    output_dir = os.path.join(config.general.output_dir, "tracking")
    os.makedirs(output_dir, exist_ok=True)
    
    # ByteTrack 추적기 초기화
    tracker = ByteTracker(config)
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 속성 확인
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    log.info(f"Processing video: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # 결과 비디오 작성자 객체 생성
    output_path = os.path.join(output_dir, "tracking_results.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 프레임 처리
    frame_idx = 0
    
    # 임시 세그멘테이션 결과 생성 함수 (실제 구현 시 SAM2 모델 호출로 대체)
    def generate_dummy_segmentation(frame):
        h, w = frame.shape[:2]
        results = []
        
        # 첫 번째 마스크 (원)
        mask1 = np.zeros((h, w), dtype=bool)
        center_x = w // 2 + int(np.sin(frame_idx / 10) * w // 8)
        center_y = h // 2 + int(np.cos(frame_idx / 10) * h // 8)
        radius = min(w, h) // 6
        
        for y in range(h):
            for x in range(w):
                if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                    mask1[y, x] = True
        
        # 바운딩 박스 계산
        y_indices, x_indices = np.where(mask1)
        if len(y_indices) > 0 and len(x_indices) > 0:
            x1, x2 = np.min(x_indices), np.max(x_indices)
            y1, y2 = np.min(y_indices), np.max(y_indices)
            bbox1 = [x1, y1, x2 - x1, y2 - y1]
        else:
            bbox1 = [0, 0, 0, 0]
        
        # 두 번째 마스크 (사각형)
        mask2 = np.zeros((h, w), dtype=bool)
        x1 = w // 4 + int(np.cos(frame_idx / 15) * w // 10)
        y1 = h // 4 + int(np.sin(frame_idx / 15) * h // 10)
        x2 = 3 * w // 4
        y2 = 3 * h // 4
        
        mask2[y1:y2, x1:x2] = True
        bbox2 = [x1, y1, x2 - x1, y2 - y1]
        
        results.append({
            "id": 0,
            "mask": mask1,
            "bbox": bbox1,
            "area": np.sum(mask1),
            "score": 0.95
        })
        
        results.append({
            "id": 1,
            "mask": mask2,
            "bbox": bbox2,
            "area": np.sum(mask2),
            "score": 0.85
        })
        
        return results
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 세그멘테이션 수행 (실제 구현 시 SAM2 모델 호출로 대체)
        segmentation_results = generate_dummy_segmentation(frame)
        
        # 추적 수행
        tracked_results = tracker.update(segmentation_results, frame)
        
        # 결과 시각화
        vis_frame = frame.copy()
        
        # 트랙 ID별 고유 색상 생성
        track_colors = {}
        
        for result in tracked_results:
            track_id = result.get("track_id", -1)
            
            # 트랙 ID별 고유 색상 할당
            if track_id not in track_colors and track_id != -1:
                # HSV 색상 공간에서 고르게 분포된 색상 생성
                hue = (track_id * 30) % 180
                track_colors[track_id] = tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]))
            
            color = track_colors.get(track_id, (0, 0, 255))  # 매칭되지 않은 객체는 빨간색
            
            # 마스크 시각화
            mask = result["mask"]
            
            # 마스크 오버레이
            mask_overlay = vis_frame.copy()
            mask_overlay[mask] = [c * 0.5 + m * 0.5 for c, m in zip(mask_overlay[mask], color)]
            
            # 알파 블렌딩
            alpha = 0.5
            vis_frame = cv2.addWeighted(mask_overlay, alpha, vis_frame, 1 - alpha, 0)
            
            # 바운딩 박스 그리기
            x, y, w, h = result["bbox"]
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # 트랙 ID 및 점수 표시
            if track_id != -1:
                id_text = f"ID: {track_id}, {result['score']:.2f}"
                cv2.putText(vis_frame, id_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 프레임 저장
        out.write(vis_frame)
        
        # 진행 상황 로깅
        if frame_idx % 10 == 0:
            log.info(f"Processed frame {frame_idx}/{total_frames}")
        
        frame_idx += 1
    
    # 자원 해제
    cap.release()
    out.release()
    
    log.info(f"Tracking completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()
