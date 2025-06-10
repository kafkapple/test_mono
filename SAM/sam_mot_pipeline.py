import os
import cv2
import numpy as np
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import List, Dict, Any, Optional

# 모듈 임포트
from src.models.sam2_segmenter import SAM2Segmenter
from src.trackers.bytetrack_tracker import ByteTracker

log = logging.getLogger(__name__)

class SAM_MOT_Pipeline:
    """SAM2와 ByteTrack을 결합한 객체 추적 파이프라인"""
    
    def __init__(self, config: DictConfig):
        """
        파이프라인 초기화
        
        Args:
            config (DictConfig): Hydra 설정
        """
        self.config = config
        
        # SAM2 세그멘터 초기화
        log.info("Initializing SAM2 segmenter")
        self.segmenter = SAM2Segmenter(config)
        
        # ByteTrack 추적기 초기화
        log.info("Initializing ByteTrack tracker")
        self.tracker = ByteTracker(config)
        
        # 궤적 저장용 딕셔너리
        self.trajectories = {}
        self.max_trajectory_len = config.visualization.trajectory.length
        
        log.info("SAM-MOT pipeline initialized successfully")
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """
        단일 프레임 처리
        
        Args:
            frame (np.ndarray): 입력 비디오 프레임
            frame_idx (int): 프레임 인덱스
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        # 1. 세그멘테이션 수행
        segmentation_results = self.segmenter.segment_video_frame(frame, frame_idx)
        
        # 2. 추적 수행
        tracked_results = self.tracker.update(segmentation_results, frame)
        
        # 3. 궤적 업데이트
        self._update_trajectories(tracked_results)
        
        return {
            "frame": frame,
            "segmentation_results": segmentation_results,
            "tracked_results": tracked_results,
            "trajectories": self.trajectories
        }
    
    def _update_trajectories(self, tracked_results: List[Dict[str, Any]]):
        """
        객체 궤적 업데이트
        
        Args:
            tracked_results (List[Dict[str, Any]]): 추적 결과 리스트
        """
        # 현재 프레임의 트랙 ID 목록
        current_track_ids = set()
        
        for result in tracked_results:
            track_id = result.get("track_id", -1)
            if track_id != -1:
                current_track_ids.add(track_id)
                
                # 바운딩 박스 중심점 계산
                x, y, w, h = result["bbox"]
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 궤적 업데이트
                if track_id not in self.trajectories:
                    self.trajectories[track_id] = []
                
                self.trajectories[track_id].append((center_x, center_y))
                
                # 최대 궤적 길이 제한
                if len(self.trajectories[track_id]) > self.max_trajectory_len:
                    self.trajectories[track_id] = self.trajectories[track_id][-self.max_trajectory_len:]
    
    def visualize_results(self, results: Dict[str, Any]) -> np.ndarray:
        """
        처리 결과 시각화
        
        Args:
            results (Dict[str, Any]): 처리 결과
            
        Returns:
            np.ndarray: 시각화된 프레임
        """
        frame = results["frame"].copy()
        tracked_results = results["tracked_results"]
        trajectories = results["trajectories"]
        
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
            if self.config.visualization.mask.show_contour:
                # 마스크 윤곽선 그리기
                mask = result["mask"].astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color, self.config.visualization.mask.contour_thickness)
            
            # 마스크 오버레이
            if self.config.visualization.mask.alpha > 0:
                mask = result["mask"]
                mask_overlay = frame.copy()
                mask_overlay[mask] = [c * (1 - self.config.visualization.mask.alpha) + m * self.config.visualization.mask.alpha for c, m in zip(mask_overlay[mask], color)]
                frame = mask_overlay
            
            # 바운딩 박스 그리기
            if self.config.visualization.bbox.show:
                x, y, w, h = result["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.config.visualization.bbox.thickness)
            
            # 트랙 ID 및 점수 표시
            if track_id != -1 and self.config.visualization.show_track_id:
                id_text = f"ID: {track_id}"
                if self.config.visualization.show_confidence:
                    id_text += f", {result['score']:.2f}"
                
                cv2.putText(frame, id_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           self.config.visualization.general.font_size, 
                           color, 
                           self.config.visualization.general.line_thickness)
        
        # 궤적 그리기
        if self.config.visualization.trajectory.show:
            for track_id, trajectory in trajectories.items():
                if track_id in track_colors:
                    color = track_colors[track_id]
                    
                    # 궤적 점들을 선으로 연결
                    for i in range(1, len(trajectory)):
                        cv2.line(frame, 
                                trajectory[i-1], 
                                trajectory[i], 
                                color, 
                                self.config.visualization.trajectory.thickness)
        
        # FPS 표시
        if self.config.visualization.general.show_fps and hasattr(self, 'fps'):
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       self.config.visualization.general.font_size, 
                       (0, 255, 0), 
                       self.config.visualization.general.line_thickness)
        
        return frame

@hydra.main(config_path="../configs", config_name="config")
def main(config: DictConfig):
    """
    SAM-MOT 통합 파이프라인 실행
    
    Args:
        config (DictConfig): Hydra 설정
    """
    log.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # 비디오 경로
    video_path = config.run.video_path
    if not os.path.exists(video_path):
        processed_path = os.path.join(config.data.save.processed_dir, "sample_video_processed.mp4")
        if os.path.exists(processed_path):
            video_path = processed_path
            log.info(f"Using processed video: {video_path}")
        else:
            log.error(f"Video not found at {video_path} or {processed_path}")
            return
    
    # 출력 디렉토리 생성
    output_dir = os.path.join(config.general.output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # SAM-MOT 파이프라인 초기화
    pipeline = SAM_MOT_Pipeline(config)
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 속성 확인
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    log.info(f"Processing video: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # 결과 비디오 작성자 객체 생성
    if config.run.save_video:
        output_path = os.path.join(output_dir, "sam_mot_results.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 프레임 처리
    frame_idx = 0
    processing_times = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 처리 시간 측정 시작
        start_time = cv2.getTickCount()
        
        # 프레임 처리
        results = pipeline.process_frame(frame, frame_idx)
        
        # 결과 시각화
        vis_frame = pipeline.visualize_results(results)
        
        # 프레임 처리 시간 측정 종료
        end_time = cv2.getTickCount()
        processing_time = (end_time - start_time) / cv2.getTickFrequency()
        processing_times.append(processing_time)
        
        # FPS 계산 (최근 10프레임 평균)
        if len(processing_times) > 10:
            processing_times = processing_times[-10:]
        pipeline.fps = 1.0 / np.mean(processing_times)
        
        # 결과 저장
        if config.run.save_video:
            out.write(vis_frame)
        
        # 결과 표시
        if config.run.display:
            cv2.imshow("SAM-MOT Tracking", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 진행 상황 로깅
        if frame_idx % 10 == 0:
            log.info(f"Processed frame {frame_idx}/{total_frames}, FPS: {pipeline.fps:.1f}")
        
        frame_idx += 1
    
    # 자원 해제
    cap.release()
    if config.run.save_video:
        out.release()
    cv2.destroyAllWindows()
    
    log.info(f"Processing completed. Results saved to {output_dir}")
    
    # 결과 요약
    log.info(f"Processed {frame_idx} frames")
    log.info(f"Average FPS: {1.0 / np.mean(processing_times):.1f}")
    
    if config.run.save_video:
        log.info(f"Result video saved to {output_path}")

if __name__ == "__main__":
    main()
