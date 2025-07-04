import os
import time
import cv2
import numpy as np
import json
from bson.objectid import ObjectId
import pymongo
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB 연결
try:
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mongo_client["sam_mot_db"]
    videos_collection = db["videos"]
    tracking_results_collection = db["tracking_results"]
    logger.info("MongoDB 연결 성공")
except Exception as e:
    logger.error(f"MongoDB 연결 실패: {e}")
    mongo_client = None
    db = None
    videos_collection = None
    tracking_results_collection = None

# 설정
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

class SAMMOTWorker:
    """SAM-MOT 알고리즘을 실행하는 워커 클래스"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__ + '.SAMMOTWorker')
        self.logger.info("SAM-MOT 워커 초기화")
    
    def process_video(self, video_id):
        """
        비디오 처리 메인 함수
        
        Args:
            video_id (str): 처리할 비디오 ID
        
        Returns:
            bool: 처리 성공 여부
        """
        try:
            # 비디오 정보 조회
            video_info = videos_collection.find_one({"_id": ObjectId(video_id)})
            
            if not video_info:
                self.logger.error(f"비디오를 찾을 수 없음: {video_id}")
                return False
            
            # 처리 시작 시간 업데이트
            videos_collection.update_one(
                {"_id": ObjectId(video_id)},
                {"$set": {
                    "status": "processing",
                    "processing_start_time": datetime.now()
                }}
            )
            
            # 비디오 파일 경로
            video_path = video_info.get("storage_path")
            
            if not video_path or not os.path.exists(video_path):
                self.logger.error(f"비디오 파일을 찾을 수 없음: {video_path}")
                self._update_error_status(video_id, "비디오 파일을 찾을 수 없습니다.")
                return False
            
            # 결과 파일 경로
            result_filename = f"result_{video_id}.mp4"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            
            # SAM-MOT 알고리즘 실행 (여기서는 더미 구현)
            success = self._run_sam_mot_algorithm(video_path, result_path)
            
            if not success:
                self._update_error_status(video_id, "SAM-MOT 알고리즘 실행 중 오류가 발생했습니다.")
                return False
            
            # 추적 결과 데이터 생성 (더미 데이터)
            tracking_data = self._generate_tracking_results(video_id)
            
            # 추적 결과 저장
            tracking_results_collection.insert_one(tracking_data)
            
            # 처리 완료 상태 업데이트
            videos_collection.update_one(
                {"_id": ObjectId(video_id)},
                {"$set": {
                    "status": "completed",
                    "processing_end_time": datetime.now(),
                    "result_path": result_path
                }}
            )
            
            self.logger.info(f"비디오 처리 완료: {video_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"비디오 처리 중 오류 발생: {e}")
            self._update_error_status(video_id, f"처리 중 오류가 발생했습니다: {str(e)}")
            return False
    
    def _run_sam_mot_algorithm(self, video_path, result_path):
        """
        SAM-MOT 알고리즘 실행 (더미 구현)
        
        Args:
            video_path (str): 입력 비디오 경로
            result_path (str): 결과 비디오 저장 경로
        
        Returns:
            bool: 처리 성공 여부
        """
        try:
            self.logger.info(f"SAM-MOT 알고리즘 실행 중: {video_path}")
            
            # 비디오 열기
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                self.logger.error(f"비디오를 열 수 없음: {video_path}")
                return False
            
            # 비디오 속성
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 결과 비디오 작성자
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
            
            # 프레임 처리 (더미 처리: 텍스트 및 사각형 추가)
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # 진행 상황 로깅 (10% 단위)
                if frame_idx % (total_frames // 10) == 0:
                    progress = (frame_idx / total_frames) * 100
                    self.logger.info(f"처리 진행률: {progress:.1f}%")
                
                # 더미 처리: 프레임에 텍스트 및 사각형 추가
                # 실제 구현에서는 여기서 SAM-MOT 알고리즘 호출
                
                # 프레임 번호 표시
                cv2.putText(
                    frame, 
                    f"Frame: {frame_idx}", 
                    (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                # 더미 객체 추적 결과 표시
                num_objects = 3
                for i in range(num_objects):
                    # 객체마다 다른 색상
                    color = [
                        (0, 0, 255),  # 빨강
                        (0, 255, 0),  # 초록
                        (255, 0, 0)   # 파랑
                    ][i % 3]
                    
                    # 움직이는 사각형 생성
                    x = int(width * 0.2 + (width * 0.6) * (i / num_objects) + 50 * np.sin(frame_idx * 0.05))
                    y = int(height * 0.2 + 50 * np.cos(frame_idx * 0.05))
                    w = 100
                    h = 100
                    
                    # 사각형 그리기
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # 객체 ID 표시
                    cv2.putText(
                        frame, 
                        f"ID: {i+1}", 
                        (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2
                    )
                
                # 결과 저장
                out.write(frame)
                
                frame_idx += 1
                
                # 처리 시간 시뮬레이션 (실제 구현에서는 제거)
                time.sleep(0.01)
            
            # 자원 해제
            cap.release()
            out.release()
            
            self.logger.info(f"SAM-MOT 알고리즘 실행 완료: {result_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"SAM-MOT 알고리즘 실행 중 오류: {e}")
            return False
    
    def _generate_tracking_results(self, video_id):
        """
        추적 결과 데이터 생성 (더미 데이터)
        
        Args:
            video_id (str): 비디오 ID
        
        Returns:
            dict: 추적 결과 데이터
        """
        # 더미 객체 추적 결과 생성
        num_objects = 3
        num_frames = 100
        
        objects = []
        
        for i in range(num_objects):
            frames = []
            
            for j in range(num_frames):
                # 움직이는 바운딩 박스 생성
                x = int(100 + 300 * (i / num_objects) + 50 * np.sin(j * 0.05))
                y = int(100 + 50 * np.cos(j * 0.05))
                w = 100
                h = 100
                
                # 더미 마스크 경로 (실제 구현에서는 RLE 인코딩된 마스크 또는 경로)
                mask = f"mask_{video_id}_{i}_{j}.png"
                
                frames.append({
                    "frame_id": j,
                    "bbox": [x, y, w, h],
                    "mask": mask,
                    "score": 0.9 - (0.1 * i)  # 더미 점수
                })
            
            objects.append({
                "track_id": i + 1,
                "frames": frames
            })
        
        # 추적 결과 데이터
        tracking_data = {
            "video_id": ObjectId(video_id),
            "frame_count": num_frames,
            "fps": 30,
            "objects": objects,
            "performance_metrics": {
                "processing_time": 10.5,
                "average_fps": 28.5
            }
        }
        
        return tracking_data
    
    def _update_error_status(self, video_id, error_message):
        """
        오류 상태 업데이트
        
        Args:
            video_id (str): 비디오 ID
            error_message (str): 오류 메시지
        """
        try:
            videos_collection.update_one(
                {"_id": ObjectId(video_id)},
                {"$set": {
                    "status": "failed",
                    "processing_end_time": datetime.now(),
                    "error_message": error_message
                }}
            )
        except Exception as e:
            self.logger.error(f"오류 상태 업데이트 실패: {e}")

def process_pending_videos():
    """대기 중인 비디오 처리"""
    try:
        # 대기 중인 비디오 조회
        pending_videos = videos_collection.find({"status": "uploaded"})
        
        worker = SAMMOTWorker()
        
        for video in pending_videos:
            video_id = str(video["_id"])
            logger.info(f"대기 중인 비디오 처리 시작: {video_id}")
            
            # 비디오 처리
            worker.process_video(video_id)
    
    except Exception as e:
        logger.error(f"대기 중인 비디오 처리 중 오류: {e}")

if __name__ == "__main__":
    # 대기 중인 비디오 처리
    process_pending_videos()
    
    # 실제 구현에서는 여기서 Celery 워커 또는 주기적 작업 실행
    logger.info("SAM-MOT 워커 실행 중...")
    
    try:
        while True:
            # 주기적으로 대기 중인 비디오 확인 및 처리
            process_pending_videos()
            
            # 10초 대기
            time.sleep(10)
    
    except KeyboardInterrupt:
        logger.info("SAM-MOT 워커 종료")
