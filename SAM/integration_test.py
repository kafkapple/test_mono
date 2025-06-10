import unittest
import os
import sys
import json
import time
import requests
import pymongo
from bson.objectid import ObjectId
from datetime import datetime

# 테스트 설정
API_BASE_URL = "http://localhost:5000/api"
TEST_VIDEO_PATH = "/home/ubuntu/sam_mot_web/test_video.mp4"
MONGODB_URI = "mongodb://localhost:27017/"
DB_NAME = "sam_mot_db"

class SAMMOTIntegrationTest(unittest.TestCase):
    """SAM-MOT 웹 앱 통합 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 설정"""
        # 테스트 비디오 생성 (더미 파일)
        if not os.path.exists(TEST_VIDEO_PATH):
            cls._create_dummy_video(TEST_VIDEO_PATH)
        
        # MongoDB 연결
        try:
            cls.mongo_client = pymongo.MongoClient(MONGODB_URI)
            cls.db = cls.mongo_client[DB_NAME]
            cls.videos_collection = cls.db["videos"]
            cls.tracking_results_collection = cls.db["tracking_results"]
            print("MongoDB 연결 성공")
        except Exception as e:
            print(f"MongoDB 연결 실패: {e}")
            cls.mongo_client = None
            cls.db = None
            cls.videos_collection = None
            cls.tracking_results_collection = None
    
    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 정리"""
        # MongoDB 연결 종료
        if cls.mongo_client:
            cls.mongo_client.close()
        
        # 테스트 비디오 삭제
        if os.path.exists(TEST_VIDEO_PATH):
            os.remove(TEST_VIDEO_PATH)
    
    @staticmethod
    def _create_dummy_video(path, duration=5):
        """더미 비디오 파일 생성"""
        try:
            import cv2
            import numpy as np
            
            # 비디오 속성
            width, height = 640, 480
            fps = 30
            total_frames = duration * fps
            
            # 비디오 작성자
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(path, fourcc, fps, (width, height))
            
            # 프레임 생성
            for i in range(total_frames):
                # 검은 배경에 움직이는 사각형
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # 움직이는 사각형
                x = int(width * 0.5 + width * 0.3 * np.sin(i * 0.05))
                y = int(height * 0.5 + height * 0.3 * np.cos(i * 0.05))
                
                # 사각형 그리기
                cv2.rectangle(frame, (x - 50, y - 50), (x + 50, y + 50), (0, 255, 0), -1)
                
                # 프레임 번호
                cv2.putText(
                    frame, 
                    f"Frame: {i}", 
                    (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 255, 255), 
                    2
                )
                
                # 결과 저장
                out.write(frame)
            
            # 자원 해제
            out.release()
            
            print(f"더미 비디오 생성 완료: {path}")
            return True
            
        except Exception as e:
            print(f"더미 비디오 생성 실패: {e}")
            return False
    
    def test_01_upload_video(self):
        """비디오 업로드 테스트"""
        print("\n테스트 1: 비디오 업로드")
        
        # 파일 존재 확인
        self.assertTrue(os.path.exists(TEST_VIDEO_PATH), "테스트 비디오 파일이 존재하지 않습니다")
        
        # 업로드 요청
        url = f"{API_BASE_URL}/videos/upload"
        
        with open(TEST_VIDEO_PATH, 'rb') as f:
            files = {'video': (os.path.basename(TEST_VIDEO_PATH), f, 'video/mp4')}
            response = requests.post(url, files=files)
        
        # 응답 확인
        self.assertEqual(response.status_code, 201, f"업로드 실패: {response.text}")
        
        # 응답 데이터 확인
        data = response.json()
        self.assertIn('videoId', data, "응답에 videoId가 없습니다")
        self.assertIn('status', data, "응답에 status가 없습니다")
        self.assertEqual(data['status'], 'processing', "상태가 'processing'이 아닙니다")
        
        # 비디오 ID 저장
        self.video_id = data['videoId']
        print(f"업로드된 비디오 ID: {self.video_id}")
    
    def test_02_check_video_status(self):
        """비디오 상태 확인 테스트"""
        print("\n테스트 2: 비디오 상태 확인")
        
        # 비디오 ID 확인
        self.assertTrue(hasattr(self, 'video_id'), "비디오 ID가 없습니다")
        
        # 상태 확인 요청
        url = f"{API_BASE_URL}/videos/{self.video_id}/status"
        
        # 최대 30초 동안 상태 확인 (5초 간격)
        max_attempts = 6
        for attempt in range(max_attempts):
            response = requests.get(url)
            
            # 응답 확인
            self.assertEqual(response.status_code, 200, f"상태 확인 실패: {response.text}")
            
            # 응답 데이터 확인
            data = response.json()
            self.assertIn('status', data, "응답에 status가 없습니다")
            
            print(f"비디오 상태: {data['status']}")
            
            # 처리 완료 또는 실패 시 종료
            if data['status'] in ['completed', 'failed']:
                break
            
            # 5초 대기
            time.sleep(5)
        
        # 최종 상태 확인
        self.assertIn(data['status'], ['completed', 'failed'], "처리가 완료되지 않았습니다")
    
    def test_03_get_video_result(self):
        """비디오 결과 조회 테스트"""
        print("\n테스트 3: 비디오 결과 조회")
        
        # 비디오 ID 확인
        self.assertTrue(hasattr(self, 'video_id'), "비디오 ID가 없습니다")
        
        # 결과 조회 요청
        url = f"{API_BASE_URL}/videos/{self.video_id}/result"
        response = requests.get(url)
        
        # 응답 확인
        self.assertEqual(response.status_code, 200, f"결과 조회 실패: {response.text}")
        
        # 응답 데이터 확인
        data = response.json()
        self.assertIn('status', data, "응답에 status가 없습니다")
        
        # 처리 완료 상태인 경우에만 결과 확인
        if data['status'] == 'completed':
            self.assertIn('result', data, "응답에 result가 없습니다")
            self.assertIn('result_video_url', data, "응답에 result_video_url이 없습니다")
            
            # 추적 결과 확인
            result = data['result']
            self.assertIn('objects', result, "결과에 objects가 없습니다")
            self.assertTrue(len(result['objects']) > 0, "추적된 객체가 없습니다")
            
            print(f"추적된 객체 수: {len(result['objects'])}")
    
    def test_04_download_result(self):
        """결과 다운로드 테스트"""
        print("\n테스트 4: 결과 다운로드")
        
        # 비디오 ID 확인
        self.assertTrue(hasattr(self, 'video_id'), "비디오 ID가 없습니다")
        
        # 상태 확인
        url = f"{API_BASE_URL}/videos/{self.video_id}/status"
        response = requests.get(url)
        data = response.json()
        
        # 처리 완료 상태인 경우에만 다운로드 테스트
        if data['status'] == 'completed':
            # 다운로드 요청
            url = f"{API_BASE_URL}/videos/{self.video_id}/download"
            response = requests.get(url)
            
            # 응답 확인
            self.assertEqual(response.status_code, 200, "다운로드 실패")
            self.assertIn('content-type', response.headers, "응답에 content-type이 없습니다")
            self.assertEqual(response.headers['content-type'], 'video/mp4', "content-type이 'video/mp4'가 아닙니다")
            
            # 파일 크기 확인
            self.assertTrue(len(response.content) > 0, "다운로드된 파일이 비어 있습니다")
            
            print(f"다운로드된 파일 크기: {len(response.content)} 바이트")
    
    def test_05_check_database_integrity(self):
        """데이터베이스 무결성 확인 테스트"""
        print("\n테스트 5: 데이터베이스 무결성 확인")
        
        # MongoDB 연결 확인
        self.assertIsNotNone(self.videos_collection, "videos_collection이 None입니다")
        self.assertIsNotNone(self.tracking_results_collection, "tracking_results_collection이 None입니다")
        
        # 비디오 ID 확인
        self.assertTrue(hasattr(self, 'video_id'), "비디오 ID가 없습니다")
        
        # 비디오 메타데이터 조회
        video = self.videos_collection.find_one({"_id": ObjectId(self.video_id)})
        self.assertIsNotNone(video, "비디오 메타데이터를 찾을 수 없습니다")
        
        # 필수 필드 확인
        required_fields = ["original_filename", "storage_path", "upload_time", "status"]
        for field in required_fields:
            self.assertIn(field, video, f"비디오 메타데이터에 {field}가 없습니다")
        
        # 처리 완료 상태인 경우 추가 필드 확인
        if video['status'] == 'completed':
            self.assertIn("processing_end_time", video, "비디오 메타데이터에 processing_end_time이 없습니다")
            self.assertIn("result_path", video, "비디오 메타데이터에 result_path가 없습니다")
            
            # 결과 파일 존재 확인
            self.assertTrue(os.path.exists(video['result_path']), "결과 파일이 존재하지 않습니다")
            
            # 추적 결과 조회
            tracking_result = self.tracking_results_collection.find_one({"video_id": ObjectId(self.video_id)})
            self.assertIsNotNone(tracking_result, "추적 결과를 찾을 수 없습니다")
            
            # 필수 필드 확인
            required_fields = ["frame_count", "fps", "objects"]
            for field in required_fields:
                self.assertIn(field, tracking_result, f"추적 결과에 {field}가 없습니다")
            
            print(f"추적 결과 프레임 수: {tracking_result['frame_count']}")
            print(f"추적 결과 객체 수: {len(tracking_result['objects'])}")

if __name__ == "__main__":
    unittest.main()
