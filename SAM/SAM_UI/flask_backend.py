from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import datetime
import pymongo
from bson.objectid import ObjectId
import logging
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 크로스 오리진 요청 허용

# 설정
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB

# 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/videos/upload', methods=['POST'])
def upload_video():
    # MongoDB 연결 확인
    if videos_collection is None:
        return jsonify({"error": "데이터베이스 연결 오류"}), 500
    
    # 파일 확인
    if 'video' not in request.files:
        return jsonify({"error": "비디오 파일이 없습니다"}), 400
    
    file = request.files['video']
    
    # 파일명 확인
    if file.filename == '':
        return jsonify({"error": "선택된 파일이 없습니다"}), 400
    
    # 파일 형식 확인
    if not allowed_file(file.filename):
        return jsonify({"error": "지원하지 않는 파일 형식입니다"}), 400
    
    try:
        # 안전한 파일명 생성
        filename = secure_filename(file.filename)
        # 고유 ID 생성
        video_id = str(ObjectId())
        # 저장 경로 생성
        file_extension = filename.rsplit('.', 1)[1].lower()
        storage_filename = f"{video_id}.{file_extension}"
        storage_path = os.path.join(app.config['UPLOAD_FOLDER'], storage_filename)
        
        # 파일 저장
        file.save(storage_path)
        
        # 파일 크기 확인
        file_size = os.path.getsize(storage_path)
        
        # 메타데이터 생성
        video_metadata = {
            "_id": ObjectId(video_id),
            "original_filename": filename,
            "storage_path": storage_path,
            "upload_time": datetime.datetime.now(),
            "file_size": file_size,
            "status": "uploaded",
            "result_path": None
        }
        
        # 데이터베이스에 저장
        videos_collection.insert_one(video_metadata)
        
        # 작업 큐에 추가 (여기서는 간단히 상태만 업데이트)
        videos_collection.update_one(
            {"_id": ObjectId(video_id)},
            {"$set": {"status": "processing"}}
        )
        
        # TODO: 실제 구현에서는 여기서 Celery 등을 사용해 비동기 작업 큐에 추가
        
        return jsonify({
            "message": "비디오 업로드 성공",
            "videoId": video_id,
            "filename": filename,
            "size": file_size,
            "status": "processing"
        }), 201
        
    except Exception as e:
        logger.error(f"업로드 오류: {e}")
        return jsonify({"error": f"업로드 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/api/videos', methods=['GET'])
def get_videos():
    # MongoDB 연결 확인
    if videos_collection is None:
        return jsonify({"error": "데이터베이스 연결 오류"}), 500
    
    try:
        # 모든 비디오 조회
        videos = list(videos_collection.find())
        
        # ObjectId를 문자열로 변환
        for video in videos:
            video["_id"] = str(video["_id"])
            # datetime 객체를 ISO 형식 문자열로 변환
            if "upload_time" in video:
                video["upload_time"] = video["upload_time"].isoformat()
            if "processing_start_time" in video and video["processing_start_time"]:
                video["processing_start_time"] = video["processing_start_time"].isoformat()
            if "processing_end_time" in video and video["processing_end_time"]:
                video["processing_end_time"] = video["processing_end_time"].isoformat()
        
        return jsonify({"videos": videos}), 200
        
    except Exception as e:
        logger.error(f"비디오 목록 조회 오류: {e}")
        return jsonify({"error": f"비디오 목록 조회 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/api/videos/<video_id>', methods=['GET'])
def get_video(video_id):
    # MongoDB 연결 확인
    if videos_collection is None:
        return jsonify({"error": "데이터베이스 연결 오류"}), 500
    
    try:
        # ObjectId 변환
        try:
            object_id = ObjectId(video_id)
        except:
            return jsonify({"error": "유효하지 않은 비디오 ID입니다"}), 400
        
        # 비디오 조회
        video = videos_collection.find_one({"_id": object_id})
        
        if not video:
            return jsonify({"error": "비디오를 찾을 수 없습니다"}), 404
        
        # ObjectId를 문자열로 변환
        video["_id"] = str(video["_id"])
        # datetime 객체를 ISO 형식 문자열로 변환
        if "upload_time" in video:
            video["upload_time"] = video["upload_time"].isoformat()
        if "processing_start_time" in video and video["processing_start_time"]:
            video["processing_start_time"] = video["processing_start_time"].isoformat()
        if "processing_end_time" in video and video["processing_end_time"]:
            video["processing_end_time"] = video["processing_end_time"].isoformat()
        
        return jsonify({"video": video}), 200
        
    except Exception as e:
        logger.error(f"비디오 조회 오류: {e}")
        return jsonify({"error": f"비디오 조회 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/api/videos/<video_id>/status', methods=['GET'])
def get_video_status(video_id):
    # MongoDB 연결 확인
    if videos_collection is None:
        return jsonify({"error": "데이터베이스 연결 오류"}), 500
    
    try:
        # ObjectId 변환
        try:
            object_id = ObjectId(video_id)
        except:
            return jsonify({"error": "유효하지 않은 비디오 ID입니다"}), 400
        
        # 비디오 상태 조회
        video = videos_collection.find_one(
            {"_id": object_id},
            {"status": 1, "error_message": 1}
        )
        
        if not video:
            return jsonify({"error": "비디오를 찾을 수 없습니다"}), 404
        
        status_info = {
            "videoId": video_id,
            "status": video.get("status", "unknown")
        }
        
        # 오류 메시지가 있으면 추가
        if "error_message" in video and video["error_message"]:
            status_info["error_message"] = video["error_message"]
        
        return jsonify(status_info), 200
        
    except Exception as e:
        logger.error(f"상태 조회 오류: {e}")
        return jsonify({"error": f"상태 조회 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/api/videos/<video_id>/result', methods=['GET'])
def get_video_result(video_id):
    # MongoDB 연결 확인
    if videos_collection is None or tracking_results_collection is None:
        return jsonify({"error": "데이터베이스 연결 오류"}), 500
    
    try:
        # ObjectId 변환
        try:
            object_id = ObjectId(video_id)
        except:
            return jsonify({"error": "유효하지 않은 비디오 ID입니다"}), 400
        
        # 비디오 조회
        video = videos_collection.find_one({"_id": object_id})
        
        if not video:
            return jsonify({"error": "비디오를 찾을 수 없습니다"}), 404
        
        # 상태 확인
        if video.get("status") != "completed":
            return jsonify({
                "videoId": video_id,
                "status": video.get("status", "unknown"),
                "message": "처리가 완료되지 않았습니다"
            }), 200
        
        # 추적 결과 조회
        tracking_result = tracking_results_collection.find_one({"video_id": object_id})
        
        if not tracking_result:
            return jsonify({"error": "추적 결과를 찾을 수 없습니다"}), 404
        
        # ObjectId를 문자열로 변환
        tracking_result["_id"] = str(tracking_result["_id"])
        tracking_result["video_id"] = str(tracking_result["video_id"])
        
        return jsonify({
            "videoId": video_id,
            "status": "completed",
            "result": tracking_result,
            "result_video_url": f"/api/videos/{video_id}/download"
        }), 200
        
    except Exception as e:
        logger.error(f"결과 조회 오류: {e}")
        return jsonify({"error": f"결과 조회 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/api/videos/<video_id>/download', methods=['GET'])
def download_result_video(video_id):
    # MongoDB 연결 확인
    if videos_collection is None:
        return jsonify({"error": "데이터베이스 연결 오류"}), 500
    
    try:
        # ObjectId 변환
        try:
            object_id = ObjectId(video_id)
        except:
            return jsonify({"error": "유효하지 않은 비디오 ID입니다"}), 400
        
        # 비디오 조회
        video = videos_collection.find_one({"_id": object_id})
        
        if not video:
            return jsonify({"error": "비디오를 찾을 수 없습니다"}), 404
        
        # 결과 파일 경로 확인
        result_path = video.get("result_path")
        
        if not result_path or not os.path.exists(result_path):
            return jsonify({"error": "결과 파일을 찾을 수 없습니다"}), 404
        
        # 파일 전송
        return send_file(
            result_path,
            as_attachment=True,
            download_name=f"result_{video.get('original_filename', 'video.mp4')}"
        )
        
    except Exception as e:
        logger.error(f"다운로드 오류: {e}")
        return jsonify({"error": f"다운로드 중 오류가 발생했습니다: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
