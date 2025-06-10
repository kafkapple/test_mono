import React from 'react';
import { Code, FileText, Terminal } from 'lucide-react';

const Implementation: React.FC = () => {
  return (
    <section id="implementation" className="py-16 bg-white">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">구현 및 코드 샘플</h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            SAM-MOT 객체 추적 시스템의 핵심 구현 내용과 주요 코드 샘플을 살펴봅니다.
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
          <div className="bg-white p-6 rounded-xl shadow-md col-span-1">
            <h3 className="text-xl font-bold mb-4 flex items-center">
              <div className="bg-blue-100 p-2 rounded-full mr-3">
                <FileText size={20} className="text-blue-600" />
              </div>
              프로젝트 구조
            </h3>
            
            <div className="bg-slate-900 text-slate-200 p-4 rounded-lg font-mono text-sm overflow-auto max-h-96">
              <pre>
{`sam_mot_tracking/
├── configs/                  # Hydra 설정 파일
│   ├── config.yaml           # 기본 설정 파일
│   ├── models/               # 모델 관련 설정
│   │   └── sam2.yaml
│   ├── trackers/             # 추적기 관련 설정
│   │   └── bytetrack.yaml
│   ├── data/                 # 데이터 관련 설정
│   │   └── default.yaml
│   └── visualization/        # 시각화 관련 설정
│       └── default.yaml
├── data/                     # 데이터 저장 디렉토리
│   ├── raw/                  # 원본 영상 데이터
│   └── processed/            # 전처리된 영상 데이터
├── src/                      # 소스 코드
│   ├── models/               # 세그멘테이션 모델
│   │   └── sam2_segmenter.py
│   ├── trackers/             # 추적 알고리즘
│   │   └── bytetrack_tracker.py
│   ├── data/                 # 데이터 처리
│   │   └── prepare_data.py
│   ├── utils/                # 유틸리티 함수
│   ├── visualization/        # 시각화 관련 코드
│   └── sam_mot_pipeline.py   # 통합 파이프라인
├── outputs/                  # 결과물 저장 디렉토리
├── run_pipeline.py           # 파이프라인 실행 스크립트
└── requirements_analysis.md  # 요구사항 분석 문서`}
              </pre>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-md col-span-2">
            <h3 className="text-xl font-bold mb-4 flex items-center">
              <div className="bg-purple-100 p-2 rounded-full mr-3">
                <Terminal size={20} className="text-purple-600" />
              </div>
              Hydra 설정 관리
            </h3>
            
            <p className="text-gray-700 mb-4">
              Hydra 프레임워크를 사용하여 다양한 설정을 쉽게 관리하고 실험할 수 있습니다.
              아래는 기본 설정 파일의 예시입니다:
            </p>
            
            <div className="bg-slate-900 text-slate-200 p-4 rounded-lg font-mono text-sm overflow-auto">
              <pre>
{`# 기본 설정 파일
defaults:
  - _self_
  - models: sam2
  - trackers: bytetrack
  - data: default
  - visualization: default
  - override hydra/job_logging: colorlog
  - override hydra/hydra_logging: colorlog

# 일반 설정
general:
  seed: 42
  device: cuda  # cuda 또는 cpu
  debug: false
  save_results: true
  output_dir: \${hydra:runtime.cwd}/outputs/\${now:%Y-%m-%d_%H-%M-%S}

# 실행 모드 설정
run:
  mode: video  # video 또는 webcam
  video_path: \${data.video_path}
  save_video: true
  display: true
  num_objects: 5  # 추적할 객체 수`}
              </pre>
            </div>
            
            <div className="mt-6">
              <h4 className="text-lg font-semibold mb-2">설정 변경 예시</h4>
              <div className="bg-slate-900 text-slate-200 p-4 rounded-lg font-mono text-sm overflow-auto">
                <pre>
{`# 추적할 객체 수 변경
python run_pipeline.py run.num_objects=10

# 다른 비디오 파일 사용
python run_pipeline.py run.video_path=/path/to/your/video.mp4

# 디스플레이 끄기
python run_pipeline.py run.display=false

# 여러 설정 동시 변경
python run_pipeline.py models=sam2 trackers=bytetrack run.num_objects=5 visualization.trajectory.show=true`}
                </pre>
              </div>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-1 gap-8 mb-12">
          <div className="bg-white p-6 rounded-xl shadow-md">
            <h3 className="text-xl font-bold mb-4 flex items-center">
              <div className="bg-green-100 p-2 rounded-full mr-3">
                <Code size={20} className="text-green-600" />
              </div>
              통합 파이프라인 코드
            </h3>
            
            <p className="text-gray-700 mb-4">
              SAM2 세그멘테이션과 ByteTrack 추적을 통합하는 핵심 파이프라인 코드입니다:
            </p>
            
            <div className="bg-slate-900 text-slate-200 p-4 rounded-lg font-mono text-sm overflow-auto max-h-96">
              <pre>
{`class SAM_MOT_Pipeline:
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
                    self.trajectories[track_id] = self.trajectories[track_id][-self.max_trajectory_len:]`}
              </pre>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white p-6 rounded-xl shadow-md">
            <h3 className="text-xl font-bold mb-4">SAM2 세그멘테이션 모듈</h3>
            
            <p className="text-gray-700 mb-4">
              SAM2 모델을 사용하여 비디오 프레임에서 객체를 세그멘테이션하는 핵심 코드입니다:
            </p>
            
            <div className="bg-slate-900 text-slate-200 p-4 rounded-lg font-mono text-sm overflow-auto max-h-80">
              <pre>
{`def segment_video_frame(self, frame: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
    """
    비디오 프레임에 대한 세그멘테이션 수행
    
    Args:
        frame (np.ndarray): 입력 비디오 프레임
        frame_idx (int): 프레임 인덱스
        
    Returns:
        List[Dict[str, Any]]: 세그멘테이션 결과 리스트
    """
    # RGB로 변환 (SAM2는 RGB 입력 예상)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 프롬프트 생성
    prompts = self.generate_prompts(rgb_image)
    
    # SAM2 모델 예측
    self.predictor.set_image(rgb_image)
    masks, scores, logits = self.predictor.predict(**prompts)
    
    # 결과 형식화
    results = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
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
    
    # 메모리 모듈 업데이트 (비디오 처리용)
    if frame_idx % self.config.models.video.update_frequency == 0:
        self.update_memory(frame, results)
    
    return results`}
              </pre>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-md">
            <h3 className="text-xl font-bold mb-4">ByteTrack 추적 모듈</h3>
            
            <p className="text-gray-700 mb-4">
              세그멘테이션 결과를 기반으로 객체를 추적하는 ByteTrack 모듈의 핵심 코드입니다:
            </p>
            
            <div className="bg-slate-900 text-slate-200 p-4 rounded-lg font-mono text-sm overflow-auto max-h-80">
              <pre>
{`def update(self, segmentation_results: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
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
    
    # ByteTrack 업데이트
    online_targets = self.tracker.update(
        track_dets, 
        [frame.shape[0], frame.shape[1]], 
        [frame.shape[0], frame.shape[1]]
    )
    
    # 추적 결과 처리
    tracked_results = []
    for i, result in enumerate(segmentation_results):
        if i < len(online_targets):
            target = online_targets[i]
            result["track_id"] = target.track_id
            result["matched"] = True
        else:
            result["track_id"] = -1
            result["matched"] = False
        
        tracked_results.append(result)
    
    return tracked_results`}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Implementation;
