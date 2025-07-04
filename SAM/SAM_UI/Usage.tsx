import React from 'react';
import { BookOpen, Settings, HelpCircle } from 'lucide-react';

const Usage: React.FC = () => {
  return (
    <section id="usage" className="py-16 bg-white">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">사용 방법</h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            SAM-MOT 객체 추적 시스템의 설치 및 사용 방법을 안내합니다.
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
          <div className="bg-white p-6 rounded-xl shadow-md">
            <h3 className="text-xl font-bold mb-4 flex items-center">
              <div className="bg-blue-100 p-2 rounded-full mr-3">
                <BookOpen size={20} className="text-blue-600" />
              </div>
              설치 방법
            </h3>
            
            <div className="bg-slate-900 text-slate-200 p-4 rounded-lg font-mono text-sm overflow-auto mb-4">
              <pre>
{`# 가상 환경 생성 (선택 사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate  # Windows

# 필요한 패키지 설치
pip install torch torchvision
pip install opencv-python numpy hydra-core omegaconf
pip install requests tqdm matplotlib

# SAM2 모델 설치
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# ByteTrack 설치
pip install git+https://github.com/FoundationVision/ByteTrack.git`}
              </pre>
            </div>
            
            <p className="text-gray-700">
              위 명령어를 순서대로 실행하여 필요한 패키지와 모델을 설치합니다. 
              CUDA가 지원되는 환경에서는 GPU 가속을 활용할 수 있습니다.
            </p>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-md">
            <h3 className="text-xl font-bold mb-4 flex items-center">
              <div className="bg-green-100 p-2 rounded-full mr-3">
                <Settings size={20} className="text-green-600" />
              </div>
              실행 방법
            </h3>
            
            <div className="bg-slate-900 text-slate-200 p-4 rounded-lg font-mono text-sm overflow-auto mb-4">
              <pre>
{`# 기본 설정으로 실행
python run_pipeline.py

# 추적할 객체 수 변경
python run_pipeline.py run.num_objects=10

# 다른 비디오 파일 사용
python run_pipeline.py run.video_path=/path/to/your/video.mp4

# 디스플레이 끄기
python run_pipeline.py run.display=false

# 여러 설정 동시 변경
python run_pipeline.py models=sam2 trackers=bytetrack run.num_objects=5 visualization.trajectory.show=true`}
              </pre>
            </div>
            
            <p className="text-gray-700">
              Hydra 프레임워크를 통해 다양한 설정을 명령줄에서 쉽게 변경할 수 있습니다. 
              결과는 outputs/ 디렉토리에 저장됩니다.
            </p>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-md">
            <h3 className="text-xl font-bold mb-4 flex items-center">
              <div className="bg-purple-100 p-2 rounded-full mr-3">
                <HelpCircle size={20} className="text-purple-600" />
              </div>
              자주 묻는 질문
            </h3>
            
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold mb-1">어떤 비디오 형식을 지원하나요?</h4>
                <p className="text-gray-700 text-sm">
                  MP4, AVI, MOV 등 OpenCV에서 지원하는 모든 비디오 형식을 사용할 수 있습니다.
                </p>
              </div>
              
              <div>
                <h4 className="font-semibold mb-1">웹캠으로 실시간 추적이 가능한가요?</h4>
                <p className="text-gray-700 text-sm">
                  네, run.mode=webcam 설정으로 웹캠 입력을 사용할 수 있습니다.
                </p>
              </div>
              
              <div>
                <h4 className="font-semibold mb-1">어떤 하드웨어가 필요한가요?</h4>
                <p className="text-gray-700 text-sm">
                  최소 8GB RAM과 CUDA 지원 GPU를 권장합니다. CPU에서도 실행 가능하지만 처리 속도가 느립니다.
                </p>
              </div>
              
              <div>
                <h4 className="font-semibold mb-1">특정 객체만 추적할 수 있나요?</h4>
                <p className="text-gray-700 text-sm">
                  네, 프롬프트를 사용하여 특정 객체를 지정할 수 있습니다. configs/models/sam2.yaml 파일에서 설정할 수 있습니다.
                </p>
              </div>
              
              <div>
                <h4 className="font-semibold mb-1">결과를 어떻게 저장하나요?</h4>
                <p className="text-gray-700 text-sm">
                  run.save_video=true 설정으로 결과 영상을 저장할 수 있습니다. 결과는 outputs/ 디렉토리에 저장됩니다.
                </p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-xl shadow-md">
          <h3 className="text-xl font-bold mb-4">고급 사용법</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h4 className="text-lg font-semibold mb-3">성능 최적화</h4>
              <p className="text-gray-700 mb-3">
                다양한 설정으로 성능을 테스트하고 최적화할 수 있습니다:
              </p>
              <div className="bg-slate-900 text-slate-200 p-4 rounded-lg font-mono text-sm overflow-auto">
                <pre>
{`# 성능 테스트 실행
python src/utils/performance_test.py

# 더 작은 모델 사용
python run_pipeline.py models.model_type=vit_b

# 추적기 버전 변경
python run_pipeline.py trackers.version=s

# 메모리 사용량 조절
python run_pipeline.py models.video.memory_size=3`}
                </pre>
              </div>
            </div>
            
            <div>
              <h4 className="text-lg font-semibold mb-3">커스텀 모델 통합</h4>
              <p className="text-gray-700 mb-3">
                자체 세그멘테이션 모델이나 추적 알고리즘을 통합할 수 있습니다:
              </p>
              <ol className="list-decimal pl-6 space-y-2 text-gray-700">
                <li>src/models/ 또는 src/trackers/ 디렉토리에 새 모듈 추가</li>
                <li>기존 인터페이스를 따르는 클래스 구현</li>
                <li>configs/ 디렉토리에 해당 설정 파일 추가</li>
                <li>config.yaml에 새 모듈 등록</li>
              </ol>
              <p className="text-gray-700 mt-3">
                모듈화된 설계로 각 구성 요소를 쉽게 교체하거나 확장할 수 있습니다.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Usage;
