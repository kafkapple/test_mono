import React from 'react';
import { Image, Play, BarChart } from 'lucide-react';

const Results: React.FC = () => {
  return (
    <section id="results" className="py-16 bg-slate-50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">결과 시각화</h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            SAM-MOT 객체 추적 시스템의 실행 결과와 시각화 예시를 살펴봅니다.
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          <div className="bg-white p-6 rounded-xl shadow-md">
            <h3 className="text-xl font-bold mb-4 flex items-center">
              <div className="bg-blue-100 p-2 rounded-full mr-3">
                <Image size={20} className="text-blue-600" />
              </div>
              세그멘테이션 결과
            </h3>
            
            <div className="mb-6">
              <div className="bg-slate-100 p-2 rounded-lg mb-2">
                <img 
                  src="/images/segmentation_result.jpg" 
                  alt="세그멘테이션 결과" 
                  className="w-full h-auto rounded-lg shadow-md"
                />
              </div>
              <p className="text-sm text-gray-500 text-center">SAM2 모델을 사용한 픽셀 수준 세그멘테이션 결과</p>
            </div>
            
            <p className="text-gray-700 mb-4">
              SAM2 모델은 다양한 객체에 대해 정확한 픽셀 수준 마스크를 생성합니다. 
              위 이미지는 여러 객체가 포함된 장면에서 각 객체를 서로 다른 색상으로 
              세그멘테이션한 결과를 보여줍니다.
            </p>
            
            <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
              <h4 className="text-lg font-semibold mb-2">주요 특징</h4>
              <ul className="list-disc pl-6 space-y-1 text-gray-700">
                <li>정확한 객체 경계 검출</li>
                <li>복잡한 형태의 객체도 정확히 세그멘테이션</li>
                <li>겹치는 객체 구분 능력</li>
                <li>다양한 객체 유형 지원</li>
              </ul>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-md">
            <h3 className="text-xl font-bold mb-4 flex items-center">
              <div className="bg-green-100 p-2 rounded-full mr-3">
                <Play size={20} className="text-green-600" />
              </div>
              객체 추적 결과
            </h3>
            
            <div className="mb-6">
              <div className="bg-slate-100 p-2 rounded-lg mb-2">
                <img 
                  src="/images/tracking_result.jpg" 
                  alt="객체 추적 결과" 
                  className="w-full h-auto rounded-lg shadow-md"
                />
              </div>
              <p className="text-sm text-gray-500 text-center">ByteTrack을 통한 객체 추적 및 ID 할당 결과</p>
            </div>
            
            <p className="text-gray-700 mb-4">
              ByteTrack 알고리즘은 세그멘테이션된 객체에 일관된 ID를 할당하고 
              프레임 간 객체를 추적합니다. 위 이미지는 여러 객체에 고유 ID가 할당되고 
              궤적이 시각화된 결과를 보여줍니다.
            </p>
            
            <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
              <h4 className="text-lg font-semibold mb-2">주요 특징</h4>
              <ul className="list-disc pl-6 space-y-1 text-gray-700">
                <li>객체별 고유 ID 할당 및 유지</li>
                <li>객체 궤적 시각화</li>
                <li>가림 상황에서도 ID 유지</li>
                <li>다양한 움직임 패턴 추적 가능</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-xl shadow-md mb-12">
          <h3 className="text-xl font-bold mb-4 flex items-center">
            <div className="bg-purple-100 p-2 rounded-full mr-3">
              <Play size={20} className="text-purple-600" />
            </div>
            통합 파이프라인 결과 영상
          </h3>
          
          <div className="aspect-w-16 aspect-h-9 mb-6">
            <div className="w-full h-0 pb-[56.25%] relative bg-slate-200 rounded-lg">
              <div className="absolute inset-0 flex items-center justify-center">
                <Play size={48} className="text-slate-400" />
                <span className="ml-2 text-slate-500">영상 미리보기</span>
              </div>
            </div>
          </div>
          
          <p className="text-gray-700 mb-4">
            위 영상은 SAM-MOT 통합 파이프라인을 실행한 결과를 보여줍니다. 
            각 프레임에서 객체가 세그멘테이션되고, 프레임 간 일관된 ID로 추적되며, 
            객체의 궤적이 시각화됩니다. 이 파이프라인은 다양한 환경과 객체 유형에 
            적용할 수 있으며, Hydra 설정을 통해 쉽게 조정할 수 있습니다.
          </p>
        </div>
        
        <div className="bg-white p-6 rounded-xl shadow-md">
          <h3 className="text-xl font-bold mb-4 flex items-center">
            <div className="bg-amber-100 p-2 rounded-full mr-3">
              <BarChart size={20} className="text-amber-600" />
            </div>
            성능 분석
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h4 className="text-lg font-semibold mb-3">처리 속도 (FPS)</h4>
              <div className="bg-slate-100 p-4 rounded-lg">
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-medium">SAM2 (vit_b)</span>
                      <span className="text-sm font-medium">15 FPS</span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2.5">
                      <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: '50%' }}></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-medium">ByteTrack (x)</span>
                      <span className="text-sm font-medium">30 FPS</span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2.5">
                      <div className="bg-green-600 h-2.5 rounded-full" style={{ width: '75%' }}></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-medium">통합 파이프라인</span>
                      <span className="text-sm font-medium">12 FPS</span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2.5">
                      <div className="bg-purple-600 h-2.5 rounded-full" style={{ width: '40%' }}></div>
                    </div>
                  </div>
                </div>
                <p className="text-sm text-gray-500 mt-3">
                  * NVIDIA V100 GPU 기준, 640x480 해상도 영상
                </p>
              </div>
            </div>
            
            <div>
              <h4 className="text-lg font-semibold mb-3">정확도 지표</h4>
              <div className="bg-slate-100 p-4 rounded-lg">
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-medium">세그멘테이션 IoU</span>
                      <span className="text-sm font-medium">0.85</span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2.5">
                      <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: '85%' }}></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-medium">추적 MOTA</span>
                      <span className="text-sm font-medium">78.5%</span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2.5">
                      <div className="bg-green-600 h-2.5 rounded-full" style={{ width: '78.5%' }}></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-medium">ID 스위칭 (낮을수록 좋음)</span>
                      <span className="text-sm font-medium">12</span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2.5">
                      <div className="bg-amber-600 h-2.5 rounded-full" style={{ width: '20%' }}></div>
                    </div>
                  </div>
                </div>
                <p className="text-sm text-gray-500 mt-3">
                  * MOT17 데이터셋 기준 평가 결과
                </p>
              </div>
            </div>
          </div>
          
          <div className="mt-6 bg-slate-50 p-4 rounded-lg border border-slate-200">
            <h4 className="text-lg font-semibold mb-2">최적 설정 조합</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h5 className="font-medium mb-1">세그멘테이션 설정</h5>
                <ul className="list-disc pl-6 space-y-1 text-gray-700 text-sm">
                  <li>모델: SAM2 (vit_b)</li>
                  <li>pred_iou_thresh: 0.88</li>
                  <li>stability_score_thresh: 0.95</li>
                  <li>메모리 크기: 5 프레임</li>
                </ul>
              </div>
              <div>
                <h5 className="font-medium mb-1">추적 설정</h5>
                <ul className="list-disc pl-6 space-y-1 text-gray-700 text-sm">
                  <li>모델: ByteTrack (x)</li>
                  <li>track_thresh: 0.5</li>
                  <li>match_thresh: 0.8</li>
                  <li>마스크 IoU 가중치: 0.5</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Results;
