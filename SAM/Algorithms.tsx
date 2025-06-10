import React from 'react';
import { Code2, FileCode } from 'lucide-react';

const Algorithms: React.FC = () => {
  return (
    <section id="algorithms" className="py-16 bg-slate-50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">알고리즘 설명</h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            SAM2와 ByteTrack의 핵심 알고리즘과 이들이 어떻게 통합되어 작동하는지 살펴봅니다.
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          <div className="bg-white p-6 rounded-xl shadow-md">
            <h3 className="text-2xl font-bold mb-4 flex items-center">
              <div className="bg-blue-100 p-2 rounded-full mr-3">
                <Code2 size={24} className="text-blue-600" />
              </div>
              SAM2 (Segment Anything Model 2)
            </h3>
            
            <div className="mb-6">
              <h4 className="text-lg font-semibold mb-2">핵심 특징</h4>
              <ul className="list-disc pl-6 space-y-2 text-gray-700">
                <li>이미지와 비디오 모두에서 프롬프트 기반 세그멘테이션 지원</li>
                <li>스트리밍 아키텍처로 비디오 프레임을 실시간으로 처리</li>
                <li>메모리 모듈을 통해 비디오에서 객체가 일시적으로 사라져도 추적 가능</li>
                <li>마스크 예측에 대한 추가 프롬프트 기반 수정 지원</li>
                <li>다양한 입력 프롬프트(점, 박스 등) 지원</li>
              </ul>
            </div>
            
            <div className="mb-6">
              <h4 className="text-lg font-semibold mb-2">작동 원리</h4>
              <p className="text-gray-700">
                SAM2는 비전 트랜스포머(ViT) 기반 인코더와 마스크 디코더로 구성됩니다. 
                비디오 처리 시 프레임별 세그멘테이션 결과를 메모리 모듈에 저장하고, 
                이를 활용해 객체의 시간적 일관성을 유지합니다. 
                이미지 임베딩과 프롬프트를 결합하여 픽셀 수준의 세그멘테이션 마스크를 생성합니다.
              </p>
            </div>
            
            <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
              <h4 className="text-lg font-semibold mb-2">기술적 장점</h4>
              <ul className="list-disc pl-6 space-y-1 text-gray-700">
                <li>단일 프레임과 비디오 모두에 통합된 접근 방식</li>
                <li>메모리 효율적인 스트리밍 아키텍처</li>
                <li>다양한 시각적 도메인에서 강력한 성능</li>
                <li>실시간 처리 가능한 효율적인 설계</li>
              </ul>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-md">
            <h3 className="text-2xl font-bold mb-4 flex items-center">
              <div className="bg-green-100 p-2 rounded-full mr-3">
                <FileCode size={24} className="text-green-600" />
              </div>
              ByteTrack
            </h3>
            
            <div className="mb-6">
              <h4 className="text-lg font-semibold mb-2">핵심 특징</h4>
              <ul className="list-disc pl-6 space-y-2 text-gray-700">
                <li>모든 검출 박스를 연관시켜 추적하는 방식</li>
                <li>낮은 점수의 검출 박스도 활용하여 가려진 객체나 단편화된 궤적 문제 해결</li>
                <li>MOT17, MOT20 데이터셋에서 SOTA 성능 달성</li>
                <li>다양한 크기의 모델 제공(nano, tiny, s, m, l, x)</li>
                <li>높은 정확도와 빠른 처리 속도</li>
              </ul>
            </div>
            
            <div className="mb-6">
              <h4 className="text-lg font-semibold mb-2">작동 원리</h4>
              <p className="text-gray-700">
                ByteTrack은 YOLOX 객체 탐지기와 연관성 기반 추적 알고리즘을 결합합니다. 
                높은 점수의 검출 결과와 낮은 점수의 검출 결과를 분리하여 처리하고, 
                칼만 필터와 헝가리안 알고리즘을 사용해 객체 ID를 일관되게 유지합니다. 
                이를 통해 가림 상황에서도 강인한 추적 성능을 제공합니다.
              </p>
            </div>
            
            <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
              <h4 className="text-lg font-semibold mb-2">기술적 장점</h4>
              <ul className="list-disc pl-6 space-y-1 text-gray-700">
                <li>낮은 점수의 검출 결과도 활용하는 효과적인 연관 전략</li>
                <li>다양한 객체 탐지기와 호환 가능한 유연한 설계</li>
                <li>실시간 처리 가능한 높은 효율성</li>
                <li>가림 상황에서도 ID 스위칭 최소화</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-xl shadow-md">
          <h3 className="text-2xl font-bold mb-4">통합 파이프라인</h3>
          
          <div className="mb-6">
            <p className="text-gray-700 mb-4">
              SAM2와 ByteTrack을 결합한 통합 파이프라인은 다음과 같은 단계로 작동합니다:
            </p>
            
            <ol className="list-decimal pl-6 space-y-3 text-gray-700">
              <li>
                <strong>세그멘테이션 단계:</strong> SAM2를 사용하여 각 비디오 프레임에서 객체의 픽셀 수준 마스크를 생성합니다.
                이 과정에서 프롬프트(점, 박스 등)를 사용하거나 자동 모드로 객체를 세그멘테이션합니다.
              </li>
              <li>
                <strong>마스크 처리:</strong> 세그멘테이션 마스크에서 바운딩 박스를 추출하고, 각 객체의 면적, 점수 등 메타데이터를 계산합니다.
              </li>
              <li>
                <strong>객체 추적:</strong> ByteTrack 알고리즘을 사용하여 현재 프레임의 객체와 이전 프레임의 객체를 연관시킵니다.
                이 과정에서 바운딩 박스 IoU와 마스크 IoU를 함께 고려하여 더 정확한 연관성을 계산합니다.
              </li>
              <li>
                <strong>ID 할당:</strong> 연관된 객체에 일관된 ID를 할당하고, 새로운 객체에는 새 ID를 부여합니다.
              </li>
              <li>
                <strong>궤적 관리:</strong> 각 객체의 궤적(위치 이력)을 관리하고, 일정 시간 동안 보이지 않는 객체는 추적에서 제외합니다.
              </li>
              <li>
                <strong>결과 시각화:</strong> 추적 결과를 시각화하여 각 객체의 마스크, 바운딩 박스, ID, 궤적 등을 표시합니다.
              </li>
            </ol>
          </div>
          
          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
            <h4 className="text-lg font-semibold mb-2">통합의 이점</h4>
            <ul className="list-disc pl-6 space-y-1 text-gray-700">
              <li>픽셀 수준 세그멘테이션의 정확성과 객체 추적의 시간적 일관성 결합</li>
              <li>마스크 IoU를 활용한 더 정확한 객체 연관성 계산</li>
              <li>가림 상황에서도 객체 ID 유지 능력 향상</li>
              <li>Hydra 기반 설정 관리로 다양한 실험 및 최적화 용이</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Algorithms;
