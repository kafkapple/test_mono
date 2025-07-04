import React from 'react';
import { FileCode, GitBranch, Layers } from 'lucide-react';

const Overview: React.FC = () => {
  return (
    <section id="overview" className="py-16 bg-white">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">프로젝트 개요</h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            SAM2와 ByteTrack을 결합한 객체 추적 시스템은 픽셀 수준 세그멘테이션과 다중 객체 추적을 통합하여 
            영상에서 객체를 효과적으로 추적하는 파이프라인입니다.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="bg-slate-50 p-6 rounded-xl shadow-md">
            <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mb-4 mx-auto">
              <Layers size={32} className="text-blue-600" />
            </div>
            <h3 className="text-xl font-bold text-center mb-3">SAM2 세그멘테이션</h3>
            <p className="text-gray-600">
              Meta AI의 SAM2(Segment Anything Model 2)는 이미지와 비디오에서 프롬프트 기반 세그멘테이션을 지원합니다. 
              내장된 메모리 모듈을 통해 비디오에서 객체가 일시적으로 사라져도 추적이 가능합니다.
            </p>
          </div>
          
          <div className="bg-slate-50 p-6 rounded-xl shadow-md">
            <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mb-4 mx-auto">
              <GitBranch size={32} className="text-green-600" />
            </div>
            <h3 className="text-xl font-bold text-center mb-3">ByteTrack 추적</h3>
            <p className="text-gray-600">
              ByteTrack은 모든 검출 박스를 연관시켜 추적하는 방식으로, 낮은 점수의 검출 박스도 활용하여 
              가려진 객체나 단편화된 궤적 문제를 해결합니다. MOT17, MOT20 데이터셋에서 SOTA 성능을 달성했습니다.
            </p>
          </div>
          
          <div className="bg-slate-50 p-6 rounded-xl shadow-md">
            <div className="bg-purple-100 w-16 h-16 rounded-full flex items-center justify-center mb-4 mx-auto">
              <FileCode size={32} className="text-purple-600" />
            </div>
            <h3 className="text-xl font-bold text-center mb-3">Hydra 설정 관리</h3>
            <p className="text-gray-600">
              Hydra 프레임워크를 통해 다양한 설정을 쉽게 관리하고 실험할 수 있습니다. 
              모델 파라미터, 추적 알고리즘 설정, 시각화 옵션 등을 유연하게 조정할 수 있어 
              다양한 환경과 요구사항에 맞게 시스템을 최적화할 수 있습니다.
            </p>
          </div>
        </div>
        
        <div className="mt-12 bg-slate-100 p-6 rounded-xl">
          <h3 className="text-xl font-bold mb-4">주요 특징</h3>
          <ul className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <li className="flex items-start">
              <div className="bg-blue-500 text-white p-1 rounded-full mr-3 mt-1">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <span>픽셀 수준 세그멘테이션으로 정확한 객체 마스크 생성</span>
            </li>
            <li className="flex items-start">
              <div className="bg-blue-500 text-white p-1 rounded-full mr-3 mt-1">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <span>비디오 프레임 간 객체 ID 일관성 유지</span>
            </li>
            <li className="flex items-start">
              <div className="bg-blue-500 text-white p-1 rounded-full mr-3 mt-1">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <span>객체 가림 상황에서도 강인한 추적 성능</span>
            </li>
            <li className="flex items-start">
              <div className="bg-blue-500 text-white p-1 rounded-full mr-3 mt-1">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <span>모듈화된 설계로 각 구성 요소 쉽게 교체 가능</span>
            </li>
            <li className="flex items-start">
              <div className="bg-blue-500 text-white p-1 rounded-full mr-3 mt-1">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <span>Hydra 기반 설정 관리로 유연한 실험 환경</span>
            </li>
            <li className="flex items-start">
              <div className="bg-blue-500 text-white p-1 rounded-full mr-3 mt-1">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <span>결과 시각화 및 영상 저장 기능 내장</span>
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
};

export default Overview;
