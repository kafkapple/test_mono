import React from 'react';
import { Code, Image, Play, Settings } from 'lucide-react';

const Hero: React.FC = () => {
  return (
    <section className="bg-gradient-to-b from-slate-900 to-slate-800 text-white py-20">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row items-center">
          <div className="md:w-1/2 mb-10 md:mb-0">
            <h1 className="text-4xl md:text-5xl font-bold mb-6">
              SAM2 + ByteTrack 객체 추적 시스템
            </h1>
            <p className="text-xl mb-8 text-slate-300">
              픽셀 수준 세그멘테이션과 다중 객체 추적을 결합한 고성능 영상 분석 파이프라인
            </p>
            <div className="flex flex-wrap gap-4">
              <a 
                href="#implementation" 
                className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-colors"
              >
                구현 살펴보기
              </a>
              <a 
                href="#results" 
                className="bg-transparent border border-white hover:bg-white hover:text-slate-900 text-white font-bold py-3 px-6 rounded-lg transition-colors"
              >
                결과 확인하기
              </a>
            </div>
          </div>
          <div className="md:w-1/2">
            <div className="bg-slate-800 p-6 rounded-xl shadow-2xl border border-slate-700">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-700 p-4 rounded-lg flex flex-col items-center">
                  <Image size={40} className="text-blue-400 mb-2" />
                  <h3 className="font-bold">픽셀 수준 세그멘테이션</h3>
                </div>
                <div className="bg-slate-700 p-4 rounded-lg flex flex-col items-center">
                  <Play size={40} className="text-green-400 mb-2" />
                  <h3 className="font-bold">비디오 객체 추적</h3>
                </div>
                <div className="bg-slate-700 p-4 rounded-lg flex flex-col items-center">
                  <Code size={40} className="text-purple-400 mb-2" />
                  <h3 className="font-bold">모듈화된 설계</h3>
                </div>
                <div className="bg-slate-700 p-4 rounded-lg flex flex-col items-center">
                  <Settings size={40} className="text-amber-400 mb-2" />
                  <h3 className="font-bold">유연한 설정</h3>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
