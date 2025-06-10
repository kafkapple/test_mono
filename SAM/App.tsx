import './App.css';
import Header from './components/Header';
import Hero from './components/Hero';
import Overview from './components/Overview';
import Algorithms from './components/Algorithms';
import Implementation from './components/Implementation';
import Results from './components/Results';
import Usage from './components/Usage';

function App() {
  return (
    <div className="App">
      <Header />
      <Hero />
      <Overview />
      <Algorithms />
      <Implementation />
      <Results />
      <Usage />
      
      <footer className="bg-slate-900 text-white py-8">
        <div className="container mx-auto px-4 text-center">
          <p className="mb-2">SAM-MOT 객체 추적 시스템 © 2025</p>
          <p className="text-slate-400 text-sm">
            SAM2와 ByteTrack을 결합한 픽셀 수준 세그멘테이션 및 다중 객체 추적 파이프라인
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
