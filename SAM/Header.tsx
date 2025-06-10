import React from 'react';
import { Menu } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-slate-900 text-white shadow-lg">
      <div className="container mx-auto px-4 py-6 flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center">
            <span className="font-bold text-xl">S</span>
          </div>
          <h1 className="text-2xl font-bold">SAM-MOT Tracking</h1>
        </div>
        
        <nav className="hidden md:flex space-x-8">
          <a href="#overview" className="hover:text-blue-400 transition-colors">개요</a>
          <a href="#algorithms" className="hover:text-blue-400 transition-colors">알고리즘</a>
          <a href="#implementation" className="hover:text-blue-400 transition-colors">구현</a>
          <a href="#results" className="hover:text-blue-400 transition-colors">결과</a>
          <a href="#usage" className="hover:text-blue-400 transition-colors">사용법</a>
        </nav>
        
        <div className="md:hidden">
          <button className="p-2 focus:outline-none">
            <Menu size={24} />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
