import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileVideo, AlertCircle, CheckCircle } from 'lucide-react';
import axios from 'axios';

const VideoUploadComponent = () => {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});
  const [uploadErrors, setUploadErrors] = useState({});

  const onDrop = useCallback(acceptedFiles => {
    // 파일 목록에 추가
    const newFiles = acceptedFiles.map(file => ({
      file,
      id: `${file.name}-${Date.now()}`,
      status: 'ready', // ready, uploading, success, error
      progress: 0
    }));
    
    setUploadedFiles(prev => [...prev, ...newFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv']
    },
    maxSize: 1024 * 1024 * 500, // 500MB 제한
    multiple: true
  });

  const uploadFile = async (fileItem) => {
    const { file, id } = fileItem;
    const formData = new FormData();
    formData.append('video', file);
    
    try {
      setUploading(true);
      
      // 해당 파일 상태 업데이트
      setUploadedFiles(prev => 
        prev.map(item => 
          item.id === id 
            ? { ...item, status: 'uploading' } 
            : item
        )
      );
      
      // 업로드 요청
      const response = await axios.post('/api/videos/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          
          // 진행률 업데이트
          setUploadProgress(prev => ({
            ...prev,
            [id]: percentCompleted
          }));
          
          // 파일 목록의 진행률도 업데이트
          setUploadedFiles(prev => 
            prev.map(item => 
              item.id === id 
                ? { ...item, progress: percentCompleted } 
                : item
            )
          );
        }
      });
      
      // 업로드 성공
      setUploadedFiles(prev => 
        prev.map(item => 
          item.id === id 
            ? { 
                ...item, 
                status: 'success',
                videoId: response.data.videoId, // 서버에서 반환한 비디오 ID
                progress: 100 
              } 
            : item
        )
      );
      
      // 에러 상태 제거
      setUploadErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[id];
        return newErrors;
      });
      
      return response.data;
      
    } catch (error) {
      console.error('Upload error:', error);
      
      // 업로드 실패
      setUploadedFiles(prev => 
        prev.map(item => 
          item.id === id 
            ? { ...item, status: 'error' } 
            : item
        )
      );
      
      // 에러 메시지 저장
      setUploadErrors(prev => ({
        ...prev,
        [id]: error.response?.data?.message || '업로드 중 오류가 발생했습니다.'
      }));
      
      return null;
    }
  };

  const handleUploadAll = async () => {
    const readyFiles = uploadedFiles.filter(file => file.status === 'ready');
    
    if (readyFiles.length === 0) return;
    
    setUploading(true);
    
    // 모든 파일 순차적으로 업로드
    for (const fileItem of readyFiles) {
      await uploadFile(fileItem);
    }
    
    setUploading(false);
  };

  const removeFile = (id) => {
    setUploadedFiles(prev => prev.filter(item => item.id !== id));
    
    // 진행률 및 에러 상태에서도 제거
    setUploadProgress(prev => {
      const newProgress = { ...prev };
      delete newProgress[id];
      return newProgress;
    });
    
    setUploadErrors(prev => {
      const newErrors = { ...prev };
      delete newErrors[id];
      return newErrors;
    });
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'ready':
        return <FileVideo className="text-slate-400" />;
      case 'uploading':
        return <Upload className="text-blue-500 animate-pulse" />;
      case 'success':
        return <CheckCircle className="text-green-500" />;
      case 'error':
        return <AlertCircle className="text-red-500" />;
      default:
        return <FileVideo className="text-slate-400" />;
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6">영상 업로드</h2>
      
      {/* 드래그 앤 드롭 영역 */}
      <div 
        {...getRootProps()} 
        className={`border-2 border-dashed rounded-lg p-8 mb-6 text-center cursor-pointer transition-colors
          ${isDragActive 
            ? 'border-blue-500 bg-blue-50' 
            : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'}`}
      >
        <input {...getInputProps()} />
        <Upload className="mx-auto h-12 w-12 text-slate-400 mb-4" />
        
        {isDragActive ? (
          <p className="text-lg text-blue-600">파일을 여기에 놓으세요...</p>
        ) : (
          <div>
            <p className="text-lg text-slate-600 mb-2">
              영상 파일을 끌어다 놓거나 클릭하여 선택하세요
            </p>
            <p className="text-sm text-slate-500">
              지원 형식: MP4, AVI, MOV, MKV (최대 500MB)
            </p>
          </div>
        )}
      </div>
      
      {/* 업로드 버튼 */}
      {uploadedFiles.some(file => file.status === 'ready') && (
        <div className="mb-6">
          <button
            onClick={handleUploadAll}
            disabled={uploading}
            className={`px-4 py-2 rounded-lg font-medium ${
              uploading
                ? 'bg-slate-300 text-slate-500 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {uploading ? '업로드 중...' : '모든 파일 업로드'}
          </button>
        </div>
      )}
      
      {/* 파일 목록 */}
      {uploadedFiles.length > 0 && (
        <div className="border rounded-lg overflow-hidden">
          <div className="bg-slate-100 px-4 py-3 border-b">
            <h3 className="font-semibold">업로드 파일 목록</h3>
          </div>
          <ul className="divide-y">
            {uploadedFiles.map((fileItem) => (
              <li key={fileItem.id} className="px-4 py-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(fileItem.status)}
                    <div>
                      <p className="font-medium truncate max-w-md">{fileItem.file.name}</p>
                      <p className="text-sm text-slate-500">
                        {(fileItem.file.size / (1024 * 1024)).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-4">
                    {/* 진행 상태 표시 */}
                    {fileItem.status === 'uploading' && (
                      <div className="w-32">
                        <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-blue-500 rounded-full" 
                            style={{ width: `${fileItem.progress}%` }}
                          ></div>
                        </div>
                        <p className="text-xs text-right mt-1">{fileItem.progress}%</p>
                      </div>
                    )}
                    
                    {/* 상태 텍스트 */}
                    {fileItem.status === 'success' && (
                      <span className="text-sm text-green-600">업로드 완료</span>
                    )}
                    
                    {fileItem.status === 'error' && (
                      <span className="text-sm text-red-600">
                        {uploadErrors[fileItem.id] || '오류 발생'}
                      </span>
                    )}
                    
                    {/* 삭제 버튼 */}
                    {fileItem.status !== 'uploading' && (
                      <button
                        onClick={() => removeFile(fileItem.id)}
                        className="text-slate-400 hover:text-red-500"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                        </svg>
                      </button>
                    )}
                  </div>
                </div>
                
                {/* 에러 메시지 */}
                {fileItem.status === 'error' && uploadErrors[fileItem.id] && (
                  <div className="mt-2 text-sm text-red-600 bg-red-50 p-2 rounded">
                    {uploadErrors[fileItem.id]}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default VideoUploadComponent;
