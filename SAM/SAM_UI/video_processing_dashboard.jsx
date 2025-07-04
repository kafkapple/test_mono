import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Clock, CheckCircle, XCircle, Download, RefreshCw } from 'lucide-react';

const VideoProcessingDashboard = () => {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshInterval, setRefreshInterval] = useState(null);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoDetails, setVideoDetails] = useState(null);
  const [detailsLoading, setDetailsLoading] = useState(false);

  // 비디오 목록 조회
  const fetchVideos = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/videos');
      setVideos(response.data.videos);
      setError(null);
    } catch (err) {
      console.error('Error fetching videos:', err);
      setError('비디오 목록을 불러오는 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  // 비디오 상세 정보 조회
  const fetchVideoDetails = async (videoId) => {
    try {
      setDetailsLoading(true);
      const response = await axios.get(`/api/videos/${videoId}`);
      setVideoDetails(response.data.video);
      
      // 처리 중인 비디오인 경우 상태 업데이트를 위한 폴링 시작
      if (response.data.video.status === 'processing') {
        startStatusPolling(videoId);
      } else {
        stopStatusPolling();
      }
    } catch (err) {
      console.error('Error fetching video details:', err);
      setError('비디오 상세 정보를 불러오는 중 오류가 발생했습니다.');
    } finally {
      setDetailsLoading(false);
    }
  };

  // 비디오 상태 폴링 시작
  const startStatusPolling = (videoId) => {
    // 기존 폴링 중지
    stopStatusPolling();
    
    // 새 폴링 시작 (5초 간격)
    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`/api/videos/${videoId}/status`);
        const status = response.data.status;
        
        // 상태 업데이트
        setVideoDetails(prev => prev ? { ...prev, status } : null);
        
        // 처리 완료 또는 오류 상태면 폴링 중지
        if (status === 'completed' || status === 'failed') {
          stopStatusPolling();
          // 완료된 경우 전체 상세 정보 다시 조회
          if (status === 'completed') {
            fetchVideoDetails(videoId);
          }
        }
      } catch (err) {
        console.error('Error polling status:', err);
      }
    }, 5000);
    
    setRefreshInterval(interval);
  };

  // 비디오 상태 폴링 중지
  const stopStatusPolling = () => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
      setRefreshInterval(null);
    }
  };

  // 결과 다운로드
  const downloadResult = (videoId, filename) => {
    window.open(`/api/videos/${videoId}/download`, '_blank');
  };

  // 컴포넌트 마운트 시 비디오 목록 조회
  useEffect(() => {
    fetchVideos();
    
    // 컴포넌트 언마운트 시 폴링 중지
    return () => stopStatusPolling();
  }, []);

  // 비디오 선택 시 상세 정보 조회
  const handleVideoSelect = (videoId) => {
    setSelectedVideo(videoId);
    fetchVideoDetails(videoId);
  };

  // 상태에 따른 아이콘 및 색상
  const getStatusInfo = (status) => {
    switch (status) {
      case 'uploaded':
        return { icon: <Clock className="text-blue-500" />, color: 'text-blue-500', text: '업로드됨' };
      case 'processing':
        return { icon: <RefreshCw className="text-yellow-500 animate-spin" />, color: 'text-yellow-500', text: '처리 중' };
      case 'completed':
        return { icon: <CheckCircle className="text-green-500" />, color: 'text-green-500', text: '완료됨' };
      case 'failed':
        return { icon: <XCircle className="text-red-500" />, color: 'text-red-500', text: '실패' };
      default:
        return { icon: <Clock className="text-gray-500" />, color: 'text-gray-500', text: '알 수 없음' };
    }
  };

  // 날짜 포맷팅
  const formatDate = (dateString) => {
    if (!dateString) return '-';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">영상 처리 대시보드</h2>
        <button 
          onClick={fetchVideos}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center"
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          새로고침
        </button>
      </div>
      
      {loading ? (
        <div className="text-center py-10">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p>비디오 목록을 불러오는 중...</p>
        </div>
      ) : error ? (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <p>{error}</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 비디오 목록 */}
          <div className="lg:col-span-1 border rounded-lg overflow-hidden">
            <div className="bg-slate-100 px-4 py-3 border-b">
              <h3 className="font-semibold">업로드된 영상 목록</h3>
            </div>
            
            {videos.length === 0 ? (
              <div className="p-6 text-center text-gray-500">
                <p>업로드된 영상이 없습니다.</p>
              </div>
            ) : (
              <ul className="divide-y max-h-[600px] overflow-y-auto">
                {videos.map((video) => {
                  const { icon, color } = getStatusInfo(video.status);
                  return (
                    <li 
                      key={video._id} 
                      className={`px-4 py-3 cursor-pointer hover:bg-slate-50 ${selectedVideo === video._id ? 'bg-blue-50' : ''}`}
                      onClick={() => handleVideoSelect(video._id)}
                    >
                      <div className="flex items-center space-x-3">
                        {icon}
                        <div className="flex-1 min-w-0">
                          <p className="font-medium truncate">{video.original_filename}</p>
                          <p className="text-sm text-slate-500">
                            {formatDate(video.upload_time)}
                          </p>
                        </div>
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
          
          {/* 비디오 상세 정보 */}
          <div className="lg:col-span-2 border rounded-lg overflow-hidden">
            <div className="bg-slate-100 px-4 py-3 border-b">
              <h3 className="font-semibold">영상 상세 정보</h3>
            </div>
            
            {!selectedVideo ? (
              <div className="p-10 text-center text-gray-500">
                <p>좌측 목록에서 영상을 선택하세요.</p>
              </div>
            ) : detailsLoading ? (
              <div className="text-center py-10">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                <p>상세 정보를 불러오는 중...</p>
              </div>
            ) : !videoDetails ? (
              <div className="p-6 text-center text-red-500">
                <p>상세 정보를 불러올 수 없습니다.</p>
              </div>
            ) : (
              <div className="p-6">
                <div className="mb-6">
                  <h4 className="text-xl font-semibold mb-4">{videoDetails.original_filename}</h4>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    <div className="bg-slate-50 p-4 rounded-lg">
                      <p className="text-sm text-slate-500 mb-1">상태</p>
                      <p className={`font-medium ${getStatusInfo(videoDetails.status).color}`}>
                        {getStatusInfo(videoDetails.status).text}
                      </p>
                    </div>
                    
                    <div className="bg-slate-50 p-4 rounded-lg">
                      <p className="text-sm text-slate-500 mb-1">파일 크기</p>
                      <p className="font-medium">
                        {videoDetails.file_size ? `${(videoDetails.file_size / (1024 * 1024)).toFixed(2)} MB` : '-'}
                      </p>
                    </div>
                    
                    <div className="bg-slate-50 p-4 rounded-lg">
                      <p className="text-sm text-slate-500 mb-1">업로드 시간</p>
                      <p className="font-medium">{formatDate(videoDetails.upload_time)}</p>
                    </div>
                    
                    <div className="bg-slate-50 p-4 rounded-lg">
                      <p className="text-sm text-slate-500 mb-1">처리 완료 시간</p>
                      <p className="font-medium">{formatDate(videoDetails.processing_end_time)}</p>
                    </div>
                  </div>
                  
                  {/* 오류 메시지 */}
                  {videoDetails.status === 'failed' && videoDetails.error_message && (
                    <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg mb-6">
                      <h5 className="font-semibold mb-2">오류 메시지</h5>
                      <p>{videoDetails.error_message}</p>
                    </div>
                  )}
                  
                  {/* 결과 다운로드 버튼 */}
                  {videoDetails.status === 'completed' && videoDetails.result_path && (
                    <button
                      onClick={() => downloadResult(videoDetails._id, videoDetails.original_filename)}
                      className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      결과 다운로드
                    </button>
                  )}
                </div>
                
                {/* 처리 중인 경우 진행 상태 표시 */}
                {videoDetails.status === 'processing' && (
                  <div className="mt-6">
                    <div className="flex items-center mb-2">
                      <RefreshCw className="w-5 h-5 text-yellow-500 animate-spin mr-2" />
                      <p>영상 처리 중...</p>
                    </div>
                    <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                      <div className="h-full bg-yellow-500 rounded-full animate-pulse" style={{ width: '100%' }}></div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoProcessingDashboard;
