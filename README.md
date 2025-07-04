# Test Mono Repository

이 모노레포는 다양한 AI/ML 프로젝트들을 체계적으로 관리하기 위한 아카이브입니다. 각 프로젝트는 독립적으로 실행 가능하며, 실험적 연구와 학습 목적으로 개발되었습니다.

## 📁 프로젝트 구조

### 🤖 LLM (Large Language Models) 관련 프로젝트

#### `llm/test_llava/`
- **목적**: LLaVA-NeXT-Video 모델을 활용한 이미지/비디오 감정 분석 파이프라인
- **주요 기능**:
  - 이미지 및 비디오 감정 분석
  - 자유형 감정 설명 및 사전 정의된 감정 라벨 분류
  - 효율적인 비디오 프레임 처리
  - 재현 가능성을 위한 랜덤 시드 설정
- **기술 스택**: PyTorch, Transformers, LLaVA-NeXT-Video

#### `llm/test_llm/`
- **목적**: 한국어/영어 텍스트 감정 분석, 요약, 챗봇 기능을 제공하는 LLM 기반 파이프라인
- **주요 기능**:
  - 7가지 감정 분류 (기쁨, 분노, 슬픔, 놀람, 혐오, 두려움, 중립)
  - Arousal-Valence 모델을 통한 감정 강도 및 긍정/부정 측정
  - 텍스트 요약 (논리적 순서 정리, 마크다운 형식)
  - 챗봇 기능 (다국어 지원, 대화 기록 관리)
  - 배치 처리 (CSV 파일, 디렉토리 처리)
- **기술 스택**: PyTorch, Transformers, Hydra, PEFT, BitsAndBytes

#### `llm/prompt_eng/`
- **목적**: 프롬프트 엔지니어링 실험 및 연구
- **주요 기능**:
  - 다양한 프롬프트 템플릿 실험
  - 다중 모델 테스트
  - 웹 인터페이스를 통한 프롬프트 테스트
- **구성**: `prompt_eng_1/`, `prompt_eng_2/` 하위 프로젝트

#### `llm/rag_study/`
- **목적**: RAG (Retrieval-Augmented Generation) 연구 및 구현
- **주요 기능**:
  - 문서 검색 및 생성 시스템
  - 벡터 데이터베이스 활용
  - 지식 기반 질의응답 시스템
- **기술 스택**: Jupyter Notebook, 벡터 DB

#### `llm/0_test_llm_agent/`
- **목적**: 다중 AI 모델 협업 시스템
- **주요 기능**:
  - 다중 AI 모델 통합 (OpenAI, Perplexity, Google Gemini, Ollama)
  - ReACT 및 Tool-calling 에이전트 지원
  - 다양한 도구 통합 (계산기, 검색, 텍스트 분석)
  - 모델 간 협업 및 작업 위임
- **기술 스택**: OpenAI API, Perplexity API, Google API, Ollama

### 🎯 Computer Vision 관련 프로젝트

#### `SAM/SAM_UI/`
- **목적**: SAM2 기반 객체 세그멘테이션과 ByteTrack 기반 다중 객체 추적 통합 시스템
- **주요 기능**:
  - SAM2 기반 픽셀 수준 세그멘테이션
  - ByteTrack 기반 다중 객체 추적
  - 세그멘테이션과 추적 결과 통합
  - 객체 ID, 마스크, 바운딩 박스, 궤적 시각화
  - 웹 인터페이스를 통한 비디오 업로드 및 처리
- **기술 스택**: SAM2, ByteTrack, OpenCV, React, Flask

### 🧠 딥러닝 프레임워크

#### `0_pytorch_integrated/`
- **목적**: PyTorch를 활용한 딥러닝 모델 실험 및 시각화를 위한 통합 프레임워크
- **주요 기능**:
  - 다양한 모델 아키텍처 지원 (AE, VAE, VQ-VAE, CLIP, Flamingo 등)
  - 다양한 데이터셋 지원 (이미지, 텍스트, 멀티모달)
  - 시각화 도구 (임베딩 시각화, 재구성 비교, 생성 샘플)
  - 실험 자동화 (모델 학습, 평가, 비교 실험)
  - 웹 인터페이스를 통한 실험 및 시각화
- **기술 스택**: PyTorch, Flask, t-SNE, PCA, UMAP

### 🎵 음성 처리

#### `0_stt/`
- **목적**: 음성-텍스트 변환(STT) 및 텍스트-음성 변환(TTS) 실험
- **주요 기능**:
  - 음성 파일을 텍스트로 변환
  - 텍스트를 음성으로 변환
  - 연속 잠재 공간에서 LLM 추론 관련 연구 자료
- **기술 스택**: 음성 처리 라이브러리

### 🧮 기타 연구 프로젝트

#### `behvior_ssm/`
- **목적**: 동물 행동 분석을 위한 State Space Model 연구
- **주요 기능**:
  - 동물 행동 패턴 분석
  - 상태 공간 모델을 통한 시계열 데이터 처리
- **기술 스택**: Python, 수학적 모델링

## 🚀 사용 방법

각 프로젝트는 독립적으로 실행 가능합니다. 특정 프로젝트를 실행하려면 해당 디렉토리로 이동하여 README.md 파일의 지침을 따르세요.

### 예시:
```bash
# LLM 감정 분석 프로젝트 실행
cd llm/test_llm
python main.py

# SAM 객체 추적 프로젝트 실행
cd SAM/SAM_UI
python run_pipeline.py

# PyTorch 통합 프레임워크 실행
cd 0_pytorch_integrated
python examples/run_vae_mnist.py
```

## 📋 프로젝트별 상세 정보

각 프로젝트의 상세한 사용법, 설정 방법, API 문서 등은 해당 프로젝트 디렉토리의 README.md 파일을 참조하세요.

## 🔧 공통 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (선택사항, 일부 프로젝트에서 필요)
- 각 프로젝트별 추가 의존성은 해당 프로젝트의 requirements.txt 참조

## 📝 라이선스

각 프로젝트는 개별적으로 라이선스가 적용됩니다. 자세한 내용은 각 프로젝트 디렉토리의 LICENSE 파일을 참조하세요.

## 🤝 기여

이 모노레포는 학습 및 연구 목적으로 개발되었습니다. 개선 사항이나 버그 리포트는 각 프로젝트별로 제출해 주세요.