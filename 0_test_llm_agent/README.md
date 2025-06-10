# 다중 AI 모델 협업 시스템

이 시스템은 여러 AI 모델을 통합하여 각 모델의 장점을 활용할 수 있는 협업 시스템입니다.

## 🚀 주요 기능

- 다중 AI 모델 통합 (OpenAI, Perplexity, Google Gemini, Ollama)
- ReACT 및 Tool-calling 에이전트 지원
- 다양한 도구 통합 (계산기, 검색, 텍스트 분석 등)
- 모델 간 협업 및 작업 위임

## 🔧 시스템 요구사항

- Python 3.8 이상
- 필요한 API 키:
  - OpenAI API 키
  - Perplexity API 키
  - Google API 키 (Gemini 사용 시)

## 📦 설치 방법

1. 저장소 클론:
```bash
git clone [repository-url]
cd [repository-name]
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정:
`.env` 파일을 생성하고 다음 API 키를 설정:
```
OPENAI_API_KEY=your_openai_api_key
PPLX_API_KEY=your_perplexity_api_key
GOOGLE_API_KEY=your_google_api_key
```

## 🤖 지원 모델 및 에이전트 타입

### OpenAI (GPT-4)
- 에이전트 타입: Tool-calling
- 주요 특징: 복잡한 추론과 도구 사용에 최적화
- 사용 예시: 복잡한 계산, 텍스트 분석, 다른 모델과의 협업

### Perplexity
- 에이전트 타입: ReACT
- 주요 특징: 실시간 웹 검색 및 최신 정보 제공
- 사용 예시: 최신 트렌드 검색, 실시간 정보 조회

### Google Gemini
- 에이전트 타입: Tool-calling
- 주요 특징: 창의적인 답변과 다국어 지원
- 사용 예시: 창의적 글쓰기, 다국어 텍스트 분석

### Ollama
- 에이전트 타입: ReACT
- 주요 특징: 로컬 실행, 오프라인 지원
- 사용 예시: 로컬 데이터 처리, 오프라인 작업

## 🛠️ 사용 가능한 도구

### 1. 안전한 계산기 (safe_calculator)
- 기능: 기본 산술 연산 수행
- 지원 연산: 덧셈, 뺄셈, 곱셈, 나눗셈, 거듭제곱, 모듈로
- 사용 예시: "127 * 89 + 456을 계산해주세요"

### 2. Perplexity 검색 (perplexity_search)
- 기능: 실시간 웹 검색 및 정보 조회
- 특징: 최신 정보 제공, 신뢰할 수 있는 출처 포함
- 사용 예시: "2024년 AI 기술 트렌드에 대해 검색해주세요"

### 3. 텍스트 분석기 (text_analyzer)
- 기능: 텍스트 상세 분석
- 분석 항목: 단어 수, 문자 수, 문장 수, 언어 감지, 읽기 시간
- 사용 예시: "다음 텍스트를 분석해주세요: [텍스트]"

### 4. 모델 선택기 (model_selector)
- 기능: 특정 AI 모델에게 작업 할당
- 지원 모델: openai, gemini, perplexity, ollama
- 사용 예시: "ChatGPT의 최신 업데이트에 대해 perplexity 모델에게 물어보세요"

## 🔄 ReACT 모듈

ReACT (Reasoning and Acting) 모듈은 다음과 같은 특징을 가집니다:

### 작동 방식
1. Question: 사용자의 질문 이해
2. Thought: 현재 상황 분석 및 다음 단계 계획
3. Action: 적절한 도구 선택
4. Action Input: 도구에 전달할 입력 준비
5. Observation: 도구 실행 결과 관찰
6. Thought: 결과 분석 및 다음 단계 결정
7. Answer: 최종 답변 제공

### 지원 모델
- Perplexity
- Ollama

### 제한사항
- ReACT 에이전트는 단일 입력 도구만 지원
- 최대 5회의 반복 실행 제한
- 도구 사용 시 파싱 오류 자동 처리

## ⚠️ 알려진 문제

1. Perplexity 모델 오류
   - 현재 "custom stop words are not implemented for completions" 오류 발생
   - 해결 방법: Perplexity API 설정에서 stop words 관련 파라미터 제거 필요

## 📝 사용 예시

```python
from system import MultiModelAISystem

# 시스템 초기화
ai_system = MultiModelAISystem()

# 상태 확인
status = ai_system.get_status()
print(f"로드된 모델: {status['models']}")
print(f"활성 에이전트: {status['agents']}")

# 요청 처리
result = ai_system.process_request(
    "2024년 AI 기술 트렌드에 대해 검색해주세요",
    preferred_agent="perplexity"
)
print(result)
```

## 🔍 문제 해결

1. 모델 초기화 실패
   - API 키가 올바르게 설정되었는지 확인
   - 인터넷 연결 상태 확인
   - 모델별 API 할당량 확인

2. 도구 실행 오류
   - 입력 형식 확인
   - 도구별 제한사항 확인
   - 에이전트 타입과 도구 호환성 확인

## 📄 라이선스

MIT License 