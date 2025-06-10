from src.system import MultiModelAISystem

def main():
    print("🚀 다중 AI 모델 협업 시스템 시작")
    print("=" * 50)
    
    # 시스템 초기화
    ai_system = MultiModelAISystem()
    
    # 상태 확인
    status = ai_system.get_status()
    print(f"\n📊 시스템 상태:")
    print(f"   • 로드된 모델: {', '.join(status['models'].keys())}")
    print(f"   • 활성 에이전트: {', '.join(status['agents'].keys())}")
    print(f"   • 사용 가능한 도구: {len(status['tools'])}개")
    
    # 테스트 케이스들
    test_cases = [
        ("127 * 89 + 456을 계산해주세요", "openai"),
        ("What are the latest AI technology trends in 2024? Please provide detailed information with recent examples.", "perplexity"),
        ("다음 텍스트를 분석해주세요: 인공지능 기술의 발전은 우리 삶을 크게 변화시키고 있습니다. 특히 자연어 처리 분야에서의 혁신은 놀라울 정도입니다.", "gemini"),
        ("Ask Perplexity about the latest updates in ChatGPT and summarize the key points", "openai")
    ]
    
    for i, (query, agent) in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"🧪 테스트 {i}: {agent.upper()} 에이전트")
        print(f"❓ 질문: {query}")
        print(f"{'='*80}")
        
        result = ai_system.process_request(query, agent)
        print(f"💬 답변: {result}")

if __name__ == "__main__":
    main()
