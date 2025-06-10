from typing import Dict, Any, Optional, List
from langchain.agents import (
    AgentExecutor, create_tool_calling_agent,
    initialize_agent, AgentType
)
from langchain_core.prompts import ChatPromptTemplate
from src.models import ModelManager
from src.tools import ToolManager

class AgentManager:
    """에이전트 관리 클래스"""
    
    def __init__(self, model_manager: ModelManager, tool_manager: ToolManager):
        self.model_manager = model_manager
        self.tool_manager = tool_manager
        self.agents: Dict[str, Any] = {}
        self._initialize_agents()
    
    def _get_tools_for_agent_type(self, agent_type: str) -> List[Any]:
        """에이전트 타입에 맞는 도구 목록을 반환합니다."""
        all_tools = self.tool_manager.get_tools()
        if agent_type == "react":
            # ReAct 에이전트는 단일 입력 도구만 지원
            return [tool for tool in all_tools 
                   if len(tool.args_schema.schema()["properties"]) <= 1]
        return all_tools
    
    def _initialize_agents(self):
        """에이전트들을 초기화합니다."""
        # Tool-calling 지원 모델용 프롬프트
        tool_calling_prompt = ChatPromptTemplate.from_messages([
            ("system", """🤖 당신은 다중 AI 모델 협업 시스템의 스마트 코디네이터입니다.

🛠️ 사용 가능한 도구들:
{tools}

🎯 사용자의 요청을 정확히 분석하고, 가장 적절한 도구를 선택하여 최고 품질의 답변을 제공하세요.
필요시 여러 도구를 조합하여 사용할 수 있습니다."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # ReAct 에이전트용 프롬프트
        react_prompt = ChatPromptTemplate.from_messages([
            ("system", """🤖 당신은 다중 AI 모델 협업 시스템의 ReAct 에이전트입니다.

🛠️ 사용 가능한 도구들:
{tools}

🎯 다음 형식으로 응답하세요:
Question: 사용자의 질문
Thought: 생각하는 과정
Action: 사용할 도구
Action Input: 도구에 전달할 입력
Observation: 도구의 결과
... (Thought/Action/Action Input/Observation 반복)
Thought: 최종 생각
Answer: 최종 답변

중요: 각 단계는 새로운 줄에서 시작하고, 정확히 위의 형식을 따라주세요."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        # Chat 모드용 프롬프트
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """🤖 당신은 다중 AI 모델 협업 시스템의 채팅 어시스턴트입니다.

🎯 사용자의 질문에 대해 정확하고 상세한 답변을 제공하세요.
- 최신 정보와 데이터를 포함
- 신뢰할 수 있는 출처 언급
- 명확하고 구조화된 형식으로 응답"""),
            ("human", "{input}")
        ])
        
        # Tool-calling 에이전트 생성
        tool_calling_models = self.model_manager.get_model_for_agent("tool_calling")
        for name, model in tool_calling_models.items():
            try:
                tools = self._get_tools_for_agent_type("tool_calling")
                agent = create_tool_calling_agent(
                    model,
                    tools,
                    tool_calling_prompt.partial(
                        tools="\n".join(f"{i+1}. {tool.name}: {tool.description}"
                                      for i, tool in enumerate(tools))
                    )
                )
                self.agents[name] = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=5
                )
                print(f"✅ {name} 에이전트 생성 완료")
            except Exception as e:
                print(f"❌ {name} 에이전트 생성 실패: {e}")
        
        # Chat 모드 에이전트 생성 (Perplexity)
        chat_models = self.model_manager.get_model_for_agent("chat")
        for name, model in chat_models.items():
            try:
                # Perplexity 모델은 직접 모델 인스턴스를 사용
                self.agents[name] = model
                print(f"✅ {name} 채팅 에이전트 생성 완료")
            except Exception as e:
                print(f"❌ {name} 채팅 에이전트 생성 실패: {e}")
        
        # ReAct 에이전트 생성 (Gemini)
        react_models = self.model_manager.get_model_for_agent("react")
        for name, model in react_models.items():
            try:
                tools = self._get_tools_for_agent_type("react")
                agent = initialize_agent(
                    tools=tools,
                    llm=model,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=5,
                    prompt=react_prompt.partial(
                        tools="\n".join(f"{i+1}. {tool.name}: {tool.description}"
                                      for i, tool in enumerate(tools))
                    )
                )
                self.agents[name] = agent
                print(f"✅ {name} ReAct 에이전트 생성 완료")
            except Exception as e:
                print(f"❌ {name} ReAct 에이전트 생성 실패: {e}")
    
    def get_agent(self, name: str) -> Optional[Any]:
        """지정된 이름의 에이전트를 반환합니다."""
        return self.agents.get(name)
    
    def get_available_agents(self) -> Dict[str, str]:
        """사용 가능한 에이전트 목록을 반환합니다."""
        return {
            name: "tool_calling" if hasattr(agent, "agent_type") else "react"
            for name, agent in self.agents.items()
        }
    
    def process_request(self, query: str, preferred_agent: str = "openai") -> str:
        """사용자 요청을 처리합니다."""
        try:
            if not self.agents:
                return "❌ 사용 가능한 에이전트가 없습니다. 환경 설정을 확인해주세요."
            
            # 에이전트가 없으면 첫 번째 사용 가능한 에이전트 사용
            if preferred_agent not in self.agents:
                available_agents = list(self.agents.keys())
                if not available_agents:
                    return "❌ 사용 가능한 에이전트가 없습니다."
                preferred_agent = available_agents[0]
                print(f"⚠️  요청한 에이전트를 찾을 수 없어 {preferred_agent}를 사용합니다.")
            
            agent = self.agents[preferred_agent]
            print(f"🚀 {preferred_agent} 에이전트로 처리 중...")
            
            # Chat 모드 에이전트 처리 (Perplexity)
            if preferred_agent == "perplexity":
                try:
                    # 직접 모델에 메시지 전달
                    messages = [
                        {"role": "system", "content": "You are a helpful AI assistant. Provide detailed and accurate responses with recent examples and reliable sources."},
                        {"role": "user", "content": query}
                    ]
                    response = agent.invoke(messages)
                    return response.content if hasattr(response, 'content') else str(response)
                except Exception as e:
                    print(f"⚠️ Chat 모드 처리 중 오류: {str(e)}")
                    # 대체 방법: 직접 모델 호출
                    model = self.model_manager.get_model("perplexity")
                    if model:
                        response = model.invoke(query)
                        return response.content if hasattr(response, 'content') else str(response)
                    raise e
            
            # Tool-calling 및 ReAct 에이전트 처리
            if hasattr(agent, 'invoke'):
                result = agent.invoke({"input": query})
                return result.get('output', str(result))
            else:
                result = agent.run(query)
                return result
            
        except Exception as e:
            return f"❌ 처리 중 오류 발생: {str(e)}" 