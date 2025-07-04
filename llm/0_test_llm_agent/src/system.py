from typing import Dict, Any, Optional
from src.config import Config
from src.models import ModelManager
from src.tools import ToolManager
from src.agents import AgentManager

class MultiModelAISystem:
    """다중 AI 모델 협업 시스템"""
    
    def __init__(self):
        """시스템을 초기화합니다."""
        # 환경 변수 로드
        self.env_vars = Config.load_environment()
        
        # 컴포넌트 초기화
        self.model_manager = ModelManager()
        self.tool_manager = ToolManager(self.model_manager)
        self.agent_manager = AgentManager(self.model_manager, self.tool_manager)
    
    def process_request(self, query: str, preferred_agent: str = "openai") -> str:
        """사용자 요청을 처리합니다."""
        return self.agent_manager.process_request(query, preferred_agent)
    
    def get_status(self) -> Dict[str, Any]:
        """시스템 상태를 반환합니다."""
        return {
            "models": self.model_manager.get_available_models(),
            "agents": self.agent_manager.get_available_agents(),
            "tools": self.tool_manager.get_tool_names(),
            "env_loaded": bool(self.env_vars)
        }
    
    def get_available_agents(self) -> Dict[str, str]:
        """사용 가능한 에이전트 목록을 반환합니다."""
        return self.agent_manager.get_available_agents()
    
    def get_available_models(self) -> Dict[str, str]:
        """사용 가능한 모델 목록을 반환합니다."""
        return self.model_manager.get_available_models()
    
    def get_available_tools(self) -> list[str]:
        """사용 가능한 도구 목록을 반환합니다."""
        return self.tool_manager.get_tool_names()
    
    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """특정 에이전트의 상세 정보를 반환합니다."""
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return None
        
        return {
            "name": agent_name,
            "type": "tool_calling" if hasattr(agent, "agent_type") else "react",
            "tools": self.tool_manager.get_tool_names(),
            "model": next(
                (name for name, model in self.model_manager.models.items()
                 if model == agent.llm),
                None
            )
        } 