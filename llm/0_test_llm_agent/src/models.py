from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatPerplexity
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from src.config import Config

class ModelManager:
    """AI 모델 관리 클래스"""
    
    MODEL_CLASSES = {
        "ChatOpenAI": ChatOpenAI,
        "ChatPerplexity": ChatPerplexity,
        "ChatGoogleGenerativeAI": ChatGoogleGenerativeAI,
        "Ollama": Ollama
    }
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """설정된 모델들을 초기화합니다."""
        for config in Config.get_model_configs():
            try:
                model_class = self.MODEL_CLASSES[config["class"]]
                self.models[config["name"]] = model_class(**config["params"])
                print(f"✅ {config['name']} 모델 초기화 완료")
            except Exception as e:
                print(f"❌ {config['name']} 모델 초기화 실패: {e}")
    
    def get_model(self, name: str) -> Any:
        """지정된 이름의 모델을 반환합니다."""
        return self.models.get(name)
    
    def get_available_models(self) -> Dict[str, str]:
        """사용 가능한 모델 목록을 반환합니다."""
        return {
            name: config["agent_type"]
            for config in Config.MODEL_CONFIGS
            for name in [config["name"]]
            if name in self.models
        }
    
    def get_model_for_agent(self, agent_type: str) -> Dict[str, Any]:
        """특정 에이전트 타입에 사용 가능한 모델들을 반환합니다."""
        return {
            name: model
            for name, model in self.models.items()
            for config in Config.MODEL_CONFIGS
            if config["name"] == name and config["agent_type"] == agent_type
        } 