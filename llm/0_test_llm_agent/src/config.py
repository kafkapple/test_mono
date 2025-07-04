from typing import Dict, List
import os
from dotenv import load_dotenv

class Config:
    """시스템 설정 관리 클래스"""
    
    REQUIRED_KEYS = {
        "PPLX_API_KEY": "Perplexity API 키",
        "OPENAI_API_KEY": "OpenAI API 키",
        "GOOGLE_API_KEY": "Google API 키"
    }
    
    MODEL_CONFIGS = [
        {
            "name": "openai",
            "class": "ChatOpenAI",
            "params": {
                "model": "gpt-4o",
                "temperature": 0.7
            },
            "env_key": "OPENAI_API_KEY",
            "agent_type": "tool_calling"
        },
        {
            "name": "perplexity",
            "class": "ChatPerplexity",
            "params": {
                "model": "llama-3.1-sonar-large-128k-online",
                "temperature": 0.7,
                "max_tokens": 2048,
                "streaming": False,
                "request_timeout": 120
            },
            "env_key": "PPLX_API_KEY",
            "agent_type": "chat"
        },
        {
            "name": "gemini",
            "class": "ChatGoogleGenerativeAI",
            "params": {
                "model": "gemini-2.0-flash-exp",
                "temperature": 0.7
            },
            "env_key": "GOOGLE_API_KEY",
            "agent_type": "react"
        },
        {
            "name": "ollama",
            "class": "Ollama",
            "params": {
                "model": "llama3.2"
            },
            "env_key": None,
            "agent_type": "react"
        }
    ]
    
    @classmethod
    def load_environment(cls) -> Dict[str, str]:
        """환경 변수를 로드하고 검증합니다."""
        load_dotenv()
        
        missing_keys = []
        for key, description in cls.REQUIRED_KEYS.items():
            if not os.getenv(key):
                missing_keys.append(f"{key} ({description})")
        
        if missing_keys:
            print("⚠️  누락된 환경변수:")
            for key in missing_keys:
                print(f"   - {key}")
            print("\n.env 파일을 확인하거나 환경변수를 설정해주세요.")
            return {}
        
        print("✅ 모든 API 키가 로드되었습니다.")
        return {key: os.getenv(key) for key in cls.REQUIRED_KEYS}
    
    @classmethod
    def get_model_configs(cls) -> List[Dict]:
        """사용 가능한 모델 설정을 반환합니다."""
        env_vars = {key: os.getenv(key) for key in cls.REQUIRED_KEYS}
        return [
            config for config in cls.MODEL_CONFIGS
            if not config["env_key"] or env_vars.get(config["env_key"])
        ] 