from typing import List, Dict
from langchain.schema import AIMessage, HumanMessage
import logging

class ConversationMemory:
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history: List[Dict] = []
        
    def add_interaction(self, query: str, response: str):
        """대화 이력 추가"""
        self.history.append({
            "human": HumanMessage(content=query),
            "ai": AIMessage(content=response)
        })
        
        # 최대 기록 개수 유지
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        logging.info(f"\n=== 대화 이력 업데이트 ===")
        logging.info(f"저장된 대화 수: {len(self.history)}")
    
    def get_chat_history(self) -> str:
        """대화 이력을 문자열로 반환"""
        history_str = ""
        for interaction in self.history:
            history_str += f"Human: {interaction['human'].content}\n"
            history_str += f"Assistant: {interaction['ai'].content}\n"
        return history_str
    
    def clear(self):
        """대화 이력 초기화"""
        self.history = []
        logging.info("대화 이력이 초기화되었습니다.") 