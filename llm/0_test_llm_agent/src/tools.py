from typing import Dict, Any, List
import ast
import operator
from langchain.tools import tool
from src.models import ModelManager

class ToolManager:
    """도구 관리 클래스"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.tools: List[Any] = []
        self._initialize_tools()
    
    def _initialize_tools(self):
        """도구들을 초기화합니다."""
        self.tools = [
            self._create_safe_calculator(),
            self._create_perplexity_search(),
            self._create_text_analyzer(),
            self._create_model_selector()
        ]
    
    @staticmethod
    def _create_safe_calculator():
        @tool
        def safe_calculator(expression: str) -> str:
            """안전한 수학 계산기 - 기본 산술 연산만 지원합니다.
            
            Args:
                expression: 계산할 수식 (예: "2 + 3 * 4")
            
            Returns:
                str: 계산 결과 또는 오류 메시지
            """
            try:
                allowed_operators = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.Pow: operator.pow,
                    ast.USub: operator.neg,
                    ast.UAdd: operator.pos,
                    ast.Mod: operator.mod,
                }
                
                def eval_node(node):
                    if isinstance(node, ast.Constant):
                        return node.value
                    elif isinstance(node, ast.BinOp):
                        left = eval_node(node.left)
                        right = eval_node(node.right)
                        if isinstance(node.op, ast.Div) and right == 0:
                            raise ValueError("0으로 나눌 수 없습니다")
                        return allowed_operators[type(node.op)](left, right)
                    elif isinstance(node, ast.UnaryOp):
                        operand = eval_node(node.operand)
                        return allowed_operators[type(node.op)](operand)
                    else:
                        raise ValueError(f"지원되지 않는 연산: {type(node).__name__}")
                
                tree = ast.parse(expression, mode='eval')
                result = eval_node(tree.body)
                return f"💡 계산 결과: {result}"
                
            except Exception as e:
                return f"❌ 계산 오류: {str(e)}"
        
        return safe_calculator
    
    def _create_perplexity_search(self):
        @tool
        def perplexity_search(query: str) -> str:
            """Perplexity를 사용한 실시간 웹 검색 및 정보 조회
            
            Args:
                query: 검색할 질문이나 키워드
            
            Returns:
                str: 검색 결과 또는 오류 메시지
            """
            try:
                model = self.model_manager.get_model("perplexity")
                if not model:
                    return "❌ Perplexity 모델을 사용할 수 없습니다. API 키를 확인해주세요."
                
                search_prompt = f"""You are a helpful AI assistant. Please provide a detailed and accurate response to the following query:

Query: {query}

Please follow these guidelines:
1. Provide factual and up-to-date information
2. Include relevant statistics or data when available
3. Cite reliable sources
4. Keep the response clear and well-structured
5. If the query is in Korean, respond in Korean; if in English, respond in English

Response:"""
                
                response = model.invoke(search_prompt)
                return f"🔍 검색 결과:\n{response.content}"
                
            except Exception as e:
                return f"❌ 검색 오류: {str(e)}"
        
        return perplexity_search
    
    @staticmethod
    def _create_text_analyzer():
        @tool
        def text_analyzer(text: str) -> str:
            """텍스트 분석 도구 - 단어 수, 문자 수, 감정 분석 등
            
            Args:
                text: 분석할 텍스트
            
            Returns:
                str: 분석 결과
            """
            try:
                words = text.split()
                word_count = len(words)
                char_count = len(text)
                char_count_no_spaces = len(text.replace(' ', ''))
                sentence_count = text.count('.') + text.count('!') + text.count('?')
                
                korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
                english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
                
                if korean_chars > english_chars:
                    primary_language = "한국어"
                elif english_chars > korean_chars:
                    primary_language = "영어"
                else:
                    primary_language = "혼합/기타"
                
                avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
                
                result = f"""📊 텍스트 분석 결과:
                
📝 기본 통계:
   • 단어 수: {word_count:,}개
   • 전체 문자 수: {char_count:,}자
   • 공백 제외 문자 수: {char_count_no_spaces:,}자
   • 문장 수: {sentence_count}개
   • 평균 단어 길이: {avg_word_length:.1f}자
   
🌍 언어 정보:
   • 주요 언어: {primary_language}
   • 한글 문자: {korean_chars}자
   • 영문 문자: {english_chars}자
   
📈 읽기 정보:
   • 예상 읽기 시간: {word_count / 200:.1f}분 (분당 200단어 기준)"""
                
                return result
                
            except Exception as e:
                return f"❌ 분석 오류: {str(e)}"
        
        return text_analyzer
    
    def _create_model_selector(self):
        @tool
        def model_selector(task: str, preferred_model: str = "openai") -> str:
            """특정 AI 모델에게 작업을 할당합니다.
            
            Args:
                task: 수행할 작업이나 질문
                preferred_model: 사용할 모델 (openai, gemini, perplexity, ollama)
            
            Returns:
                str: 모델의 응답 결과
            """
            try:
                available_models = list(self.model_manager.models.keys())
                
                if preferred_model not in available_models:
                    return f"❌ 모델 '{preferred_model}'을 찾을 수 없습니다.\n사용 가능한 모델: {', '.join(available_models)}"
                
                model = self.model_manager.get_model(preferred_model)
                
                model_prompts = {
                    "openai": f"OpenAI GPT 모델로서 다음 작업을 수행해주세요:\n\n{task}",
                    "gemini": f"Google Gemini 모델로서 창의적이고 상세한 답변을 제공해주세요:\n\n{task}",
                    "perplexity": f"최신 정보와 함께 정확한 답변을 제공해주세요:\n\n{task}",
                    "ollama": f"로컬 AI 모델로서 다음 작업을 수행해주세요:\n\n{task}"
                }
                
                prompt = model_prompts.get(preferred_model, task)
                response = model.invoke(prompt)
                
                return f"🤖 [{preferred_model.upper()} 응답]:\n{response.content}"
                
            except Exception as e:
                return f"❌ 모델 호출 오류: {str(e)}"
        
        return model_selector
    
    def get_tools(self) -> List[Any]:
        """사용 가능한 도구 목록을 반환합니다."""
        return self.tools
    
    def get_tool_names(self) -> List[str]:
        """도구 이름 목록을 반환합니다."""
        return [tool.name for tool in self.tools] 