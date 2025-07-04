from openai import OpenAI
from dotenv import load_dotenv
import os
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def load_prompt(prompt_file='prompt.yaml'):
    """프롬프트 템플릿을 로드합니다."""
    try:
        with open(prompt_file, 'r', encoding='utf-8-sig') as file:
            prompt_data = yaml.safe_load(file)
            return prompt_data.get('prompt', '')
    except Exception as e:
        logger.error(f"프롬프트 로딩 오류: {e}")
        return ""

def extract_urls(text):
    """텍스트에서 URL을 추출합니다."""
    import re
    url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^\s]*)?')
    return list(set(url_pattern.findall(text)))

def save_result(output_text, filename, prompt_text=None):
    """결과를 파일로 저장합니다."""
    Path("results").mkdir(exist_ok=True)
    output_path = Path("results") / filename
    
    with open(output_path, "w", encoding="utf-8-sig") as file:
        file.write(output_text)
    
    logger.info(f"HTML 파일 저장 완료: {output_path}")
    
    if prompt_text:
        prompt_filename = f"{filename.rsplit('.', 1)[0]}_prompt.txt"
        prompt_path = Path("results") / prompt_filename
        with open(prompt_path, "w", encoding="utf-8-sig") as file:
            file.write(prompt_text)
        logger.info(f"프롬프트 파일 저장 완료: {prompt_path}")
    
    return output_path

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """웹 검색을 통한 보도자료 생성"""
    logger.info(f"설정:\n{OmegaConf.to_yaml(cfg)}")
    
    # API 클라이언트 초기화
    client = OpenAI()
    
    # 프롬프트 로드
    base_prompt = load_prompt(cfg.prompt.file)
    if not base_prompt:
        base_prompt = "입력된 키워드에 대한 보도자료를 HTML 형식으로 작성해주세요."
    
    # 키워드 설정
    keyword = cfg.keyword
    logger.info(f"검색 키워드: {keyword}")
    
    # 모델 설정
    model_name = cfg.model.name
    logger.info(f"사용 모델: {model_name}")
    
    # 프롬프트 구성
    system_message = cfg.prompt.system_message
    user_message = base_prompt.replace("[키워드]:", "[키워드]: "+keyword)
    
    # 저장용 프롬프트 텍스트 생성
    full_prompt = f"시스템: {system_message}\n\n사용자: {user_message}"
    
    try:
        # 웹 검색 지원 모델 확인
        if 'search' in model_name:
            # 웹 검색 옵션 설정
            web_search_options = {}
            
            # 검색 컨텍스트 크기 설정
            if hasattr(cfg.web_search, 'search_context_size'):
                web_search_options["search_context_size"] = cfg.web_search.search_context_size
            
            # 사용자 위치 설정 (간단한 형태)
            if hasattr(cfg.web_search, 'user_location'):
                if cfg.web_search.user_location.lower() == 'kr':
                    web_search_options["user_location"] = {
                        "type": "default",
                        "country": "KR"
                    }
                else:
                    country_code = "KR"  # 기본값
                    if ":" in cfg.web_search.user_location:
                        parts = cfg.web_search.user_location.split(":", 1)
                        country_code = parts[0].strip()
                        city = parts[1].strip() if len(parts) > 1 else None
                        
                        location_obj = {
                            "type": "default",
                            "country": country_code
                        }
                        if city:
                            location_obj["city"] = city
                            
                        web_search_options["user_location"] = location_obj
            
            # API 호출 파라미터
            api_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": cfg.model.max_tokens
            }
            
            # 웹 검색 옵션이 있으면 추가
            if web_search_options:
                api_params["web_search_options"] = web_search_options
                
            logger.info("OpenAI API 호출 중...")
            response = client.chat.completions.create(**api_params)
        else:
            # 일반 모델 호출
            logger.info("일반 모델로 API 호출 중...")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=cfg.model.temperature,
                max_tokens=cfg.model.max_tokens,
                top_p=cfg.model.top_p
            )
        
        # 응답 처리
        if response.choices and len(response.choices) > 0:
            output_text = response.choices[0].message.content
            
            if output_text:
                # 미리보기 출력
                preview_length = min(cfg.output.preview_length, len(output_text))
                logger.info(f"\n결과 미리보기:\n{output_text[:preview_length]}...")
                
                # URL 추출 및 로깅
                urls = extract_urls(output_text)
                if urls:
                    logger.info("\n참조된 URL:")
                    for i, url in enumerate(urls, 1):
                        logger.info(f"[URL {i}] {url}")
                
                # 파일 저장
                if cfg.output.file:
                    filename = cfg.output.file
                else:
                    safe_keyword = keyword.replace(' ', '_').replace('/', '_')
                    filename = f"press_release_{safe_keyword}.html"
                
                save_result(output_text, filename, full_prompt)
            else:
                logger.error("유효한 응답을 받지 못했습니다.")
        else:
            logger.error("응답을 받지 못했습니다.")
    
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
    
    return 0

if __name__ == "__main__":
    main() 