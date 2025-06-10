from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import yaml
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import time
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'  # 로깅 파일 인코딩 설정
)
logger = logging.getLogger(__name__)

# 윈도우 콘솔 로깅 인코딩 문제 해결
if sys.platform == 'win32':
    # 로거 핸들러에 인코딩 설정
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stdout)
    # sys.stdout 강제 인코딩 변경
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 환경 변수 로드
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def load_prompt(prompt_file='prompt.yaml'):
    """
    YAML 파일에서 프롬프트 템플릿을 로드합니다.
    """
    try:
        with open(prompt_file, 'r', encoding='utf-8-sig') as file:
            prompt_data = yaml.safe_load(file)
            return prompt_data.get('prompt', '')
    except Exception as e:
        logger.error(f"프롬프트 로딩 오류: {e}")
        return ""

def extract_and_log_search_results(query, content):
    """
    최종 응답에서 웹 검색 결과의 출처 정보를 추출하여 로깅합니다.
    """
    logger.info(f"\n===== 웹 검색 결과 분석 (키워드: {query}) =====")
    
    # 검색 결과 디렉토리 생성
    log_dir = Path("results/search_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로그 파일명 생성 (시간 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/search_{query.replace(' ', '_')}_{timestamp}.log"
    
    # 추출된 결과 요약
    results_summary = []
    
    # 검색 결과를 파일에 기록 - Windows에서 한글 표시를 위해 utf-8-sig 사용
    with open(log_filename, "w", encoding="utf-8-sig") as log_file:
        log_file.write(f"검색 키워드: {query}\n")
        log_file.write(f"검색 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_file.write("====== 생성된 응답에서 참조된 URL 및 출처 분석 ======\n\n")
        
        # URL 추출 시도
        urls = extract_urls(content)
        
        # 인용 패턴 추출 시도 (대괄호 인용, 각주 등)
        citations = extract_citations(content)
        
        if urls or citations:
            # URL 기록
            if urls:
                log_file.write("1. 참조된 URL:\n")
                for i, url in enumerate(urls, 1):
                    url_info = f"[URL {i}] {url}\n"
                    log_file.write(url_info)
                    logger.info(url_info)
                    results_summary.append(f"URL {i}: {url}")
                log_file.write("\n")
            
            # 인용 기록
            if citations:
                log_file.write("2. 감지된 인용 및 출처:\n")
                for i, citation in enumerate(citations, 1):
                    citation_info = f"[인용 {i}] {citation}\n"
                    log_file.write(citation_info)
                    logger.info(citation_info)
                    results_summary.append(f"인용 {i}: {citation}")
            
        else:
            no_url_msg = "응답에서 URL이나 출처 정보를 찾을 수 없습니다."
            log_file.write(no_url_msg + "\n")
            logger.info(no_url_msg)
            results_summary.append("출처 정보 없음")
        
        # 전체 응답 내용도 기록
        log_file.write("\n====== 전체 응답 내용 ======\n\n")
        log_file.write(content)
    
    logger.info(f"웹 검색 결과 분석 로그 파일 저장 완료: {log_filename}")
    logger.info("=============================================\n")
    
    # 요약 정보 반환 (프롬프트에 포함하기 위함)
    return "\n".join(results_summary)

def extract_urls(text):
    """
    텍스트에서 URL을 추출합니다.
    """
    import re
    # URL 패턴 정규식
    url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^\s]*)?')
    return list(set(url_pattern.findall(text)))  # 중복 제거하여 반환

def extract_citations(text):
    """
    텍스트에서 인용 정보(출처 등)를 추출합니다.
    다양한 인용 패턴(각주, 대괄호 참조 등)을 검색합니다.
    """
    import re
    citations = []
    
    # 1. 각주 스타일 패턴 ([1], [2] 등)
    footnote_pattern = re.compile(r'\[\d+\][^[]*?(?:https?://\S+|[^.]*?출처[^.]*)')
    footnotes = footnote_pattern.findall(text)
    
    # 2. 출처: 또는 참고: 뒤에 나오는 정보
    source_pattern = re.compile(r'(?:출처|참고|참조|source|reference)(?:[^\n.]*?)((?:https?://\S+|[^.]*?))')
    sources = source_pattern.findall(text)
    
    # 3. 언론사/기관명과 날짜가 함께 있는 패턴
    org_date_pattern = re.compile(r'(?:[가-힣a-zA-Z\s]+(?:신문|뉴스|일보|타임즈|저널|방송|기관|부|청|처|과))(?:[^,]*?),?\s?(\d{4}[년\-\.]\s?\d{1,2}[월\-\.]\s?\d{1,2}[일]?|\d{1,2}[월\-\.]\s?\d{1,2}[일]?,?\s?\d{4})')
    org_dates = org_date_pattern.findall(text)
    
    # 결과 합치기
    citations.extend(footnotes)
    citations.extend(sources)
    citations.extend(org_dates)
    
    # 중복 제거 및 공백 정리
    cleaned_citations = []
    for citation in set(citations):
        citation = citation.strip()
        if citation and len(citation) > 5:  # 의미 있는 길이의 인용만 포함
            cleaned_citations.append(citation)
    
    return cleaned_citations

def save_result(output_text, filename, prompt_text=None):
    """
    결과를 파일로 저장합니다. 프롬프트 텍스트가 제공된 경우 함께 저장합니다.
    """
    # results 디렉토리가 없으면 생성
    Path("results").mkdir(exist_ok=True)
    output_path = Path("results") / filename
    
    # HTML 파일 저장 (BOM 포함)
    with open(output_path, "w", encoding="utf-8-sig") as file:
        file.write(output_text)
    
    logger.info(f"HTML 파일 저장 완료: {output_path}")
    
    # 프롬프트도 저장
    if prompt_text:
        prompt_filename = f"{filename.rsplit('.', 1)[0]}_prompt.txt"
        prompt_path = Path("results") / prompt_filename
        with open(prompt_path, "w", encoding="utf-8-sig") as file:
            file.write(prompt_text)
        logger.info(f"프롬프트 파일 저장 완료: {prompt_path}")
    
    return output_path

def print_parameter_info():
    """
    사용 가능한 파라미터 정보를 출력합니다.
    """
    logger.info("\n사용 가능한 설정 파라미터:")
    logger.info("모델 설정:")
    logger.info("  model.name: 모델 이름 (웹 검색 지원 모델: gpt-4o-mini-search-preview, gpt-4o-search-preview)")
    logger.info("  model.temperature: 생성 온도 (0-2, 낮을수록 일관성 높음)")
    logger.info("  model.max_tokens: 최대 생성 토큰 수")
    logger.info("  model.top_p: 확률 분포의 상위 p%에서만 토큰 선택 (0-1)")
    
    logger.info("\n참고: 웹 검색을 위해 모델이름에 'search'가 포함된 모델을 사용하세요.")
    logger.info("자세한 정보는 https://platform.openai.com/docs/guides/web-search 를 참조하세요.")

def parse_prompt_file(file_content):
    """YAML 프롬프트 파일 내용에서 User 메시지에 사용할 부분을 추출합니다."""
    user_content_parts = []
    # User 메시지에 포함할 섹션 제목 목록
    user_sections = [
        "## [작업 절차]",
        "## [보고서 구조]",
        "## [HTML 템플릿]",
        "## [내용 작성 지침]",
        "## [사용 방법]",
        "[주제]:"
    ]
    
    current_section = None
    lines = file_content.splitlines()
    
    for line in lines:
        stripped_line = line.strip()
        is_section_header = False
        for section in user_sections:
            if stripped_line.startswith(section):
                current_section = section
                is_section_header = True
                break
        
        if current_section:
            # 포함할 섹션이 시작되면 해당 줄부터 추가
            # 단, 역할, 핵심 규칙 등 System 메시지로 간 내용은 제외
            if not stripped_line.startswith(("## [역할]", "## [핵심 규칙]")):
                 user_content_parts.append(line)
                 
    # 마지막 "[주제]:" 부분은 남겨두고 반환 (키워드 삽입용)
    return "\n".join(user_content_parts)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    메인 함수 - 보도자료 생성 프로세스를 실행합니다.
    """
    # 설정 출력
    logger.info(f"설정:\n{OmegaConf.to_yaml(cfg)}")
    
    # API 클라이언트 초기화
    client = OpenAI()
    
    # 프롬프트 파일 로드 (전체 내용)
    full_prompt_content = load_prompt(cfg.prompt.file)
    if not full_prompt_content:
        logger.error("프롬프트 파일을 로드할 수 없습니다.")
        return 1

    # User 메시지에 사용할 부분 추출 (구조, 템플릿, 절차 등)
    user_prompt_template = parse_prompt_file(full_prompt_content)
    
    # 키워드 설정
    keyword = cfg.keyword
    logger.info(f"검색 키워드: {keyword}")
    
    # 모델 설정
    search_model = cfg.model.name
    logger.info(f"사용 모델: {search_model}")
    
    # 최종 프롬프트 구성
    system_message = cfg.prompt.system_message # config.yaml에서 가져온 시스템 메시지
    user_message = user_prompt_template.replace("[주제]:", f"[주제]: {keyword}") # 추출된 템플릿에 키워드 삽입
    
    # 저장용 프롬프트 텍스트 생성 (실제 전달되는 내용 확인용)
    full_prompt_log = f"====== 시스템 메시지 ======\n{system_message}\n\n====== 사용자 메시지 ======\n{user_message}"
    logger.info("프롬프트 준비 완료")
    # logger.debug(full_prompt_log) # 디버깅 시 프롬프트 전체 내용 확인
    
    # OpenAI API 호출
    logger.info("OpenAI API 호출 중...")
    try:
        # 검색 가능 모델 확인
        if 'search' in search_model:
            logger.info("웹 검색 지원 모델을 사용 중입니다.")
            
            # 기본 API 파라미터 설정 - 최소한의 필수 파라미터만 포함
            api_params = {
                "model": search_model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": cfg.model.max_tokens
            }
            
            # 모든 옵션 로깅만 수행
            if hasattr(cfg.model, 'temperature'):
                logger.info(f"참고: 온도 설정({cfg.model.temperature})은 이 모델에서 지원되지 않을 수 있습니다")
                
            # 웹 검색 옵션 설정
            if hasattr(cfg.web_search, 'search_context_size'):
                logger.info(f"검색 컨텍스트 크기: {cfg.web_search.search_context_size}")
                
            # 날짜 제한 설정 안내
            if hasattr(cfg.web_search, 'recency_days') and cfg.web_search.recency_days:
                logger.info(f"참고: 최근 {cfg.web_search.recency_days}일 이내 검색을 원하시면 키워드에 날짜 범위를 포함하세요")
            
            # 지역 정보 안내
            logger.info("지역 정보는 키워드에 직접 포함하세요 (예: '대전시 부동산 정책')")
            
            # 안전한 로깅을 위해 메시지 내용은 생략
            safe_params = api_params.copy()
            if 'messages' in safe_params:
                safe_params['messages'] = "[메시지 내용 생략]"
            logger.info(f"API 호출에 사용될 파라미터: {safe_params}")
            
            # API 호출 실행 (최소 파라미터만 사용)
            logger.info(f"OpenAI API 호출 중... 모델: {search_model}")
            response = client.chat.completions.create(**api_params)
        else:
            # 웹 검색을 지원하지 않는 모델은 기본 완성 사용 (모든 생성 파라미터 전달)
            logger.warning(f"모델 '{search_model}'은 웹 검색 기능을 지원하지 않습니다. 생성 파라미터를 모두 전달합니다.")
            response = client.chat.completions.create(
                model=search_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=cfg.model.temperature,
                max_tokens=cfg.model.max_tokens,
                top_p=cfg.model.top_p,
                frequency_penalty=cfg.model.frequency_penalty,
                presence_penalty=cfg.model.presence_penalty,
                seed=cfg.model.seed
            )
        
        # 응답 처리
        logger.info("응답 처리 중...")
        if response.choices and len(response.choices) > 0:
            output_text = response.choices[0].message.content
            
            if output_text:  # None이 아닌 경우에만 작업 수행
                logger.info("\n--- 결과 미리보기 ---")
                preview_length = min(cfg.output.preview_length, len(output_text))
                logger.info(f"{output_text[:preview_length]}...")
                
                # 웹 검색 결과 로깅 및 분석
                search_results_info = extract_and_log_search_results(keyword, output_text)
                full_prompt_log += f"\n\n====== 검색 결과 정보 ======\n{search_results_info}"
                
                # 파일 이름 설정
                if cfg.output.file:
                    filename = cfg.output.file
                else:
                    safe_keyword = keyword.replace(' ', '_').replace('/', '_')
                    filename = f"press_release_{safe_keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                
                # 결과 및 프롬프트 저장
                output_path = save_result(output_text, filename, full_prompt_log)
                
                # 설정 정보 저장 (디버깅 및 재현성 위해)
                config_filename = f"{filename.rsplit('.', 1)[0]}_config.yaml"
                with open(Path("results") / config_filename, "w", encoding="utf-8-sig") as f:
                    f.write(OmegaConf.to_yaml(cfg))
                
                logger.info(f"설정 파일 저장 완료: results/{config_filename}")
            else:
                logger.error("유효한 응답을 받지 못했습니다.")
        else:
            logger.error("응답을 받지 못했습니다.")
    except Exception as e:
        logger.error(f"API 호출 중 오류 발생: {str(e)}")
        logger.error("사용 중인 모델이 웹 검색 기능을 지원하지 않거나 API 설정에 문제가 있습니다.")
        logger.error("지원되는 모델: gpt-4o-search-preview, gpt-4o-mini-search-preview 등")
    
    # 파라미터 정보 출력
    print_parameter_info()
    
    return 0

if __name__ == "__main__":
    main()

# 위치 기반 검색 예시 (주석 처리)
"""
location_response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "한글로 대답해주세요."},
        {"role": "user", "content": "대전 신성동 근처의 미셀린 가이드 선정된 맛집?"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information based on location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "object",
                        "properties": {
                            "country": {"type": "string", "description": "ISO 2자리 국가 코드"},
                            "city": {"type": "string"},
                            "region": {"type": "string"}
                        }
                    }
                },
                "required": []
            }
        }
    }],
    tool_choice="auto"
)

if location_response.choices and len(location_response.choices) > 0:
    location_output = location_response.choices[0].message.content
    print("\n위치 기반 검색 결과:")
    print(location_output)
"""

# 참고: 
# country: ISO 2자리 국가 코드 (예: KR)
# city, region: 자유 텍스트로 입력 가능 (예: "Daejeon", "Yuseong-gu").
# 특정 날짜: 질의문(input)에 날짜를 명시하면 해당 날짜에 초점을 맞춘 검색이 가능합니다. 예: "2025년 3월 10일 대전 신성동 날씨"
# 특정 웹 주소(도메인 한정): 질의문에 "사이트:example.com"과 같이 명시하면, 해당 사이트 내에서만 검색하도록 유도할 수 있습니다.
# 예: "site:daum.net 대전 신성동 맛집"

# PDF 등 특정 파일 유형: 질의문에 "PDF로 제공되는 문서" 등으로 명시하면, 모델이 해당 조건에 맞는 결과를 찾으려 시도


# {"type": "web_search_preview"}
# file_search	업로드 파일 내 검색	{"type": "file_search", "file_id": "abc123"}
# code_interpreter	코드 실행, 데이터 분석	{"type": "code_interpreter"}
# computer_use	컴퓨터 작업 자동화	{"type": "computer_use"}
# deep_research	심층 다중소스 리서치 및 보고서 생성	{"type": "deep_research"}
# deep_research	심층 다중소스 리서치 및 보고서 생성	{"type": "deep_research"}