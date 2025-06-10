import hydra
from omegaconf import DictConfig
from modules.pipeline import RAGPipeline
from dotenv import load_dotenv
import os
import sys
# .env 파일 로딩을 main 함수 실행 전에 수행
load_dotenv()

import logging

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 로그 디렉토리 생성
    log_dir = os.path.dirname(cfg.general.log.path)
    os.makedirs(log_dir, exist_ok=True)
    
    # 로깅 설정 수정
    logger = logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # stdout으로 출력 변경
            logging.FileHandler(cfg.general.log.path, encoding='utf-8')  # 파일 인코딩 지정
        ]
    )

    # query: RAG 관련 or 일반 질문
    # RAG 사용 유무
    query ="맛집 추천해줘." # "multi-head attention이란?"
    pipeline = RAGPipeline(cfg, logger)
    response = pipeline.run(query=query)
    print("🔹 AI 응답:", response)

if __name__ == "__main__":
    main()
