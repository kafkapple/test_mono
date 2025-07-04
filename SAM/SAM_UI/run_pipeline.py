import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 메인 파이프라인 임포트
from src.sam_mot_pipeline import SAM_MOT_Pipeline

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config")
def main(config: DictConfig):
    """
    SAM-MOT 통합 파이프라인 실행 스크립트
    
    Args:
        config (DictConfig): Hydra 설정
    """
    log.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    log.info(f"Working directory: {os.getcwd()}")
    
    # 데이터 준비 스크립트 실행
    log.info("Preparing sample video data...")
    os.system(f"python {os.path.join(project_root, 'src/data/prepare_data.py')}")
    
    # 메인 파이프라인 실행
    log.info("Running SAM-MOT pipeline...")
    os.system(f"python {os.path.join(project_root, 'src/sam_mot_pipeline.py')}")
    
    log.info("Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
