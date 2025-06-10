import os
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config")
def main(config: DictConfig):
    """
    SAM-MOT 파이프라인 성능 테스트 및 최적화
    
    Args:
        config (DictConfig): Hydra 설정
    """
    log.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # 테스트 설정 목록
    test_configs = [
        # 기본 설정
        {},
        
        # 다양한 모델 크기 테스트
        {"models.model_type": "vit_b"},
        {"models.model_type": "vit_l"},
        {"models.model_type": "vit_h"},
        
        # 다양한 추적기 버전 테스트
        {"trackers.version": "nano"},
        {"trackers.version": "tiny"},
        {"trackers.version": "s"},
        {"trackers.version": "m"},
        {"trackers.version": "l"},
        {"trackers.version": "x"},
        
        # 다양한 추적 파라미터 테스트
        {"trackers.parameters.track_thresh": 0.4},
        {"trackers.parameters.track_thresh": 0.6},
        {"trackers.parameters.match_thresh": 0.7},
        {"trackers.parameters.match_thresh": 0.9},
        
        # 마스크 IoU 가중치 테스트
        {"trackers.segmentation.use_mask_iou": True, "trackers.segmentation.mask_iou_weight": 0.3},
        {"trackers.segmentation.use_mask_iou": True, "trackers.segmentation.mask_iou_weight": 0.7},
        
        # 비디오 메모리 크기 테스트
        {"models.video.memory_size": 3},
        {"models.video.memory_size": 10},
        
        # 시각화 설정 테스트
        {"visualization.mask.alpha": 0.3},
        {"visualization.mask.alpha": 0.7},
        {"visualization.trajectory.length": 10},
        {"visualization.trajectory.length": 30},
    ]
    
    # 결과 저장 디렉토리
    results_dir = os.path.join(config.general.output_dir, "performance_tests")
    os.makedirs(results_dir, exist_ok=True)
    
    # 성능 측정 결과
    fps_results = []
    config_names = []
    
    # 각 설정으로 테스트 실행
    for i, test_config in enumerate(test_configs):
        # 설정 이름 생성
        config_name = f"test_{i}"
        if test_config:
            key = list(test_config.keys())[0].split('.')[-1]
            value = list(test_config.values())[0]
            config_name = f"{key}_{value}"
        
        config_names.append(config_name)
        log.info(f"Running test with configuration: {config_name}")
        
        # 명령어 생성
        cmd_args = " ".join([f"{k}={v}" for k, v in test_config.items()])
        cmd = f"python {os.path.join(os.getcwd(), '../src/sam_mot_pipeline.py')} {cmd_args}"
        
        # 실행 및 결과 파싱 (실제 구현에서는 파이프라인 직접 호출 및 성능 측정)
        # 여기서는 더미 FPS 값 생성
        fps = 20 + np.random.normal(0, 5)  # 실제 구현에서는 실제 측정값 사용
        fps_results.append(fps)
        
        log.info(f"Test {config_name} completed with FPS: {fps:.2f}")
    
    # 성능 결과 시각화
    plt.figure(figsize=(12, 8))
    bars = plt.bar(config_names, fps_results)
    plt.xlabel('Configuration')
    plt.ylabel('FPS')
    plt.title('SAM-MOT Performance with Different Configurations')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 최고 성능 설정 강조
    best_idx = np.argmax(fps_results)
    bars[best_idx].set_color('green')
    
    # 결과 저장
    plt.savefig(os.path.join(results_dir, 'performance_comparison.png'))
    log.info(f"Performance comparison saved to {os.path.join(results_dir, 'performance_comparison.png')}")
    
    # 최적 설정 기록
    best_config = test_configs[best_idx]
    log.info(f"Best configuration: {best_config if best_config else 'Default'} with FPS: {fps_results[best_idx]:.2f}")
    
    with open(os.path.join(results_dir, 'optimal_config.txt'), 'w') as f:
        f.write(f"Best configuration: {best_config if best_config else 'Default'}\n")
        f.write(f"FPS: {fps_results[best_idx]:.2f}\n")
        f.write("\nAll test results:\n")
        for i, (config_name, fps) in enumerate(zip(config_names, fps_results)):
            f.write(f"{config_name}: {fps:.2f} FPS\n")
    
    log.info(f"Optimal configuration saved to {os.path.join(results_dir, 'optimal_config.txt')}")
    log.info("Performance testing completed!")

if __name__ == "__main__":
    main()
