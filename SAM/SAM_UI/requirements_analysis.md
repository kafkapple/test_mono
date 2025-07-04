# SAM-MOT 객체 추적 프로젝트 요구사항 분석 및 알고리즘 선정

## 1. 요구사항 분석

사용자의 요청에 따라 다음과 같은 요구사항을 파악했습니다:

1. SAM2, Grounding SAM과 같은 픽셀 수준 세그멘테이션 알고리즘 활용
2. MOT(Multiple Object Tracking) 알고리즘과의 결합
3. 영상에서 n개의 객체를 추적하는 기능 구현
4. 샘플 영상 데이터 및 결과 시각화 저장 코드 제공
5. Hydra 프레임워크를 통한 설정 관리

## 2. 알고리즘 조사 및 선정

### 2.1 세그멘테이션 알고리즘

#### 2.1.1 SAM2 (Segment Anything Model 2)

- **공식 소개**: [https://ai.meta.com/sam2/](https://ai.meta.com/sam2/)
- **GitHub**: [https://github.com/facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
- **논문**: [https://arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714)

**주요 특징**:
- 이미지와 비디오 모두에서 프롬프트 기반 세그멘테이션 지원
- 스트리밍 아키텍처로 비디오 프레임을 실시간으로 처리
- 메모리 모듈을 통해 비디오에서 객체가 일시적으로 사라져도 추적 가능
- 마스크 예측에 대한 추가 프롬프트 기반 수정 지원
- 다양한 입력 프롬프트(점, 박스 등) 지원

**장점**:
- 비디오 세그멘테이션에 최적화된 최신 모델
- 객체 추적 기능이 내장되어 있어 MOT 알고리즘과 결합 시 시너지 효과
- Meta AI에서 개발한 공식 모델로 지속적인 업데이트 및 지원 기대

**단점**:
- 비교적 높은 컴퓨팅 리소스 요구
- 최신 모델이라 참고 자료가 상대적으로 적음

#### 2.1.2 Grounding SAM

- **GitHub**: [https://github.com/IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- **논문**: [https://arxiv.org/abs/2401.14159](https://arxiv.org/abs/2401.14159)

**주요 특징**:
- Grounding DINO와 SAM을 결합한 모델
- 텍스트 프롬프트 기반 객체 검출 및 세그멘테이션
- 복잡한 텍스트 설명(예: "분홍색 옷을 입은 사람")으로도 객체 세그멘테이션 가능
- Grounded SAM 2는 비디오에서의 객체 추적 기능 추가

**장점**:
- 텍스트 프롬프트만으로 객체 검출 및 세그멘테이션 가능
- 다양한 응용 사례와 데모 제공
- 활발한 커뮤니티 지원

**단점**:
- 여러 모델의 조합으로 설치 및 설정이 복잡할 수 있음
- SAM2보다 비디오 처리에 최적화되지 않을 수 있음

### 2.2 MOT(Multiple Object Tracking) 알고리즘

#### 2.2.1 ByteTrack

- **GitHub**: [https://github.com/FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack)
- **논문**: [https://arxiv.org/abs/2110.06864](https://arxiv.org/abs/2110.06864)

**주요 특징**:
- 모든 검출 박스를 연관시켜 추적하는 방식
- 낮은 점수의 검출 박스도 활용하여 가려진 객체나 단편화된 궤적 문제 해결
- MOT17, MOT20 데이터셋에서 SOTA 성능 달성
- 다양한 크기의 모델 제공(nano, tiny, s, m, l, x)

**장점**:
- 높은 정확도(MOTA, IDF1, HOTA 지표)
- 빠른 처리 속도(V100 GPU에서 30 FPS)
- 다양한 객체 탐지기와 호환 가능
- 간단하고 효과적인 구현

**단점**:
- 기본적으로 바운딩 박스 기반 추적으로, 픽셀 수준 세그멘테이션과 통합 필요

#### 2.2.2 SORT/DeepSORT

- **GitHub**: [https://github.com/abewley/sort](https://github.com/abewley/sort)
- **DeepSORT**: [https://github.com/nwojke/deep_sort](https://github.com/nwojke/deep_sort)

**주요 특징**:
- 칼만 필터와 헝가리안 알고리즘 기반 추적
- DeepSORT는 외관 특징을 추가하여 성능 향상
- 간단하고 빠른 구현

**장점**:
- 구현이 간단하고 이해하기 쉬움
- 계산 효율성이 높음
- 널리 사용되는 검증된 알고리즘

**단점**:
- ByteTrack보다 성능이 다소 낮을 수 있음
- 복잡한 가림 상황에서 ID 스위칭 문제 발생 가능

### 2.3 세그멘테이션과 MOT 결합 방식

1. **세그멘테이션 후 추적 방식**:
   - 각 프레임에서 세그멘테이션 수행
   - 세그멘테이션 마스크에서 바운딩 박스 추출
   - MOT 알고리즘으로 바운딩 박스 추적
   - 추적 ID를 세그멘테이션 마스크에 할당

2. **추적 후 세그멘테이션 방식**:
   - MOT 알고리즘으로 객체 추적
   - 추적된 바운딩 박스를 세그멘테이션 알고리즘의 프롬프트로 사용
   - 각 추적 ID에 해당하는 세그멘테이션 마스크 생성

3. **통합 방식**:
   - 세그멘테이션과 추적을 동시에 수행
   - 마스크 유사성과 공간적 위치를 모두 고려한 연관성 계산
   - SAM2의 내장 메모리 모듈 활용

## 3. 최종 알고리즘 선정

### 세그멘테이션 알고리즘: SAM2
- 비디오 세그멘테이션에 최적화된 최신 모델
- 내장된 메모리 모듈로 객체 추적 기능 제공
- 다양한 프롬프트 지원으로 유연한 사용 가능

### MOT 알고리즘: ByteTrack
- 높은 정확도와 빠른 처리 속도
- 다양한 크기의 모델 제공으로 유연한 선택 가능
- 활발한 커뮤니티 지원

### 결합 방식: 통합 방식
- SAM2의 비디오 세그멘테이션 및 메모리 모듈 활용
- ByteTrack의 연관성 계산 알고리즘 활용
- 마스크 유사성과 공간적 위치를 모두 고려한 추적

## 4. 구현 계획

1. SAM2 모델 설치 및 기본 세그멘테이션 파이프라인 구현
2. ByteTrack 설치 및 기본 추적 파이프라인 구현
3. 두 알고리즘의 통합 방식 구현
4. Hydra 프레임워크를 통한 설정 관리 구조 설계
5. 샘플 영상 데이터 준비 및 테스트
6. 결과 시각화 및 저장 코드 구현
7. 성능 최적화 및 문제점 개선

## 5. 참고 자료

- SAM2 공식 사이트: [https://ai.meta.com/sam2/](https://ai.meta.com/sam2/)
- SAM2 GitHub: [https://github.com/facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
- Grounded SAM GitHub: [https://github.com/IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- ByteTrack GitHub: [https://github.com/FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack)
- ByteTrack 논문: [https://arxiv.org/abs/2110.06864](https://arxiv.org/abs/2110.06864)
