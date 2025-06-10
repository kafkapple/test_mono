from TTS.api import TTS
from TTS.utils.manage import ModelManager
manager = ModelManager()
print(manager.list_models())

# 최신 XTTS v2 다국어/음성 클로닝 모델 로드
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# 기본 합성 (한국어 예시)
tts.tts_to_file(
    text="안녕하세요, 이것은 최신 XTTS v2 모델의 예시입니다.",
    file_path="output_ko.wav",
    speaker_wav=None,  # 화자 음성 클로닝이 필요하면 6초 이상 음성 파일 경로 입력
    language="ko"
)

# 영어 예시
tts.tts_to_file(
    text="Hello, this is an example of the latest XTTS v2 model.",
    file_path="output_en.wav",
    language="en"
)
#pip install TTS==0.22.0
#pytorch 2.0

