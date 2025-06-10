import whisper
import os

def transcribe_and_save(audio_path, model_size="base"):
    # Whisper 모델 로드
    model = whisper.load_model(model_size) # tiny, base, small, medium, large 중 선택

    # 음성 파일 변환(STT) 및 자막 정보 포함
    result = model.transcribe(audio_path, task="transcribe", verbose=True, language=None, word_timestamps=False) # 언어 자동 감지(기본값)

    # 텍스트 저장
    txt_path = os.path.splitext(audio_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"텍스트 결과 저장: {txt_path}")

    # SRT 자막 저장
    srt_path = os.path.splitext(audio_path)[0] + ".srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], 1):
            # SRT 타임코드 포맷팅
            def format_timestamp(seconds):
                ms = int((seconds % 1) * 1000)
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                return f"{h:02}:{m:02}:{s:02},{ms:03}"

            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    print(f"SRT 자막 저장: {srt_path}")

# 사용 예시
audio_file = "연속 잠재 공간에서 LLM 추론_Coconut_ToT.mp3"  # 또는 "예시파일.mp3"
transcribe_and_save(audio_file, model_size="base")  # 모델 크기: tiny, base, small, medium, large 중 선택
