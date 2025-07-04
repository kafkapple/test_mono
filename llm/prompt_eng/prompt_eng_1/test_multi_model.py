from litellm import completion, batch_completion_models
import os
import logging  # logging 모듈 임포트
from dotenv import load_dotenv
os.environ["VERTEXAI_PROJECT"] = "kafkapple" #"gen-lang-client-0245024645"
os.environ["VERTEXAI_LOCATION"] = "us-central1"

# 기본 로깅 설정 (파일 로깅 추가 및 UTF-8 인코딩 명시)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='test_multi_model.log',  # 로그 파일 이름 지정
    encoding='utf-8'  # 파일 인코딩을 UTF-8로 설정
)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["PERPLEXITY_API_KEY"] = os.getenv("PERPLEXITY_API_KEY")
messages = [{"role": "user", "content": "perplexity, google gemini, anthropic, openai, ollama 등 대표적인 방식 모두 통합해서 사용 가능한 파이썬 모듈 구현하려면? "}]
# response = completion(model="gpt-4o-mini", messages=messages)


# # Google 인증 설정 (Vertex AI)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\joon\AppData\Roaming\gcloud\application_default_credentials.json"  # [5][6]
# "d:/dev/client_secret.json" #

# 모델 이름에 provider 명시 및 전체 ID 사용 (anthropic/claude-3-haiku-20240307)
models = [ "vertex_ai/gemini-1.5-flash","perplexity/sonar", "anthropic/claude-3-haiku-20240307", "openai/gpt-3.5-turbo"]#, "claude-3-5-sonnet-20240620"]
# ["gpt-3.5-turbo", , "vertex_ai/gemini-1.0-pro"],  # [4][5]
# messages="정보 지식 습득 및 관리를 체계적으로 하는 법에 대해 설명해줘."
def test_single_model(models, messages, save_results=True, log_responses=True):
    print("\n--- 싱글 모델 순차 테스트 시작 ---")
    for model in models:
        try:
            print(f"\n-- 모델: {model} --")
            response = completion(model=model, messages=messages)
            content = response.choices[0].message.content
            print(content)
            if log_responses:
                logging.info(f"Model: {model}, Response: {content}")

            if save_results:
                results_dir = "results_single"
                os.makedirs(results_dir, exist_ok=True)
                file_path = os.path.join(results_dir, f"{model.replace('/', '_')}.txt")
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ {model} 응답 저장 완료: {file_path}")
                if log_responses: # 저장 로그는 log_responses가 True일 때만 기록
                    logging.info(f"Model: {model}, Response saved to {file_path}")
        except Exception as e:
            error_message = f"Error processing single model {model}: {e}"
            print(f"❌ {error_message}")
            if log_responses: # 에러 로그는 log_responses가 True일 때만 기록
                logging.error(error_message)


def test_multi_model(models, messages, save_results=True, log_responses=True):
    print("\n--- 멀티 모델 동시 테스트 시작 ---")
    # 동시에 여러 모델 호출
    # 모델 이름에 provider 명시 및 전체 ID 사용 (anthropic/claude-3-haiku-20240307)
    responses = batch_completion_models(
        models= models,
        messages=[{"role": "user", "content": messages}]
    )

    # responses 변수의 타입과 값 확인 (디버깅용)
    print(f"DEBUG: Type of responses: {type(responses)}")
    print(f"DEBUG: Value of responses: {responses}")

    # --- 결과 처리 수정: batch_completion_models가 단일 응답 객체를 반환한다고 가정 ---
    try:
        # response가 choices 속성을 가졌는지 확인
        if hasattr(responses, 'choices') and responses.choices and responses.choices[0].message:
            # 응답 객체에서 모델 이름 가져오기
            model_name_from_response = getattr(responses, 'model', 'unknown_model') # 안전하게 속성 접근
            content_to_write = responses.choices[0].message.content
            print(f"\n-- 응답 받은 모델 (배치): {model_name_from_response} --")

            if save_results:
                # 파일 저장 경로 생성
                results_dir = "results_batch"
                os.makedirs(results_dir, exist_ok=True)
                # 응답에서 얻은 모델 이름 사용
                file_path = os.path.join(results_dir, f"{model_name_from_response.replace('/', '_')}.txt")
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write(content_to_write)
                print(f"✅ {model_name_from_response} 응답 저장 완료: {file_path}")
                if log_responses:
                    logging.info(f"Model: {model_name_from_response}, Response saved to {file_path}")
            else:
                 print(f"-- 모델: {model_name_from_response} (저장 안 함) --\n{content_to_write[:100]}...") # 일부만 출력

        elif isinstance(responses, list) and responses:
             # 만약 리스트가 반환되었지만 예상 구조가 아닌 경우 (예: 오류 포함 리스트)
             print(f"⚠️ batch_completion_models가 리스트를 반환했지만 구조가 예상과 다릅니다. 첫 번째 항목 처리 시도:")
             first_element = responses[0]
             # 첫 번째 항목 처리 시도 (위 로직과 유사하게)
             if hasattr(first_element, 'choices') and first_element.choices and first_element.choices[0].message:
                 model_name_from_response = getattr(first_element, 'model', 'unknown_model')
                 content_to_write = first_element.choices[0].message.content
                 print(f"-- 리스트 첫 항목 모델: {model_name_from_response} --")
                 # 여기에 저장/로깅 로직 추가 가능 (위와 동일하게)
             else:
                 error_message = f"Unexpected list structure from batch_completion_models. First element: Type: {type(first_element)}, Value: {first_element}"
                 print(f"⚠️ {error_message}")
                 if log_responses:
                     logging.warning(error_message)
        else:
            # 기타 예상치 못한 반환값 또는 오류 구조 처리
            error_message = f"Unexpected return value from batch_completion_models. Type: {type(responses)}, Value: {responses}"
            print(f"⚠️ {error_message}")
            if log_responses:
                logging.warning(error_message)

    except Exception as e:
        # batch_completion_models 호출 자체 또는 위 처리 로직에서 오류 발생 시
        error_message = f"Error during batch completion or processing: {e}"
        print(f"❌ {error_message}")
        if log_responses:
            logging.error(error_message)
    # --- 결과 처리 수정 끝 ---

    # 기존 zip 루프 제거 (주석 처리)
    # # 결과 저장 및 비교 - zip 사용
    # # 모델 이름에 provider 명시 및 전체 ID 사용 (anthropic/claude-3-haiku-20240307)
    # for model, response in zip(models, responses):
    #     try:
    #         # response가 choices 속성을 가졌는지 확인 (오류 응답 처리)
    #         if hasattr(response, 'choices') and response.choices and response.choices[0].message:
    #             content_to_write = response.choices[0].message.content
    #             # ... (이하 기존 저장/로깅 로직) ...
    #         else:
    #             # 오류 응답 또는 예상치 못한 구조 처리
    #             # response의 실제 타입과 값 로깅 추가
    #             error_message = f"Unexpected response structure or error for model {model}. Type: {type(response)}, Value: {response}"
    #             print(f"⚠️ {error_message}")
    #             if log_responses: # 경고 로그는 log_responses가 True일 때만 기록
    #                 logging.warning(error_message)
    #     except Exception as e:
    #         # response 객체 자체 접근 불가 등 예외 처리
    #         error_message = f"Error processing response for model {model}: {e}"
    #         print(f"❌ {error_message}")
    #         if log_responses: # 에러 로그는 log_responses가 True일 때만 기록
    #             logging.error(error_message)

if __name__ == "__main__":
    # 예시: 싱글 모델은 저장/로깅하고, 멀티 모델은 로깅만 수행
    test_single_model(models, messages, save_results=True, log_responses=True)
    test_multi_model(models, messages[0]['content'], save_results=False, log_responses=True)
    
    # 예시: 둘 다 저장 안 함
    # test_single_model(models, messages[0]['content'], save_results=False, log_responses=False)
    # test_multi_model(models, messages[0]['content'], save_results=False, log_responses=False)
