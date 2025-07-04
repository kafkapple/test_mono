import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/generative-language.retriever']

def load_creds():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds
# 서비스 계정 인증 (주로 Vertex AI에서 사용)

# 서비스 계정 생성 및 키 발급: Google Cloud Console에서 서비스 계정을 만들고, JSON 키 파일을 다운로드합니다.

# 환경 변수 설정:
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/service-account-key.json"

# Vertex AI SDK 사용:
import vertexai
from vertexai.preview.generative_models import GenerativeModel

vertexai.init(project="YOUR_PROJECT_ID", location="us-central1")
model = GenerativeModel("gemini-1.0-pro")
response = model.generate_content("AI가 어떻게 작동하는지 설명하세요")
print(response.text)
