from langchain.document_loaders import PyPDFLoader, TextLoader

class DocumentLoader:
    """PDF 또는 TXT 파일 로드"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        if self.file_path.endswith(".pdf"):
            return PyPDFLoader(self.file_path).load()
        elif self.file_path.endswith(".txt"):
            return TextLoader(self.file_path).load()
        else:
            raise ValueError("지원되지 않는 파일 형식")
