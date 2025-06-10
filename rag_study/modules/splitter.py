from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter

class TextSplitter:
    """문서 분할 (Recursive / Token 기반 선택 가능)"""
    
    def __init__(self, chunk_size=500, chunk_overlap=50, splitter_type="recursive"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter_type = splitter_type

    def split(self, documents):
        if self.splitter_type == "recursive":
            return RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap).split_documents(documents)
        elif self.splitter_type == "token":
            return TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap).split_documents(documents)
        else:
            raise ValueError("지원되지 않는 분할 방식")
