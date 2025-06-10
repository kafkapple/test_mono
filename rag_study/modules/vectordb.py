from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import os
import logging
from typing import List, Optional

class VectorStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.embedding_model = OpenAIEmbeddings()
        self.vector_db = None
        self.save_path = self.cfg.vectordb.save_path

    def create_or_load(self, documents: List[Document]) -> FAISS:
        """벡터 DB 생성 또는 로드 및 업데이트"""
        if os.path.exists(self.save_path):
            logging.info(f"기존 벡터 DB 로드 중: {self.save_path}")
            self.vector_db = FAISS.load_local(self.save_path, self.embedding_model)
            
            # 새로운 문서와 기존 문서 비교
            existing_docs = self._get_all_documents()
            new_docs = self._filter_new_documents(documents, existing_docs)
            
            if new_docs:
                logging.info(f"새로운 문서 {len(new_docs)}개 추가 중...")
                self.vector_db.add_documents(new_docs)
                self.vector_db.save_local(self.save_path)
            else:
                logging.info("추가할 새로운 문서가 없습니다.")
        else:
            logging.info("새로운 벡터 DB 생성 중...")
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.vector_db = FAISS.from_documents(documents, self.embedding_model)
            self.vector_db.save_local(self.save_path)

        if self.cfg.vectordb.debug:
            self._print_debug_info()

        return self.vector_db

    def _get_all_documents(self) -> List[Document]:
        """현재 벡터 DB의 모든 문서 반환"""
        if not self.vector_db:
            return []
        # docstore 접근 방식 수정
        return list(self.vector_db.docstore._dict.values())

    def _filter_new_documents(self, new_docs: List[Document], existing_docs: List[Document]) -> List[Document]:
        """새로운 문서만 필터링"""
        existing_contents = {doc.page_content for doc in existing_docs}
        return [doc for doc in new_docs if doc.page_content not in existing_contents]

    def _print_debug_info(self):
        """벡터 DB 디버그 정보 출력"""
        docs = self._get_all_documents()
        logging.info("\n=== 벡터 DB 현황 ===")
        logging.info(f"총 청크 수: {len(docs)}")
        
        for i, doc in enumerate(docs, 1):
            logging.info(f"\n청크 {i}:")
            logging.info(f"내용: {doc.page_content[:200]}...")  # 처음 200자만 출력
            logging.info(f"메타데이터: {doc.metadata}")
            
        # 벡터 DB 상태 추가 정보
        if hasattr(self.vector_db, 'index'):
            logging.info(f"\n벡터 차원: {self.vector_db.index.d}")
            logging.info(f"총 벡터 수: {self.vector_db.index.ntotal}") 