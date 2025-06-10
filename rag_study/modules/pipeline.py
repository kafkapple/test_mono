from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_openai import ChatOpenAI
import logging
import tiktoken
from modules.splitter import TextSplitter
from modules.memory import ConversationMemory
import sys
from modules.vectordb import VectorStore

#
class RAGPipeline:
    """RAG 검색 체인 파이프라인"""
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.memory = ConversationMemory(max_history=5)

    def _debug_print_prompt_components(self, query: str, context_docs, prompt_template):
        self.logger.info("\n=== 프롬프트 구성 요소 ===")
        self.logger.info(f"1. 질의:\n{query}")
        
        self.logger.info("\n2. 검색된 컨텍스트:")
        for i, doc in enumerate(context_docs, 1):
            self.logger.info(f"\n문서 {i}:")
            self.logger.info(f"내용: {doc.page_content[:200]}...")
            self.logger.info(f"메타데이터: {doc.metadata}")
            
        self.logger.info("\n3. 프롬프트 템플릿:")
        self.logger.info(prompt_template.template)
        
        self.logger.info("\n4. 이전 대화 이력:")
        self.logger.info(self.memory.get_chat_history())

    def run(self, query: str):
        # 1️⃣ 문서 로딩 & 분할
        from modules.loader import DocumentLoader
        loader = DocumentLoader(self.cfg.data.file_path)
        docs = loader.load()
        splitter = TextSplitter(splitter_type=self.cfg.data.splitter_type,
                                chunk_size=self.cfg.data.chunk_size, 
                                chunk_overlap=self.cfg.data.chunk_overlap)
        split_docs = splitter.split(docs)

        # 2️⃣ FAISS 벡터 DB 저장
        vector_store = VectorStore(self.cfg)
        vector_db = vector_store.create_or_load(split_docs)
        retriever = vector_db.as_retriever(
            search_type=self.cfg.retriever.search_type, 
            search_kwargs={"k": self.cfg.retriever.top_k}
        )

        # 3️⃣ LLM 선택
        llm = ChatOpenAI(model_name=self.cfg.model.name,
                         temperature=self.cfg.model.temperature,
                         max_tokens=self.cfg.model.max_tokens)

        # 4️⃣ 검색 및 프롬프트 생성
        context_docs = retriever.get_relevant_documents(query)
        prompt_template = hub.pull("rlm/rag-prompt")
        
        # 디버그 정보 출력
        self._debug_print_prompt_components(query, context_docs, prompt_template)
        
        # 5️⃣ 체인 실행
        chain = (
            {
                "context": lambda x: context_docs,
                "question": lambda x: x,
                "chat_history": lambda x: self.memory.get_chat_history()
            }
            | prompt_template
            | llm
            | StrOutputParser()
        )

        # 6️⃣ 응답 생성 및 저장
        response = chain.invoke(query)
        self.memory.add_interaction(query, response)
        self._track_tokens(response)
        
        return response

    def _track_tokens(self, response):
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(response))
        # 이모지 제거하고 일반 텍스트로 변경
        self.logger.info(f"생성된 응답 토큰 수: {token_count}")
