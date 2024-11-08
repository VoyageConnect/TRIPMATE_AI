"""
from langchain.document_loaders import TextLoader

import API_KEYS

documents = TextLoader('file.txt').load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

# 문서를 청크로 분할
def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

# docs 변수에 분할 문서를 저장
docs = split_docs(documents)

#OpenAI의 임베딩 모델 사용
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=API_KEYS.OPENAI_API_KEY)

# Chromdb에 벡터 저장, 저장 장소는 d:/data
from langchain.vectorstores import Chroma
db = Chroma.from_documents(docs, embeddings, persist_directory=".")

from langchain.chat_models import ChatOpenAI
model_name = "gpt-3.5-turbo"  #GPT-3.5-turbo 모델 사용
llm = ChatOpenAI(model_name=model_name, api_key=API_KEYS.OPENAI_API_KEY)

# Q&A 체인을 사용하여 쿼리에 대한 답변 얻기
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

# 쿼리를 작성하고 유사성 검색을 수행하여 답변을 생성,따라서 txt에 있는 내용을 질의해야 합니다
query = "AI란?"
matching_docs = db.similarity_search(query)
answer =  chain.run(input_documents=matching_docs, question=query)

print(answer)
"""

from langchain.document_loaders import PyPDFLoader
import API_KEYS

# PDF 파일을 로드
documents = PyPDFLoader('seoul.pdf').load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

# 문서를 청크로 분할
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# 분할된 문서를 docs 변수에 저장
docs = split_docs(documents)

# OpenAI 임베딩 모델 사용
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=API_KEYS.OPENAI_API_KEY)

# Chroma에 벡터 저장, 저장 장소는 현재 디렉토리로 설정
from langchain.vectorstores import Chroma
db = Chroma.from_documents(docs, embeddings, persist_directory=".")

# ChatOpenAI를 통해 LLM 생성
from langchain.chat_models import ChatOpenAI
model_name = "gpt-3.5-turbo"  # GPT-3.5-turbo 모델 사용
llm = ChatOpenAI(model_name=model_name, api_key=API_KEYS.OPENAI_API_KEY)

# Q&A 체인 로드
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# 쿼리 작성 후 유사성 검색으로 답변 생성

user_preferences = {
            "culturalTourism": 0.8,
            "historicalTourism": 0.5,
            "shopping": 0.3,
            "leisureSports": 0.5,
            "restaurant": 0.1,
            "natureTourism": 0.4
}

user_position = (37.552987017, 126.972591728)



query = (f"너는 나의 여행 플래너야. 아래 문항에 대해 확실하지 않은 정보가 있다면 주지마."
         f"나는 문화관광에 {user_preferences['culturalTourism']}만큼,"
         f"역사관광에 {user_preferences['historicalTourism']}만큼,"
         f"쇼핑에 {user_preferences['shopping']}만큼,"
         f"레저스포츠와 체험에 {user_preferences['leisureSports']}만큼,"
         f"음식에 {user_preferences['restaurant']} 만큼, 그리고"
         f"자연관광에 {user_preferences['natureTourism']} 만큼 관심이 있어" 
         f"현재 위치는 ({user_position[0]},{user_position[1]})인데, 가까이 있는 어딜 가보면 좋을까? 5개의 항목을 추천해주고 이유도 같이 설명해줘.")

matching_docs = db.similarity_search(query)
answer = chain.run(input_documents=matching_docs, question=query)

print(answer)
