from langchain.document_loaders import PyPDFLoader
import API_KEYS
import requests
from geopy.distance import geodesic
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# PDF 파일 로드 및 청크 분할
documents = PyPDFLoader('seoul.pdf').load()
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs
docs = split_docs(documents)

# 임베딩 생성 및 Chroma 벡터 저장
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=API_KEYS.OPENAI_API_KEY)
db = Chroma.from_documents(docs, embeddings, persist_directory=".")

# OpenAI Chat 모델 및 Q&A 체인 설정
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name, api_key=API_KEYS.OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# 사용자 선호도 및 위치
user_preferences = {
    "culturalTourism": 0.8,
    "historicalTourism": 0.5,
    "shopping": 0.3,
    "leisureSports": 0.5,
    "restaurant": 0.1,
    "natureTourism": 0.4
}
user_position = (37.552987017, 126.972591728)

# Google Places API 기반 주변 장소 검색 함수
def search_nearby_places(lat, lon, user_preferences):
    category_mapping = {
        "culturalTourism": "museum|art_gallery",
        "historicalTourism": "church|historical_place",
        "shopping": "shopping_mall",
        "leisureSports": "gym|stadium",
        "restaurant": "restaurant|cafe",
        "natureTourism": "park"
    }
    results = []

    for preference, weight in user_preferences.items():
        if weight > 0.5:
            types = category_mapping.get(preference, "")
            url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=3000&type={types}&key={API_KEYS.GOOGLE_MAPS_API_KEY}"
            response = requests.get(url).json()

            # API 결과 처리 및 필터링
            if response.get("status") == "OK":
                for place in response.get("results", []):
                    place_data = {
                        "name": place["name"],
                        "lat": place["geometry"]["location"]["lat"],
                        "lon": place["geometry"]["location"]["lng"],
                        "type": preference,
                        "distance": geodesic(user_position, (place["geometry"]["location"]["lat"], place["geometry"]["location"]["lng"])).km
                    }
                    results.append(place_data)
    # 거리 기준으로 정렬
    results = sorted(results, key=lambda x: x['distance'])
    return results

# Google Places API 결과를 사용해 최종 추천 수행
def recommend_spots():
    nearby_places = search_nearby_places(user_position[0], user_position[1], user_preferences)
    matching_docs = db.similarity_search("가까운 장소에 대한 정보", k=10)
    recommendations = []

    # Google API 결과와 유사성 문서 결합
    for place in nearby_places[:5]:  # 상위 5개 장소 선택
        for doc in matching_docs:
            if place['name'] in doc.page_content:
                place['description'] = chain.run(input_documents=[doc], question=f"{place['name']}에 대한 설명을 해주세요.")
                recommendations.append(place)
                break

    return recommendations

# 추천 결과 출력
recommendations = recommend_spots()
for place in recommendations:
    print(f"Name: {place['name']}, Distance: {place['distance']}km, Description: {place['description']}")
