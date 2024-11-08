import requests
import json
import os
from geopy.distance import geodesic
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import openai
import wikipedia

import API_KEYS

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)  # 모든 출처에서의 요청 허용

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Change to logging.INFO or logging.ERROR in production
logger = logging.getLogger(__name__)

# OpenAI and Google API keys
OPENAI_API_KEY = API_KEYS.OPENAI_API_KEY # 환경 변수에서 API 키를 가져옵니다.
GOOGLE_PLACES_API_KEY = API_KEYS.GOOGLE_MAPS_API_KEY  # 환경 변수에서 Google Places API 키를 가져옵니다.

# PDF Loading and Text Splitting
pdf_path = 'seoul.pdf'
if not os.path.exists(pdf_path):
    logger.error(f"{pdf_path} file not found.")
    raise FileNotFoundError(f"{pdf_path} file not found.")

pdf_loader = PyPDFLoader(pdf_path)
pdf_docs = pdf_loader.load()

# Split text and prepare FAISS vector store
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = []
for doc in pdf_docs:
    if hasattr(doc, 'page_content') and isinstance(doc.page_content, str):
        texts.extend(text_splitter.split_text(doc.page_content))
    else:
        logger.warning("Document does not have valid page content.")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_texts(texts, embeddings)

# Set up Wikipedia API
wikipedia.set_lang('ko')


# RAG-based description generation function
def generate_description_with_rag(name, place_type):
    try:
        relevant_docs = vectorstore.similarity_search(f"{name} {place_type}")
        qa_chain = load_qa_chain(
            llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7),
            chain_type="stuff"
        )

        prompt = PromptTemplate(
            input_variables=["name", "place_type"],
            template=f"당신은 유능한 관광지 전문가입니다. '{name}'({place_type})에 대해 두 문장 이내로, 그곳에 가고싶게 설명해주세요. 정보가 없으면, 고객을 절대 속이지 말고 'False'만 반환하세요."
        )

        formatted_prompt = prompt.format(name=name, place_type=place_type)
        response = qa_chain.run(input_documents=relevant_docs, question=formatted_prompt)

        logger.debug(f"Generated description for {name}: {response}")
        return response
    except openai.APIStatusError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return f"OpenAI API error: {str(e)}"


# Google Places API function
def search_nearby_places(lat, lon, user_preferences):
    results = []
    category_mapping = {
        "culturalTourism": ["museum", "art_gallery"],
        "historicalTourism": ["church", "historical_place"],
        "shopping": ["shopping_mall"],
        "leisureSports": ["gym", "stadium"],
        "restaurant": ["restaurant", "cafe"],
        "natureTourism": ["park"]
    }
    excluded_types = {"lodging", "hotel", "motel"}

    for preference, types in user_preferences.items():
        if types > 0.5:
            for place_type in category_mapping.get(preference, []):
                url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=1500&type={place_type}&language=ko&key={GOOGLE_PLACES_API_KEY}"

                # Logging the request URL for debugging
                logger.debug(f"Request URL for {place_type}: {url}")

                # Send the request
                response = requests.get(url)

                # Log response status and content
                logger.debug(f"Response status: {response.status_code}")
                logger.debug(f"Response content: {response.text}")

                if response.status_code != 200:
                    logger.error(f"Google Places API returned an error for {place_type}: {response.status_code}")
                    continue

                data = response.json()
                if data.get("status") != "OK":
                    logger.warning(f"Google Places API returned status {data.get('status')} for {place_type}")
                    continue

                for place in data.get('results', []):
                    if any(ptype in excluded_types for ptype in place.get("types", [])):
                        continue
                    if place.get('user_ratings_total', 0) >= 15:
                        results.append({
                            'name': place['name'],
                            'type': place_type,
                            'distance': geodesic((lat, lon), (
                                place['geometry']['location']['lat'],
                                place['geometry']['location']['lng']
                            )).kilometers,
                            'source': 'Google',
                            'place_id': place['place_id']
                        })
    return results


# Recommendation function
def recommend_spots(user_location, user_preferences, max_distance=2, top_n=5):
    google_places = search_nearby_places(user_location[0], user_location[1], user_preferences)
    logger.info(f"Google Places Results: {google_places}")

    recommendations = sorted(google_places, key=lambda x: x.get('distance'))

    for place in recommendations[:top_n]:
        description = generate_description_with_rag(place['name'], place['type'])
        if description == "False":
            try:
                candidate_description = wikipedia.page(place['name'])
                description = candidate_description.summary + " (from Wikipedia)"
            except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
                description = "정보가 업데이트 중입니다!"
        place['description'] = description
        logger.debug(f"Final description for {place['name']}: {description}")

    return recommendations[:top_n]


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        logger.debug("Received a new request for recommendation.")

        # JSON 데이터 수신 확인
        data = request.get_json()
        if data is None:
            logger.error("No JSON data received.")
            return jsonify({"error": "No JSON data received."}), 400
        logger.debug(f"Request Data: {data}")

        # 사용자 위치 데이터 확인
        user_latitude = data.get('latitude')
        user_longitude = data.get('longitude')
        if not user_latitude or not user_longitude:
            logger.error("User location (latitude/longitude) is missing.")
            return jsonify({"error": "User location is missing."}), 400

        # 사용자 위치 및 기본 사용자 선호도 설정
        user_location = (user_latitude, user_longitude)
        user_location = (37.715133, 126.734086)
        user_preferences = data.get('user_preferences', {
            "culturalTourism": 0.2,
            "historicalTourism": 0.5,
            "shopping": 0.3,
            "leisureSports": 0.5,
            "restaurant": 0.1,
            "natureTourism": 0.7
        })
        logger.debug(f"User Location: {user_location}")
        logger.debug(f"User Preferences: {user_preferences}")

        # 추천 장소 생성 함수 호출
        recommended_spots = recommend_spots(user_location, user_preferences)
        logger.info(f"Recommended Spots: {recommended_spots}")

        # 추천 결과 반환
        return jsonify({"results": recommended_spots})

    except Exception as e:
        logger.error(f"Error occurred during recommendation processing: {str(e)}")
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=False, port=5000)
