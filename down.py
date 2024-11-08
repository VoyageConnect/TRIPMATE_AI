from flask import Flask, request, jsonify
import requests
import json
from geopy.distance import geodesic
from flask_cors import CORS
import API_KEYS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # CORS 설정

# API 키 설정
OPENAI_API_KEY = API_KEYS.OPENAI_API_KEY
GOOGLE_PLACES_API_KEY = API_KEYS.GOOGLE_MAPS_API_KEY

# OSM 데이터 로드
with open('responses_with_descriptions.json', 'r', encoding='utf-8') as f:
    osm_data = json.load(f)


# ChatGPT를 사용한 설명 생성 함수
def generate_description_with_chatgpt(name, place_type):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "당신은 유능한 관광지 전문가입니다."},
            {"role": "user", "content": f"'{name}'({place_type})에 대해 두 문장 이내로, 그곳에 가고싶게 설명해주세요."}
        ],
        "max_tokens": 100
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        response_data = response.json()
        description = response_data.get('choices', [{}])[0].get('message', {}).get('content',
                                                                                   "Description unavailable.")
        return description
    else:
        return "Description unavailable."


# Google Places API를 통한 장소 검색
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
                url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=3000&type={place_type}&key={GOOGLE_PLACES_API_KEY}"

                # Google Places API 호출
                response = requests.get(url).json()

                # API 호출 결과 처리
                if response.get("status") == "OK":
                    for place in response.get('results', []):
                        if any(ptype in excluded_types for ptype in place.get("types", [])):
                            continue

                        results.append({
                            'name': place['name'],
                            'type': place_type,
                            'distance': geodesic((lat, lon), (
                                place['geometry']['location']['lat'], place['geometry']['location']['lng'])).kilometers,
                            'source': 'Google',
                            'place_id': place['place_id']
                        })
    return results


# 추천 함수
def recommend_spots(user_location, user_preferences, max_distance=2, top_n=5):
    google_places = search_nearby_places(user_location[0], user_location[1], user_preferences)

    # 추천 목록 통합 후 정렬
    recommendations = sorted(google_places, key=lambda x: (x['distance']))

    # 상위 top_n 추천 장소에 대해서만 description 생성
    for place in recommendations[:top_n]:
        place['description'] = generate_description_with_chatgpt(place['name'], place['type'])

    return recommendations[:top_n]


# 추천 API 엔드포인트
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()  # 요청 데이터 받기

        user_latitude = data.get('latitude', 37.496486)  # 기본값
        user_longitude = data.get('longitude', 127.028361)  # 기본값

        user_location = (user_latitude, user_longitude)  # 튜플로 변환

        user_preferences = {
            "culturalTourism": 0.8,
            "historicalTourism": 0.5,
            "shopping": 0.3,
            "leisureSports": 0.5,
            "restaurant": 0.1,
            "natureTourism": 0.4
        }

        # 추천 실행
        recommended_spots = recommend_spots(user_location, user_preferences)

        # JSON 형식으로 응답 반환
        return jsonify({"results": recommended_spots})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run()
